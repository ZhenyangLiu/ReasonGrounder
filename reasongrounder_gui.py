# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import torch
import cv2
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature, render_with_depth
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA

# from scene.gaussian_model import GaussianModel
from scene import Scene, GaussianModel, FeatureGaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from scipy.spatial.transform import Rotation as R

# from cuml.cluster.hdbscan import HDBSCAN
from hdbscan import HDBSCAN
import pdb
import random
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from torchvision.transforms import ToPILImage

# from model.LISA import LISAForCausalLM
# from model.llava import conversation as conversation_lib
# from model.llava.mm_utils import tokenizer_image_token
# from model.segment_anything.utils.transforms import ResizeLongestSide


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
import torch.nn.functional as F
from torch import nn
from skimage import measure
from sklearn.preprocessing import QuantileTransformer

import json
import os
from clip_utils.clip_utils import load_clip
from clip_utils import get_relevancy_scores

os.environ["TOKENIZERS_PARALLELISM"] = "false"


PREFIXES_TEXT_PROMPT = "Output the name of the object and the reason for output the object in the following json format, {'object_name':, 'reason':} "

PREFIXES_TEXT_PROMPT_LISA = "Please output segmentation mask and explain why."


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class CONFIG:
    r = 2   # scale ratio
    window_width = int(2160/r)
    window_height = int(1200/r)

    width = int(2160/r)
    height = int(1200/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    FEATURE_DIM = 32
    MODEL_PATH = './output/horns' # 30000

    FEATURE_GAUSSIAN_ITERATION = 10000
    SCENE_GAUSSIAN_ITERATION = 30000

    CLIP_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/clip_gate.pt')
    PATCH_CLIP_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/patch_clip_gate.pt')
    SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
    FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])

def wash_mask(pred_mask_init):
    labels = measure.label(pred_mask_init, connectivity=1)
    properties = measure.regionprops(labels)

    largest_region = max(properties, key=lambda prop: prop.area)

    largest_region_mask_np = np.zeros_like(pred_mask_init, dtype=bool)
    largest_region_mask_np[labels == largest_region.label] = True

    return [largest_region_mask_np]

def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use 3D scale statistics to normalize scales -- use quantile transformer.
    """
    scales = scales.flatten()

    scales = scales.detach().cpu().numpy()

    # Calculate quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

    def quantile_transformer_func(scales):
        # This function acts as a wrapper for QuantileTransformer.
        # QuantileTransformer expects a numpy array, while we have a torch tensor.
        scales = scales.reshape(-1,1)
        return torch.Tensor(
            quantile_transformer.transform(scales.detach().cpu().numpy())
        ).to(scales.device)

    return quantile_transformer_func
    

class GaussianSplattingGUI:
    def __init__(self, args, opt, gaussian_model:GaussianModel, feature_gaussian_model:FeatureGaussianModel, scale_gate: torch.nn.modules.container.Sequential, clip_gate: torch.nn.modules.container.Sequential, patch_clip_gate: torch.nn.modules.container.Sequential) -> None:
        self.opt = opt
        self.args = args
        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg_feature = [0 for i in range(opt.FEATURE_DIM)]
        bg_feature = torch.tensor(bg_feature, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.bg_feature = bg_feature
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug

        self.engine = {
            'scene': gaussian_model,
            'feature': feature_gaussian_model,
            'scale_gate': scale_gate,
            'clip_gate': clip_gate,
            'patch_clip_gate' : patch_clip_gate
        }

        self.cluster_point_colors = None
        self.label_to_color = np.random.rand(1000, 3)

        self.max_re_mask = None
        self.max_re_mask_local = None

        self.seg_score = None
        self.seg_score_local = None

        self.proj_mat = None
        self.global_proj_mat = None
        self.patch_proj_mat = None

        self.point_clip_features = None
        self.point_instance_features = None
        self.target_points = None
        self.target_points_local = None        
        self.point_clip_mask = None

        self.load_model = False
        print("loading model file...")
        self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
        self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
        self.engine['scale_gate'].load_state_dict(torch.load(self.opt.SCALE_GATE_PATH))
        self.engine['clip_gate'].load_state_dict(torch.load(self.opt.CLIP_GATE_PATH))
        self.engine['patch_clip_gate'].load_state_dict(torch.load(self.opt.PATCH_CLIP_GATE_PATH))


        self.do_pca()   # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0
        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False     # choose multiple object 
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.clear_edit = False                 # clear all the click prompts
        self.roll_back = False
        self.preview = False    # binary segmentation mode
        self.segment3d_flag = False
        self.reload_flag = False        # reload the whole scene / point cloud
        self.object_seg_id = 0          # to store the segmented object with increasing index order (path at: ./)
        # self.cluster_in_3D_flag = False

        self.render_mode_rgb = False
        self.render_mode_filter = False
        self.render_mode_filter_local = False
        self.pca_mat = torch.load(os.path.join(self.args.image_root, 'pca_64.pt'))['proj_v'].float().cuda()
        self.pca_mat_patch = torch.load(os.path.join(self.args.image_root, 'pca_64.pt'))['patch_proj_v'].float().cuda()

        self.render_mode_similarity = False
        self.render_mode_similarity_clip = False

        self.render_mode_pca = False
        self.render_mode_pca_global = False
        self.render_mode_pca_patch = False

        self.render_mode_cluster = False

        self.save_flag = False
        self.input_text_prompt = ""
        self.input_query_if = ""
        self.input_query_text_if = ""

        scale_dir = os.path.join(self.args.image_root, 'mask_scales')
        scale_files = [f for f in os.listdir(scale_dir) if os.path.isfile(os.path.join(scale_dir, f))]
        # selected_files = random.sample(scale_files, min(10, len(scale_files)))
        all_scales = []
        for scale_file in scale_files:
            scale_path = os.path.join(scale_dir, scale_file)
            scale_data = torch.load(scale_path).detach()
            all_scales.append(scale_data)

        self.all_scales = torch.cat(all_scales, dim=0)



        self.scales = torch.tensor([0.5]).cuda()
        self.scales_local = torch.tensor([0.25]).cuda()

        self.clip_model = load_clip()
        self.clip_model.eval()

        self.llava_model_path = "/home/user/LZY/RLA/llava-v1.5-7b"
        self.llava_model_name=get_model_name_from_path(self.llava_model_path)
        self.llava_tokenizer, self.llava_model, self.llava_image_processor, self.llava_context_len = load_pretrained_model(
            model_path=self.llava_model_path,
            model_base=None,
            model_name=self.llava_model_name,
            load_4bit=True
        )

        self.object_name = []

    def format_text_output(self, text_output, characters_num=50):
        new_text_output = []
        i = 0
        for i in range(len(text_output) // characters_num):
            new_text_output.append(text_output[i * characters_num : (i + 1) * characters_num])
            new_text_output.append('\n')
        if len(text_output) % characters_num != 0 and len(text_output) // characters_num != 0:
            new_text_output.append(text_output[(i + 1) * characters_num :])
        else:
            new_text_output.append(text_output)
        return ''.join(new_text_output)
    
    def get_object_name_from_implicit_query(self, text_prompt, base_scene:torch.Tensor):
        prompt = PREFIXES_TEXT_PROMPT + text_prompt
        to_pil = ToPILImage()
        args = type('Args', (), {
            "model_name": self.llava_model_name,
            "tokenizer": self.llava_tokenizer,
            "model": self.llava_model,
            "image_processor": self.llava_image_processor,
            "query": prompt,
            "conv_mode": None,
            "images": [to_pil(base_scene.permute(2,0,1))],
            "temperature": 0.2,
            "top_p": 1.0,
            "max_new_tokens": 1024,
            "sep": ",",
        })()

        output = eval_model(args)
        object_name = eval(output)['object_name']
        self.object_name = f'{object_name}.'
        print(self.format_text_output(output))


    def part_determination(self, base_filtered_scene:torch.Tensor): 
        PREFIXES_TEXT_PROMPT_PART = f"I'm using a robotic arm to grab the {self.object_name}. Please determine the most suitable part of the {self.object_name} for the robotic arm to grasp, taking into account factors such as stability, ease of manipulation, and avoiding damage. Provide the response in the following JSON format: {{'part': '<part_to_grasp> of {self.object_name}', 'reason': '<reason_for_grasping_that_part>'}}. Ensure that the object name is included in the 'part' field. The reason should clearly explain why that specific part is ideal for grasping."
        prompt = PREFIXES_TEXT_PROMPT_PART
        to_pil = ToPILImage()
        args = type('Args', (), {
            "model_name": self.llava_model_name,
            "tokenizer": self.llava_tokenizer,
            "model": self.llava_model,
            "image_processor": self.llava_image_processor,
            "query": prompt,
            "conv_mode": None,
            "images": [to_pil(base_filtered_scene.permute(2, 0, 1))],
            "temperature": 0.2,
            "top_p": 1.0,
            "max_new_tokens": 1024,
            "sep": ",",
        })()
        output = eval_model(args)

        part_name = eval(output)['part']
        self.part_name = f'{part_name}'

        part_reason = eval(output)['reason']
        self.part_reason = f'{part_reason}.'


    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)
    
    def grayscale_to_colormap(self, gray):
        """Convert a grayscale value to Jet colormap RGB values."""
        # Ensure the grayscale values are in the range [0, 1]
        # gray = np.clip(gray, 0, 1)

        # Jet colormap ranges (these are normalized to [0, 1])
        jet_colormap = np.array([
            [0, 0, 0.5],
            [0, 0, 1],
            [0, 0.5, 1],
            [0, 1, 1],
            [0.5, 1, 0.5],
            [1, 1, 0],
            [1, 0.5, 0],
            [1, 0, 0],
            [0.5, 0, 0]
        ])

        # Corresponding positions for the colors in the colormap
        positions = np.linspace(0, 1, jet_colormap.shape[0])

        # Interpolate the RGB values based on the grayscale value
        r = np.interp(gray, positions, jet_colormap[:, 0])
        g = np.interp(gray, positions, jet_colormap[:, 1])
        b = np.interp(gray, positions, jet_colormap[:, 2])

        return np.stack((r, g, b), axis=-1)

    def register_dpg(self):
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width+300, height=self.window_height):
            dpg.add_image("_texture")   # add the texture

        dpg.set_primary_window("_primary_window", True)

        # def callback_depth(sender, app_data):
            # self.img_mode = (self.img_mode + 1) % 4
            
        # --- interactive mode switch --- #
        def clickmode_callback(sender):
            self.clickmode_button = 1 - self.clickmode_button
        def clickmode_multi_callback(sender):
            self.clickmode_multi_button = dpg.get_value(sender)
            print("clickmode_multi_button = ", self.clickmode_multi_button)
        def preview_callback(sender):
            self.preview = dpg.get_value(sender)
            # print("binary_threshold_button = ", self.binary_threshold_button)
        def clear_edit():
            self.clear_edit = True
        def roll_back():
            self.roll_back = True
        def callback_segment3d():
            self.segment3d_flag = True
        def callback_save():
            self.save_flag = True
        def callback_reload():
            self.reload_flag = True
        # def callback_cluster():
        #     self.cluster_in_3D_flag =True
        def callback_reshuffle_color():
            self.label_to_color = np.random.rand(1000, 3)
            try:
                self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]
                self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)
            except:
                pass

        def render_mode_rgb_callback(sender):
            self.render_mode_rgb = not self.render_mode_rgb
        def render_mode_filter_callback(sender):
            self.render_mode_filter = not self.render_mode_filter
        def render_mode_filter_local_callback(sender):
            self.render_mode_filter_local = not self.render_mode_filter_local
        def render_mode_similarity_callback(sender):
            self.render_mode_similarity = not self.render_mode_similarity
        def render_mode_similarity_clip_callback(sender):
            self.render_mode_similarity_clip = not self.render_mode_similarity_clip

        def render_mode_pca_callback(sender):
            self.render_mode_pca = not self.render_mode_pca

        def render_mode_pca_global_callback(sender):
            self.render_mode_pca_global = not self.render_mode_pca_global

        def render_mode_pca_patch_callback(sender):
            self.render_mode_pca_patch = not self.render_mode_pca_patch

        def render_mode_cluster_callback(sender):
            self.render_mode_cluster = not self.render_mode_cluster
        def on_submit(sender, app_data):
            self.input_text_prompt = dpg.get_value('text_prompt')
            # dpg.set_value('text_prompt', '')

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[self.window_width+10, 0]):

            dpg.add_text("Mouse position: click anywhere to start. ", tag="pos_item")
            dpg.add_slider_float(label="Scale", default_value=0.5,
                                 min_value=0.0, max_value=1.0, tag="_Scale")
            dpg.add_slider_float(label="ScoreThres", default_value=0.0,
                                 min_value=0.0, max_value=1.0, tag="_ScoreThres")
            dpg.add_text("text prompt: ")
            dpg.add_input_text(label='', width=200, default_value='',
                                tag='text_prompt')
            dpg.add_button(label="Submit", callback=on_submit)


            dpg.add_text("\nRender option: ", tag="render")
            dpg.add_checkbox(label="RGB", callback=render_mode_rgb_callback, user_data="Some Data")
            dpg.add_checkbox(label="Filter", callback=render_mode_filter_callback, user_data="Some Data")
            dpg.add_checkbox(label="Filter_local", callback=render_mode_filter_local_callback, user_data="Some Data")
            
            dpg.add_checkbox(label="CLIP_Glo", callback=render_mode_pca_global_callback, user_data="Some Data")
            dpg.add_checkbox(label="CLIP_Patch", callback=render_mode_pca_patch_callback, user_data="Some Data")

            dpg.add_checkbox(label="PCA", callback=render_mode_pca_callback, user_data="Some Data")
            dpg.add_checkbox(label="SIMILARITY", callback=render_mode_similarity_callback, user_data="Some Data")
            dpg.add_checkbox(label="RELEVANCY", callback=render_mode_similarity_clip_callback, user_data="Some Data")

            dpg.add_checkbox(label="3D CLUSTER", callback=render_mode_cluster_callback, user_data="Some Data")
            
            dpg.add_text("\nSegment option: ", tag="seg")
            dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data")
            dpg.add_checkbox(label="multi-clickmode", callback=clickmode_multi_callback, user_data="Some Data")
            dpg.add_checkbox(label="preview_segmentation_in_2d", callback=preview_callback, user_data="Some Data")
            
            dpg.add_text("\n")
            dpg.add_button(label="segment3d", callback=callback_segment3d, user_data="Some Data")
            dpg.add_button(label="roll_back", callback=roll_back, user_data="Some Data")
            dpg.add_button(label="clear", callback=clear_edit, user_data="Some Data")
            dpg.add_button(label="save as", callback=callback_save, user_data="Some Data")
            dpg.add_input_text(label="", default_value="precomputed_mask", tag="save_name")
            dpg.add_text("\n")

            def callback(sender, app_data, user_data):
                self.load_model = False
                file_data = app_data["selections"]
                file_names = []
                for key in file_data.keys():
                    file_names.append(key)

                self.opt.ply_file = file_data[file_names[0]]

                # if not self.load_model:
                print("loading model file...")
                self.engine.load_ply(self.opt.ply_file)
                self.do_pca()   # calculate new self.proj_mat after loading new .ply file
                print("loading model file done.")
                self.load_model = True

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")


        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        

        def toggle_moving_left():
            self.moving = not self.moving


        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle


        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*30, dy*30)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*20, dy*20)
                    self.update_camera = True
            
            self.mouse_pos = pos


        def change_pos(sender, app_data):
            # if not dpg.is_item_focused("_primary_window"):
            #     return
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            if self.input_text_prompt == "" and self.clickmode_button and app_data == 1:
                print(xy)
                self.new_click_xy = np.array(xy)
                self.new_click = True

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
            dpg.add_mouse_click_handler(callback=change_pos)
            
        dpg.create_viewport(title="RLA-Viewer", width=self.window_width+320, height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam
    
    def cluster_in_3D(self):
        point_features = self.engine['feature'].get_point_features
        scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2) * self.gates.unsqueeze(0)
        normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)
        sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]
        normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)
        clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False)
        cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
        self.cluster_centers = torch.zeros(len(np.unique(cluster_labels)), normed_sampled_point_features.shape[-1])
        for i in range(0, len(np.unique(cluster_labels))):
            self.cluster_centers[i] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i-1].mean(dim = 0), dim = -1)
        self.seg_score = torch.einsum('nc,bc->bn', self.cluster_centers.cpu(), normed_point_features.cpu())
        max_gs_values, _ = torch.max(self.seg_score, dim=1)
        self.max_re_mask = (max_gs_values > 0.85)
        self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]

    def cluster_in_3D_local(self, target_points):
        point_features = self.engine['feature'].get_point_features
        scale_conditioned_point_features = torch.nn.functional.normalize(point_features[target_points], dim = -1, p = 2) * self.gates_local.unsqueeze(0)
        normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)
        sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]
        normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)
        clusterer_local = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False)
        cluster_labels_local = clusterer_local.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
        self.cluster_centers_local = torch.zeros(len(np.unique(cluster_labels_local)), normed_sampled_point_features.shape[-1])
        for i in range(0, len(np.unique(cluster_labels_local))):
            self.cluster_centers_local[i] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels_local == i-1].mean(dim = 0), dim = -1)
        self.seg_score_local = torch.einsum('nc,bc->bn', self.cluster_centers_local.cpu(), normed_point_features.cpu())
        max_gs_values_local, _ = torch.max(self.seg_score_local, dim=1)
        self.max_re_mask_local = (max_gs_values_local > 0.85)
        # self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]



    def pca(self, X, n_components=3):
        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16, 
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        eigenvalues = torch.norm(eigenvalues.unsqueeze(1), dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]
        
        return proj_mat
    

    def do_pca(self):
        sems = self.engine['feature'].get_point_features.clone().squeeze()
        N, C = sems.shape
        torch.manual_seed(0)
        randint = torch.randint(0, N, [200_000])
        sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
        sem_chosen = sems[randint, :]
        self.proj_mat = self.pca(sem_chosen, n_components=3)

        global_sems = self.engine['clip_gate'](self.engine['feature'].get_point_features.clone().squeeze())
        global_N, global_C = global_sems.shape
        global_sems /= (torch.norm(global_sems, dim=1, keepdim=True) + 1e-6)
        global_sem_chosen = global_sems[randint, :]
        self.global_proj_mat = self.pca(global_sem_chosen, n_components=3)

        patch_sems = self.engine['patch_clip_gate'](self.engine['feature'].get_point_features.clone().squeeze())
        patch_N, patch_C = patch_sems.shape
        patch_sems /= (torch.norm(patch_sems, dim=1, keepdim=True) + 1e-6)
        patch_sem_chosen = patch_sems[randint, :]
        self.patch_proj_mat = self.pca(patch_sem_chosen, n_components=3)

        print("project mat initialized !")

    def preprocess(self,
        img,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024,
    ) -> torch.Tensor:

        img = (img - pixel_mean) / pixel_std
        # Pad
        h, w = img.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        img = F.pad(img, (0, padw, 0, padh))
        return img

    def generate_2dmask(self, text_prompt, rgb, local=False):
        rgb = torch.clip(rgb, 0, 1)
        image_np = np.array(rgb.cpu())
        uint8_image = (image_np * 255).astype(np.uint8)

        box_image, mask_image, pred_mask = self.grounded_sam(uint8_image, [f'{text_prompt}.'], pil=False)
        pred_mask = pred_mask.astype(bool) 

        rgb_image = cv2.cvtColor(uint8_image, cv2.COLOR_BGR2RGB)

        save_img = rgb_image.copy()
        save_img[pred_mask] = (
                rgb_image * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask]
        if local == False:
            cv2.imwrite("./mask_filed.jpg", save_img)
        else:
            cv2.imwrite("./mask_filed_local.jpg", save_img)
        return [pred_mask]


    def generate_grid_index(self, depth):
        h, w = depth.shape
        grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack(grid, dim=-1)
        return grid

    def get_scale(self, view_camera, pred_masks):
        depth_outputs = render_with_depth(view_camera, self.engine['scene'], self.opt, self.bg_color)
        depth = depth_outputs['depth']
        depth = depth.cpu().squeeze()
        grid_index = self.generate_grid_index(depth)
        points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu()
        points_in_3D[:,:,-1] = depth
        cx = depth.shape[1] / 2
        cy = depth.shape[0] / 2
        fx = cx / np.tan(view_camera.FoVx / 2)
        fy = cy / np.tan(view_camera.FoVy / 2)
        points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
        points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy

        pred_masks = torch.tensor(pred_masks).to(dtype=torch.float32)
        upsampled_mask = torch.nn.functional.interpolate(pred_masks.unsqueeze(1), mode = 'bilinear', size = (depth.shape[0], depth.shape[1]), align_corners = False)

        eroded_masks = torch.conv2d(
            upsampled_mask.float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze() 

        if len(pred_masks) == 1:
            eroded_masks = eroded_masks.unsqueeze(0) # (num_masks, H, W)

        scale = torch.zeros(len(pred_masks))
        indices = torch.zeros(len(pred_masks),2)

        for mask_id in range(len(pred_masks)):         
            indice = torch.nonzero(pred_masks[mask_id], as_tuple=False)
            min_coords = indice.min(dim=0).values
            max_coords = indice.max(dim=0).values
            center_point = (min_coords + max_coords) / 2.0
            distances = torch.norm(indice - center_point, dim=1)
            closest_point_index = distances.argmin()
            indices[mask_id] = indice[closest_point_index]   
            # indices[mask_id] = torch.round(center_point).int()
            point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]
            scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()
        
        return indices, scale


    @torch.no_grad()
    def fetch_data(self, view_camera):
        scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color)
        feature_outputs = render_contrastive_feature(view_camera, self.engine['feature'], self.opt, self.bg_feature)
        self.rendered_cluster = None if self.cluster_point_colors is None else render(view_camera, self.engine['scene'], self.opt, self.bg_color, override_color=torch.from_numpy(self.cluster_point_colors).cuda().float())["render"].permute(1, 2, 0)
        # --- RGB image --- #
        img = scene_outputs["render"].permute(1, 2, 0)  #

        rgb_score = img.clone()
        depth_score = rgb_score.cpu().numpy().reshape(-1)

        self.gates = self.engine['scale_gate'](torch.tensor([1.]).cuda())

        if self.input_text_prompt != self.input_query_text_if:
            self.input_query_text_if = self.input_text_prompt
            self.get_object_name_from_implicit_query(self.input_text_prompt, img)
            image_feature_m = self.engine['clip_gate'](feature_outputs["render"].permute(1, 2, 0))
            image_feature_m /= torch.norm(image_feature_m, dim = -1, keepdim = True)
            image_feature = image_feature_m.reshape(-1, image_feature_m.shape[-1])
            image_feature_query = torch.matmul(image_feature, self.pca_mat.t())
            relevancy_map = get_relevancy_scores(self.clip_model, image_feature_query, self.object_name).reshape(image_feature_m.shape[0], image_feature_m.shape[1])
            max_relevancy, max_index = torch.max(relevancy_map.view(-1), dim=0)
            row = max_index // relevancy_map.size(1) 
            col = max_index % relevancy_map.size(1)
            indices = (col.item(), row.item())              
            self.new_click_xy = np.array(indices)
            self.new_click = True

        # --- semantic image --- #
        sems = feature_outputs["render"].permute(1, 2, 0)
        H, W, C = sems.shape

        sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)

        sems_clip = self.engine['clip_gate'](feature_outputs["render"].permute(1, 2, 0))
        sem_transed_global = sems_clip @ (self.global_proj_mat.real.to(torch.float32))
        sem_transed_rgb_global = torch.clip(sem_transed_global*0.5+0.5, 0, 1)

        patch_sems_clip = self.engine['patch_clip_gate'](feature_outputs["render"].permute(1, 2, 0))
        sem_transed_patch = patch_sems_clip @ (self.patch_proj_mat.real.to(torch.float32))
        sem_transed_rgb_patch = torch.clip(sem_transed_patch*0.5+0.5, 0, 1)

        scale = dpg.get_value('_Scale')
        self.gates_vis = self.engine['scale_gate'](torch.tensor([scale]).cuda())

        scale_gated_feat = (feature_outputs["render"].permute(1, 2, 0)) * self.gates_vis.unsqueeze(0).unsqueeze(0)
        scale_gated_feat = torch.nn.functional.normalize(scale_gated_feat, dim = -1, p = 2)
        
        sem_transed = scale_gated_feat @ (self.proj_mat.real.to(torch.float32))
        sem_transed_rgb = torch.clip(sem_transed*0.5+0.5, 0, 1)

        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            self.prompt_num = 0
            try:
                self.engine['scene'].clear_segment()
                self.engine['feature'].clear_segment()
            except:
                pass

        if self.roll_back:
            self.new_click_xy = []
            self.roll_back = False
            self.prompt_num = 0
            # try:
            self.engine['scene'].roll_back()
            self.engine['feature'].roll_back()
            # except:
                # pass
        
        if self.reload_flag:
            self.reload_flag = False
            print("loading model file...")
            self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
            self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
            self.engine['scale_gate'].load_state_dict(torch.load(self.opt.SCALE_GATE_PATH))
            self.engine['clip_gate'].load_state_dict(torch.load(self.opt.CLIP_GATE_PATH))
            self.engine['patch_clip_gate'].load_state_dict(torch.load(self.opt.PATCH_CLIP_GATE_PATH))

            self.do_pca()   # calculate self.proj_mat
            self.load_model = True

        score_map = None
        if len(self.new_click_xy) > 0:
            featmap = scale_gated_feat.reshape(H, W, -1)
            if self.new_click and self.input_text_prompt != self.input_query_if:
                self.input_query_if = self.input_text_prompt
                xy = self.new_click_xy.squeeze()   
                print(xy)             
                new_feat = featmap[int(xy[1])%H, int(xy[0])%W, :].reshape(featmap.shape[-1], -1)
                if (self.prompt_num == 0) or (self.clickmode_multi_button == False):
                    self.chosen_feature = new_feat
                else:
                    self.chosen_feature = torch.cat([self.chosen_feature, new_feat], dim=-1)    # extend to get more prompt features

                self.prompt_num += 1
                self.new_click = False
                
                self.cluster_in_3D()
                similarity_scores = torch.einsum('nc,cn->n', self.cluster_centers.cpu(), self.chosen_feature.cpu())
                self.closest_cluster_idx = similarity_scores.argmax().item()
                self.target_points = (self.seg_score.argmax(dim = -1) == self.closest_cluster_idx) & self.max_re_mask

                base_filtered_scene = None if self.target_points is None else render(view_camera, self.engine['scene'], self.opt, self.bg_color, filed_cluster_points=self.target_points.cuda())["render_filed"].permute(1, 2, 0)
                base_filtered_scene_feature = None if self.target_points is None else render_contrastive_feature(view_camera, self.engine['feature'], self.opt, self.bg_feature, filed_cluster_points=self.target_points.cuda())["render"].permute(1, 2, 0)
                
                self.part_determination(base_filtered_scene)
                text_prompt_query = f'{self.part_name} of {self.object_name}'
                base_filtered_scene_feature_patch_m = self.engine['patch_clip_gate'](base_filtered_scene_feature)
                base_filtered_scene_feature_patch_m /= torch.norm(base_filtered_scene_feature_patch_m, dim = -1, keepdim = True)
                base_filtered_scene_feature_patch = base_filtered_scene_feature_patch_m.reshape(-1, base_filtered_scene_feature_patch_m.shape[-1])
                filtered_image_feature_query = torch.matmul(base_filtered_scene_feature_patch, self.pca_mat_patch.t())
                filtered_relevancy_map = get_relevancy_scores(self.clip_model, filtered_image_feature_query, text_prompt_query).reshape(base_filtered_scene_feature_patch_m.shape[0], base_filtered_scene_feature_patch_m.shape[1])
                filtered_max_relevancy, filtered_max_index = torch.max(filtered_relevancy_map.view(-1), dim=0)
                filtered_row = filtered_max_index // filtered_relevancy_map.size(1) 
                filtered_col = filtered_max_index % filtered_relevancy_map.size(1)
                filtered_indices = (filtered_col.item(), filtered_row.item())

                self.gates_local = self.engine['scale_gate'](torch.tensor([0.]).cuda())
                print(self.part_reason)

                self.cluster_in_3D_local(self.target_points)
                scale_gated_filtered_feature = base_filtered_scene_feature * self.gates_local.unsqueeze(0).unsqueeze(0)
                scale_gated_filtered_feature = torch.nn.functional.normalize(scale_gated_filtered_feature, dim = -1, p = 2)

                xy_local = (np.array(filtered_indices)).squeeze()
                featmap_local = scale_gated_filtered_feature.reshape(H, W, -1)
                new_feat_local = featmap_local[int(xy_local[1])%H, int(xy_local[0])%W, :].reshape(featmap_local.shape[-1], -1)

                similarity_scores_local = torch.einsum('nc,cn->n', self.cluster_centers_local.cpu(), new_feat_local.cpu())
                self.closest_cluster_idx_local = similarity_scores_local.argmax().item()
                self.target_points_local = (self.seg_score_local.argmax(dim = -1) == self.closest_cluster_idx_local) & self.max_re_mask_local

            if self.target_points is None:
                filtered_scene = None
            else:
                filed_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color, filed_cluster_points=self.target_points.cuda(), filed_cluster_points_local=self.target_points_local.cuda())
                filtered_scene = filed_outputs["render_filed"].permute(1, 2, 0)
                filtered_scene_local = filed_outputs["render_filed_local"].permute(1, 2, 0)

            image_feature_m = self.engine['clip_gate'](feature_outputs["render"].permute(1, 2, 0))
            image_feature_m /= (torch.norm(image_feature_m, dim=-1, keepdim=True) + 1e-6)
            image_feature = image_feature_m.reshape(-1, image_feature_m.shape[-1])
            image_feature_query = torch.matmul(image_feature, self.pca_mat.t())
            relevancy_map = get_relevancy_scores(self.clip_model, image_feature_query, self.input_text_prompt).reshape(image_feature_m.shape[0], image_feature_m.shape[1]).unsqueeze(-1)

            score_map = featmap @ self.chosen_feature
            score_map = (score_map + 1.0) / 2

            score_binary = score_map > dpg.get_value('_ScoreThres')
            
            score_map[~score_binary] = 0.0
            score_map = torch.max(score_map, dim=-1).values
            score_norm = (score_map - dpg.get_value('_ScoreThres')) / (1 - dpg.get_value('_ScoreThres'))

            if self.preview:
                rgb_score = img * torch.max(score_binary, dim=-1, keepdim=True).values    # option: binary
            else:
                rgb_score = img
            depth_score = 1 - torch.clip(score_norm, 0, 1)
            depth_score = depth2img(depth_score.cpu().numpy()).astype(np.float32)/255.0

            if self.segment3d_flag:
                """ gaussian point cloud core params
                self.engine._xyz            # (N, 3)
                self.engine._features_dc    # (N, 1, 3)
                self.engine._features_rest  # (N, 15, 3)
                self.engine._opacity        # (N, 1)
                self.engine._scaling        # (N, 3)
                self.engine._rotation       # (N, 4)
                self.engine._objects_dc     # (N, 1, 16)
                """
                self.segment3d_flag = False
                feat_pts = self.engine['feature'].get_point_features.squeeze()
                scale_gated_feat_pts = feat_pts * self.gates.unsqueeze(0)
                scale_gated_feat_pts = torch.nn.functional.normalize(scale_gated_feat_pts, dim = -1, p = 2)

                score_pts = scale_gated_feat_pts @ self.chosen_feature
                score_pts = (score_pts + 1.0) / 2
                self.score_pts_binary = (score_pts > dpg.get_value('_ScoreThres')).sum(1) > 0

                # save_path = "./debug_robot_{:0>3d}.ply".format(self.object_seg_id)
                # try:
                #     self.engine['scene'].roll_back()
                #     self.engine['feature'].roll_back()
                # except:
                #     pass
                self.engine['scene'].segment(self.score_pts_binary)
                self.engine['feature'].segment(self.score_pts_binary)

        if self.save_flag:
            print("Saving ...")
            self.save_flag = False
            try:
                os.makedirs("./segmentation_res", exist_ok=True)
                save_mask = self.engine['scene']._mask == self.engine['scene'].segment_times + 1
                torch.save(save_mask, f"./segmentation_res/{dpg.get_value('save_name')}.pt")
            except:
                with dpg.window(label="Tips"):
                    dpg.add_text('You should segment the 3D object before save it (click segment3d first).')

        self.render_buffer = None
        render_num = 0
        if self.render_mode_rgb or (not self.render_mode_pca_patch and not self.render_mode_pca_global and not self.render_mode_pca and not self.render_mode_cluster and not self.render_mode_similarity and not self.render_mode_similarity_clip):
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode_filter:
            self.render_buffer = filtered_scene.cpu().numpy().reshape(-1)
            render_num += 1

        if self.render_mode_filter_local:
            self.render_buffer = filtered_scene_local.cpu().numpy().reshape(-1)
            render_num += 1

        if self.render_mode_pca_global:
            self.render_buffer = sem_transed_rgb_global.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + sem_transed_rgb_global.cpu().numpy().reshape(-1)
            render_num += 1

        if self.render_mode_pca_patch:
            self.render_buffer = sem_transed_rgb_patch.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + sem_transed_rgb_patch.cpu().numpy().reshape(-1)
            render_num += 1

        if self.render_mode_pca:
            self.render_buffer = sem_transed_rgb.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + sem_transed_rgb.cpu().numpy().reshape(-1)
            render_num += 1

        
        if self.render_mode_cluster:
            if self.rendered_cluster is None:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)
            else:
                self.render_buffer = self.rendered_cluster.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + self.rendered_cluster.cpu().numpy().reshape(-1)
            
            render_num += 1
        if self.render_mode_similarity:
            if score_map is not None:
                self.render_buffer = self.grayscale_to_colormap(score_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32) if self.render_buffer is None else self.render_buffer + self.grayscale_to_colormap(score_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32)
            else:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)

        if self.render_mode_similarity_clip:
            if relevancy_map is not None:
                self.render_buffer = self.grayscale_to_colormap(relevancy_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32) if self.render_buffer is None else self.render_buffer + self.grayscale_to_colormap(relevancy_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32)
            else:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)


            render_num += 1
        self.render_buffer /= render_num

        dpg.set_value("_texture", self.render_buffer)



if __name__ == "__main__":
    parser = ArgumentParser(description="GUI option")

    parser.add_argument('-m', '--model_path', type=str, default="./output/figurines")
    parser.add_argument('-f', '--feature_iteration', type=int, default=10000)
    parser.add_argument('-s', '--scene_iteration', type=int, default=30000)
    parser.add_argument("--version", default="./LISA-13B")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="./clip-vit-large", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--image_root", default='/datasets/nerf_data/360_v2/garden/', type=str)

    args = parser.parse_args()

    opt = CONFIG()

    opt.MODEL_PATH = args.model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration

    opt.SCALE_GATE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
    opt.CLIP_GATE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/clip_gate.pt')
    opt.PATCH_CLIP_GATE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/patch_clip_gate.pt')
    opt.FEATURE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    opt.SCENE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

    gs_model = GaussianModel(opt.sh_degree)
    feat_gs_model = FeatureGaussianModel(opt.FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.FEATURE_DIM, bias=True),
        torch.nn.Sigmoid()
    ).cuda()

    clip_gate = torch.nn.Sequential(
        torch.nn.Linear(32, 64, bias=True),
        torch.nn.Sigmoid()
    ).cuda()

    patch_clip_gate = torch.nn.Sequential(
        torch.nn.Linear(32, 64, bias=True),
        torch.nn.Sigmoid()
    ).cuda()

    gui = GaussianSplattingGUI(args, opt, gs_model, feat_gs_model, scale_gate, clip_gate, patch_clip_gate)

    gui.render()
