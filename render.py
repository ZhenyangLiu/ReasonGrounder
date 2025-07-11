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

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
import torch.nn.functional as F
from torch import nn
from skimage import measure
from sklearn.preprocessing import QuantileTransformer

import json
import os
from clip_utils.clip_utils import load_clip
from clip_utils import get_relevancy_scores, get_scale_based_clip

import ast
import torchvision.transforms as transforms
import glob
from collections import defaultdict

PREFIXES_TEXT_PROMPT = "Output the name of the object and the reason for output the object in the following json format, {'object_name':, 'reason':} "

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


class CONFIG:
    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False
    debug = False

    FEATURE_DIM = 32
    MODEL_PATH = './output/horns' # 30000

    FEATURE_GAUSSIAN_ITERATION = 10000
    SCENE_GAUSSIAN_ITERATION = 30000

    CLIP_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/clip_gate.pt')

    SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')

    FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

class GaussianSplattingRender:
    def __init__(self, args, opt, gaussian_model:GaussianModel, feature_gaussian_model:FeatureGaussianModel, scale_gate: torch.nn.modules.container.Sequential, clip_gate: torch.nn.modules.container.Sequential) -> None:
        self.opt = opt
        self.args = args

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg_feature = [0 for i in range(opt.FEATURE_DIM)]
        bg_feature = torch.tensor(bg_feature, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.bg_feature = bg_feature

        self.engine = {
            'scene': gaussian_model,
            'feature': feature_gaussian_model,
            'scale_gate': scale_gate,
            'clip_gate': clip_gate
            }
        self.debug = opt.debug

        self.cluster_point_colors = None
        self.label_to_color = np.random.rand(1000, 3)

        self.max_re_mask = None
        self.max_re_mask_local = None

        self.seg_score = None
        self.seg_score_local = None

        self.proj_mat = None
        self.clip_proj_mat = None

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


        self.do_pca()   # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")
        self.pca_mat = torch.load(os.path.join(self.args.image_root, 'pca_64.pt'))['proj_v'].float().cuda()

        self.gates_global = self.engine['scale_gate'](torch.tensor([0.6]).cuda())
        self.gates_local = self.engine['scale_gate'](torch.tensor([0.]).cuda())

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
        self.object_name = []

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
    
    def grayscale_to_colormap(self, gray):
        # jet_colormap = np.array([
        #     [0, 0, 0.5],
        #     [0, 0, 1],
        #     [0, 0.5, 1],
        #     [0, 1, 1],
        #     [0.5, 1, 0.5],
        #     [1, 1, 0],
        #     [1, 0.5, 0],
        #     [1, 0, 0],
        #     [0.5, 0, 0]
        # ])
        jet_colormap = np.array([
            [0.19, 0.07, 0.23],   # 深紫色
            [0.07, 0.27, 0.52],   # 深蓝色
            [0.0, 0.42, 0.7],     # 蓝色
            [0.25, 0.63, 0.47],   # 绿色
            [0.55, 0.74, 0.09],   # 黄绿色
            [0.87, 0.82, 0.0],    # 黄色
            [0.97, 0.59, 0.0],    # 橙色
            [0.87, 0.29, 0.09],   # 红色
            [0.61, 0.0, 0.26],    # 深红色
        ])
        positions = np.linspace(0, 1, jet_colormap.shape[0])
        r = np.interp(gray, positions, jet_colormap[:, 0])
        g = np.interp(gray, positions, jet_colormap[:, 1])
        b = np.interp(gray, positions, jet_colormap[:, 2])
        return np.stack((r, g, b), axis=-1)


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
        self.clip_proj_mat = self.pca(global_sem_chosen, n_components=3)

        print("project mat initialized !")

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

    def cluster_in_3D(self):
        point_features = self.engine['feature'].get_point_features
        scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2) * self.gates_global.unsqueeze(0)
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

    def concrete_render_set(self, model_path, name, views, render_target, query='', frame_id=None, gt_folder=None):        
        if render_target == 'scene':
            render_path = os.path.join(model_path, name, f"{render_target}")
            makedirs(render_path, exist_ok=True)
            for idx, view in enumerate(tqdm(views, desc="Scene_Rendering progress")):
                res = render(view, self.engine['scene'], self.opt, self.bg_color)['render']
                torchvision.utils.save_image(res, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                
        if render_target == 'clip_feature':
            render_path_global = os.path.join(model_path, name, f"{render_target}", "clip_features")            
            makedirs(render_path_global, exist_ok=True)            
            for idx, view in enumerate(tqdm(views, desc="CLIP_Rendering progress")):
                res = render_contrastive_feature(view, self.engine['feature'], self.opt, self.bg_feature)['render']
                res_global_init = self.engine['clip_gate'](res.permute(1, 2, 0))
                # res_global_norm = (torch.norm(res_global_init, dim=-1, keepdim=True) + 1e-6)
                # res_global = res_global_init / res_global_norm
                res_global_rgb = res_global_init @ (self.clip_proj_mat.real.to(torch.float32))
                res_global_rgb = torch.clip(res_global_rgb*0.5+0.5, 0, 1).permute(2, 0, 1)
   
                torchvision.utils.save_image(res_global_rgb, os.path.join(render_path_global, '{0:05d}'.format(idx) + ".png"))

        if render_target == 'instance_feature':
            render_path_global = os.path.join(model_path, name, f"{render_target}", "global")
            render_path_local = os.path.join(model_path, name, f"{render_target}", "local")
            makedirs(render_path_global, exist_ok=True)            
            makedirs(render_path_local, exist_ok=True)   

            for idx, view in enumerate(tqdm(views, desc="Instance_feature_Rendering progress")):
                res = render_contrastive_feature(view, self.engine['feature'], self.opt, self.bg_feature)['render']
                # res_global = (res.permute(1, 2, 0)) * self.gates_global.unsqueeze(0).unsqueeze(0)
                # res_local = (res.permute(1, 2, 0)) * self.gates_local.unsqueeze(0).unsqueeze(0)

                res_global_init = (res.permute(1, 2, 0)) * self.gates_global.unsqueeze(0).unsqueeze(0)
                res_local_init = (res.permute(1, 2, 0)) * self.gates_local.unsqueeze(0).unsqueeze(0)
        
                res_global_norm = (torch.norm(res_global_init, dim=-1, keepdim=True) + 1e-6)
                res_local_norm = (torch.norm(res_local_init, dim=-1, keepdim=True) + 1e-6)

                res_global = res_global_init / res_global_norm
                res_local = res_local_init / res_local_norm

                res_global_rgb = res_global @ (self.proj_mat.real.to(torch.float32))
                res_local_rgb = res_local @ (self.proj_mat.real.to(torch.float32))


                res_global_rgb = torch.clip(res_global_rgb*0.5+0.5, 0, 1).permute(2, 0, 1)
                res_local_rgb = torch.clip(res_local_rgb*0.5+0.5, 0, 1).permute(2, 0, 1)
                
                torchvision.utils.save_image(res_global_rgb, os.path.join(render_path_global, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(res_local_rgb, os.path.join(render_path_local, '{0:05d}'.format(idx) + ".png"))

        if render_target == "3DVQA":
            # self.input_text_prompt = input("Please enter your request: ")
            if query != '':
                self.input_text_prompt = query
            else:
                self.input_text_prompt = input("Please enter your request: ")
                self.object_name = f'{self.input_text_prompt}.'
            
            # sorted_views = sorted(views, key=lambda cam: len(cam.mask_scales), reverse=True)
            selected_view = views[1]
            # selected_view = max(views, key=lambda cam: cam.camera_center[2])

            q_trans = get_quantile_func(selected_view.mask_scales, "uniform")
            
            render_path = os.path.join(model_path, name, f"{render_target}")
            makedirs(render_path, exist_ok=True)
            relevancy_render_path = os.path.join(model_path, name, f"{render_target}", 'relevancy')
            makedirs(relevancy_render_path, exist_ok=True)

            img = render(selected_view, self.engine['scene'], self.opt, self.bg_color)['render']
            torchvision.utils.save_image(img, os.path.join(relevancy_render_path, 'relevancy_rgb' + ".png"))               
            
            init_feature = render_contrastive_feature(selected_view, self.engine['feature'], self.opt, self.bg_feature)['render'].permute(1,2,0)
            H, W, C = init_feature.shape

            opt_scale = get_scale_based_clip(self.clip_model, selected_view.original_features, selected_view.mask_scales, self.object_name, self.pca_mat)
            sampled_scale = q_trans(opt_scale).squeeze()
            qa_scale_gate = self.engine['scale_gate'](torch.tensor([sampled_scale]).cuda())
            
            clip_feature_global_init = self.engine['clip_gate'](init_feature)
            clip_feature_global = clip_feature_global_init.reshape(-1, clip_feature_global_init.shape[-1])
            clip_feature_global_query = torch.matmul(clip_feature_global, self.pca_mat.t())

            relevancy_map = get_relevancy_scores(self.clip_model, clip_feature_global_query, self.object_name).reshape(clip_feature_global_init.shape[0], clip_feature_global_init.shape[1])
            relevancy_map_rgb = self.grayscale_to_colormap(relevancy_map.squeeze().cpu().numpy()).astype(np.float32)
            torchvision.utils.save_image(torch.tensor(relevancy_map_rgb).cuda().permute(2,0,1), os.path.join(relevancy_render_path, 'relevancy' + ".png"))
                
            max_relevancy, max_index = torch.max(relevancy_map.view(-1), dim=0)
            row = max_index // relevancy_map.size(1) 
            col = max_index % relevancy_map.size(1)
            xy = np.array((col.item(), row.item())).squeeze()
            
            instance_feature_global = (init_feature) * qa_scale_gate.unsqueeze(0).unsqueeze(0)
            instance_feature_global /= (torch.norm(instance_feature_global, dim=-1, keepdim=True) + 1e-6)
            chosen_feature = instance_feature_global[int(xy[1])%H, int(xy[0])%W, :].reshape(instance_feature_global.shape[-1], -1)
            self.chosen_feature = chosen_feature.transpose(0, 1).unsqueeze(0).unsqueeze(0) 

            local_acc = 0
            total = 0
            s_iou = 0
            
            if frame_id == None:
                for idx, view in enumerate(tqdm(views, desc="3DVQA_Rendering progress")):
                    res = render_contrastive_feature(view, self.engine['feature'], self.opt, self.bg_feature)['render'].permute(1,2,0)
                    instance_res = res * qa_scale_gate.unsqueeze(0).unsqueeze(0)
                    instance_res_q = instance_res / (torch.norm(instance_res, dim=-1, keepdim=True) + 1e-6)
                    
                    similarity = F.cosine_similarity(instance_res_q, self.chosen_feature, dim=-1, eps=1e-8)
                    similarity_binary = similarity > 0.5
                    similarity[~similarity_binary] = 0.0
                    similarity_rgb = self.grayscale_to_colormap(similarity.squeeze().detach().cpu().numpy()).astype(np.float32) 
                    object_name = self.object_name.replace(".", '')
                    save_path =  os.path.join(render_path, f"{object_name}")
                    makedirs(save_path, exist_ok=True)
                    RGB = render(view, self.engine['scene'], self.opt, self.bg_color)['render'].permute(1,2,0).detach().cpu().numpy()
                    import pdb
                    pdb.set_trace()
                    mask = (similarity < 0.5).squeeze().detach().cpu().numpy()
                    similarity_rgb[mask, :] = RGB[mask, :] * 0.3
                    torchvision.utils.save_image(torch.tensor(similarity_rgb).cuda().permute(2,0,1), os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))
            else:
                for idx, frame in enumerate(frame_id):
                    res = render_contrastive_feature(views[int(frame)], self.engine['feature'], self.opt, self.bg_feature)['render'].permute(1,2,0)
                    instance_res = res * qa_scale_gate.unsqueeze(0).unsqueeze(0)
                    instance_res_q = instance_res / (torch.norm(instance_res, dim=-1, keepdim=True) + 1e-6)
                    
                    similarity = F.cosine_similarity(instance_res, self.chosen_feature, dim=-1, eps=1e-8)                                                 
                    similarity_binary = similarity > 0.5
                    similarity[~similarity_binary] = 0.0
                    
                    similarity_shape = (similarity.shape[1], similarity.shape[2])
                    transform = transforms.Compose([
                        transforms.Resize(similarity_shape), 
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor()
                    ])

                    gt_query_frame = os.path.join(gt_folder,str(frame).zfill(2), f"{query}.png")
                    if os.path.exists(gt_query_frame):
                        gt = Image.open(gt_query_frame).convert("RGB")
                        gt = transform(gt).permute(1,2,0)
                    
                    similarity_rgb = self.grayscale_to_colormap(similarity.squeeze().detach().cpu().numpy()).astype(np.float32) 
                    gt_rgb = self.grayscale_to_colormap(gt.squeeze().detach().cpu().numpy()).astype(np.float32) 

                    iou_path =  os.path.join(render_path, f"{query}")
                    os.makedirs(iou_path, exist_ok=True)

                    torchvision.utils.save_image(torch.tensor(similarity_rgb).cuda().permute(2,0,1), os.path.join(iou_path, '{0:05d}'.format(idx) + ".png"))
                    torchvision.utils.save_image(torch.tensor(gt_rgb).cuda().permute(2,0,1), os.path.join(iou_path, '{0:05d}'.format(idx) + "_gt.png"))
                    
                    similarity_bin = similarity > 0.5
                    gt_bin = gt.permute(2,0,1) > 0.
                    
                    intersection = (similarity_bin & gt_bin.cuda()).sum().float()
                    union = (similarity_bin | gt_bin.cuda()).sum().float() 
                    iou = intersection / union                    
                    
                    if iou > 0.5:
                        local_acc += 1

                    s_iou += iou
                    total += 1
                return local_acc, total, s_iou



        if render_target == "fine_filtered":
            # self.input_text_prompt = input("Please enter your request: ")
            self.input_text_prompt = 'Which item do children like best?'

            render_path_object = os.path.join(model_path, name, f"{render_target}", "object")
            render_path_part = os.path.join(model_path, name, f"{render_target}", "part")
            makedirs(render_path_object, exist_ok=True)
            makedirs(render_path_part, exist_ok=True)

            relevancy_render_path = os.path.join(model_path, name, f"{render_target}", 'relevancy')
            makedirs(relevancy_render_path, exist_ok=True)

            img = render(views[0], self.engine['scene'], self.opt, self.bg_color)['render']
            self.get_object_name_from_implicit_query(self.input_text_prompt, img)
                        
            init_feature = render_contrastive_feature(views[0], self.engine['feature'], self.opt, self.bg_feature)['render'].permute(1,2,0)
            H, W, C = init_feature.shape

            clip_feature_global_init = self.engine['clip_gate'](init_feature)     
            clip_feature_global = clip_feature_global_init.reshape(-1, clip_feature_global_init.shape[-1])
            clip_feature_global_query = torch.matmul(clip_feature_global, self.pca_mat.t())
            relevancy_map = get_relevancy_scores(self.clip_model, clip_feature_global_query, self.object_name).reshape(clip_feature_global_init.shape[0], clip_feature_global_init.shape[1])
            relevancy_map_object = self.grayscale_to_colormap(relevancy_map.squeeze().cpu().numpy()).astype(np.float32)
            torchvision.utils.save_image(torch.tensor(relevancy_map_object).cuda().permute(2,0,1), os.path.join(relevancy_render_path, 'object_relevancy' + ".png"))
                        
            max_relevancy, max_index = torch.max(relevancy_map.view(-1), dim=0)
            row = max_index // relevancy_map.size(1) 
            col = max_index % relevancy_map.size(1)
            xy = np.array((col.item(), row.item())).squeeze()

            instance_feature_global = (init_feature) * self.gates_global.unsqueeze(0).unsqueeze(0)
            # instance_feature_global /= (torch.norm(instance_feature_global, dim=-1, keepdim=True) + 1e-6)
            self.chosen_feature = instance_feature_global[int(xy[1])%H, int(xy[0])%W, :].reshape(instance_feature_global.shape[-1], -1)
            
            self.cluster_in_3D()
            similarity_scores = torch.einsum('nc,cn->n', self.cluster_centers.cpu(), self.chosen_feature.cpu())
            self.closest_cluster_idx = similarity_scores.argmax().item()
            self.target_points = (self.seg_score.argmax(dim = -1) == self.closest_cluster_idx) & self.max_re_mask
         
            base_filtered_scene = render(views[0], self.engine['scene'], self.opt, self.bg_color, filed_cluster_points=self.target_points.cuda())["render_filed"].permute(1, 2, 0)
            base_filtered_scene_feature = render_contrastive_feature(views[0], self.engine['feature'], self.opt, self.bg_feature, filed_cluster_points=self.target_points.cuda())["render"].permute(1, 2, 0)
                
            self.part_determination(base_filtered_scene)

            text_prompt_query = f'{self.part_name} of {self.object_name}'
            print(text_prompt_query)
            base_filtered_scene_feature_patch_m = self.engine['clip_gate'](base_filtered_scene_feature)
            # base_filtered_scene_feature_patch_m /= torch.norm(base_filtered_scene_feature_patch_m, dim = -1, keepdim = True)
            filtered_image_feature_query = base_filtered_scene_feature_patch_m.reshape(-1, base_filtered_scene_feature_patch_m.shape[-1])
            # filtered_image_feature_query = torch.matmul(base_filtered_scene_feature_patch, self.pca_mat.t())
            filtered_relevancy_map = get_relevancy_scores(self.clip_model, filtered_image_feature_query, text_prompt_query).reshape(base_filtered_scene_feature_patch_m.shape[0], base_filtered_scene_feature_patch_m.shape[1])
            relevancy_map_part = self.grayscale_to_colormap(filtered_relevancy_map.squeeze().cpu().numpy()).astype(np.float32)
            torchvision.utils.save_image(torch.tensor(relevancy_map_part).cuda().permute(2,0,1), os.path.join(relevancy_render_path, 'part_relevancy' + ".png"))
                        
            filtered_max_relevancy, filtered_max_index = torch.max(filtered_relevancy_map.view(-1), dim=0)
            filtered_row = filtered_max_index // filtered_relevancy_map.size(1) 
            filtered_col = filtered_max_index % filtered_relevancy_map.size(1)
            filtered_indices = (filtered_col.item(), filtered_row.item())                         
            print(self.part_reason)

            self.cluster_in_3D_local(self.target_points)

            scale_gated_filtered_feature = base_filtered_scene_feature * self.gates_local.unsqueeze(0).unsqueeze(0)
            # scale_gated_filtered_feature /= (torch.norm(scale_gated_filtered_feature, dim=-1, keepdim=True) + 1e-6)

            xy_local = (np.array(filtered_indices)).squeeze()
            featmap_local = scale_gated_filtered_feature.reshape(H, W, -1)
            new_feat_local = featmap_local[int(xy_local[1])%H, int(xy_local[0])%W, :].reshape(featmap_local.shape[-1], -1)
            similarity_scores_local = torch.einsum('nc,cn->n', self.cluster_centers_local.cpu(), new_feat_local.cpu())
            self.closest_cluster_idx_local = similarity_scores_local.argmax().item()
            self.target_points_local = (self.seg_score_local.argmax(dim = -1) == self.closest_cluster_idx_local) & self.max_re_mask_local
            # self.target_points_local = (self.seg_score_local.argmax(dim = -1) == self.closest_cluster_idx_local)


            for idx, view in enumerate(tqdm(views, desc="Fine_filtered_Rendering progress")):
                filed_outputs = render(view, self.engine['scene'], self.opt, self.bg_color, filed_cluster_points=self.target_points.cuda(), filed_cluster_points_local=self.target_points_local.cuda())
                filtered_scene = filed_outputs["render_filed"]
                filtered_scene_local = filed_outputs["render_filed_local"]
                
                torchvision.utils.save_image(filtered_scene, os.path.join(render_path_object, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(filtered_scene_local, os.path.join(render_path_part, '{0:05d}'.format(idx) + ".png"))

            
    def render_sets(self, dataset : ModelParams, iteration : int, skip_train : bool, skip_test : bool, target = 'scene', idx = 0, render_target='scene'):
        dataset.need_features = True
        dataset.need_masks = True
        scene = Scene(dataset, self.engine['scene'], self.engine['feature'], load_iteration=iteration, shuffle=False, mode='eval', target=target)
        if not skip_train:
            self.concrete_render_set(dataset.model_path, "train", scene.getTrainCameras(), render_target)
        # if not skip_test:
        #     self.concrete_render_set(dataset.model_path, "test", scene.getTestCameras(), render_target)

    def polygon_to_mask(self, img_shape, points_list):
        points = np.asarray(points_list, dtype=np.int32)
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        return mask

    def vis_mask_save(self, name, mask, save_path):
        mask_save = mask.copy()
        mask_save[mask == 1] = 255
        save_path_real = os.path.join(save_path, name)
        cv2.imwrite(str(save_path_real), mask_save)

    def stack_mask(self, mask_base, mask_add):
        mask = mask_base.copy()
        mask[mask_add != 0] = 1
        return mask

    def eval_gt_lerfdata(self, json_folder, ouput_path):
        gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
        img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
        gt_ann = {}
        for js_path in gt_json_paths:
            img_ann = defaultdict(dict)
            with open(js_path, 'r') as f:
                gt_data = json.load(f)
            
            h, w = gt_data['info']['height'], gt_data['info']['width']
            idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0])
            for prompt_data in gt_data["objects"]:
                label = prompt_data['category']
                box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
                mask = self.polygon_to_mask((h, w), prompt_data['segmentation'])
                if img_ann[label].get('mask', None) is not None:
                    mask = self.stack_mask(img_ann[label]['mask'], mask)
                    img_ann[label]['bboxes'] = np.concatenate(
                        [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
                else:
                    img_ann[label]['bboxes'] = box
                img_ann[label]['mask'] = mask
                
                # # save for visulsization
                save_path = os.path.join(ouput_path, 'gt' ,gt_data['info']['name'].split('.jpg')[0])
                os.makedirs(save_path, exist_ok=True)
                self.vis_mask_save(f'{label}.jpg', mask, save_path)
            gt_ann[f'{idx}'] = img_ann

        return gt_ann


    def metrics_calculate_lerf(self, dataset : ModelParams, iteration, json_folder='', output_folder = ''):
        gt_ann = self.eval_gt_lerfdata(json_folder, output_folder)
        dataset.need_features = True
        dataset.need_masks = True
        scene = Scene(dataset, self.engine['scene'], self.engine['feature'], load_iteration=iteration, shuffle=False, mode='eval', target='scene')
        frame_id = list(gt_ann.keys())
        for i, frame in enumerate(frame_id):
            query = list(gt_ann[frame].keys())
            total_acc = 0
            total_num = 0
            total_iou = 0
            
            for i, query in enumerate(query):
                gt_folder = os.path.join(output_folder, 'gt', f'frame_{int(frame):05d}')
                local_acc, num, iou = self.concrete_render_set(dataset.model_path, "train", scene.getTrainCameras(), '3DVQA', query, [frame], gt_folder)
                print(query)
                print("Local acc:", local_acc/num)
                print("Local iou:", iou/num)

                total_acc += local_acc
                total_num += num
                total_iou += iou


        # for j, idx in enumerate(tqdm(eval_index_list)):



    def metrics_calculate_3dovs(self, dataset : ModelParams, iteration, gt_folder=''):
        dataset.need_features = True
        dataset.need_masks = True
        scene = Scene(dataset, self.engine['scene'], self.engine['feature'], load_iteration=iteration, shuffle=False, mode='eval', target='scene')
        querys = []
        frame_id = []

        file_path = os.path.join(gt_folder, 'classes.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()
            querys.extend([line.strip() for line in lines]) 

        for folder_name in os.listdir(gt_folder):
            if folder_name.isdigit(): 
                frame_id.append(int(folder_name))

        total_acc = 0
        total_num = 0
        total_iou = 0
        for i, query in enumerate(querys):
            local_acc, num, iou = self.concrete_render_set(dataset.model_path, "train", scene.getTrainCameras(), '3DVQA', query, frame_id, gt_folder)
            print(query)
            print("Local acc:", local_acc/num)
            print("Local iou:", iou/num)

            total_acc += local_acc
            total_num += num
            total_iou += iou
        
        macc = total_acc / total_num
        miou = total_iou / total_num
        
        print('mAcc:' , macc)    
        print('mIou:' , miou)    
    
        import pdb
        pdb.set_trace()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--feature_iteration", type=int, default=10000)
    parser.add_argument("--scene_iteration", type=int, default=30000)
    
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--target', default='scene', const='scene', nargs='?', choices=['scene', 'clip_feature', 'instance_feature', '3DVQA', 'fine_filtered'])
    parser.add_argument('--idx', default=0, type=int)

    parser.add_argument("--image_root", default='./data/teatime', type=str)
    parser.add_argument("--lerf_label_root", default='./data/lerf_ovs/label', type=str)

    parser.add_argument('--render_target', default='scene', const='scene', nargs='?', choices=['scene', 'clip_feature', 'instance_feature', '3DVQA', 'fine_filtered'])
    parser.add_argument("--query", type=str, default='')
    parser.add_argument("--metrics", default='', type=str)



    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    opt = CONFIG()

    opt.MODEL_PATH = args.model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration

    opt.SCALE_GATE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
    opt.CLIP_GATE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/clip_gate.pt')

    opt.FEATURE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    opt.SCENE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

    gs_model = GaussianModel(opt.sh_degree)
    feat_gs_model = FeatureGaussianModel(opt.FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.FEATURE_DIM, bias=True),
        torch.nn.Sigmoid()
    ).cuda()

    clip_gate = torch.nn.Sequential(
        torch.nn.Linear(opt.FEATURE_DIM, 64, bias=True),
        torch.nn.ReLU(inplace=True),
    ).cuda()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    Render = GaussianSplattingRender(args, opt, gs_model, feat_gs_model, scale_gate, clip_gate)

    if args.metrics == '3dovs':
        Render.metrics_calculate_3dovs(model.extract(args), args.iteration, os.path.join(args.image_root, 'segmentations'))
    elif args.metrics == 'lerf':
        Render.metrics_calculate_lerf(model.extract(args), args.iteration, args.lerf_label_root, os.path.join(args.image_root, 'metric_vis'))
    else:
        Render.render_sets(model.extract(args), args.iteration, args.skip_train, args.skip_test, args.target, args.idx, args.render_target)