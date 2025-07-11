import os
import gc
import numpy as np
import torch
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm, trange
import cv2
from typing import Any, Dict, Generator,List

import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import maskclip_onnx



def pytorch_gc():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

def resize_image(img, longest_edge):
    # resize to have the longest edge equal to longest_edge
    width, height = img.size
    if width > height:
        ratio = longest_edge / width
    else:
        ratio = longest_edge / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return img.resize((new_width, new_height), Image.BILINEAR)

def interpolate_to_patch_size(img_bchw, patch_size):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W

def is_valid_image(filename):
    ext_test_flag = any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])
    is_file_flag = os.path.isfile(filename)
    return ext_test_flag and is_file_flag
    
def show_anns(anns):
    if len(anns) == 0:
        return
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    return img

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


class MaskCLIPFeaturizer(nn.Module):
    def __init__(self, clip_model_name):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(clip_model_name)
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

@torch.no_grad()
def batch_extract_feature(image_paths: List[str], args):
    OUTPUT_DIR = os.path.join(args.image_root, 'global_clip_features')
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    yolo_iou = 0.9
    yolo_conf = 0.4

    # For part-level CLIP
    transform = T.Compose([
        T.Resize((args.part_resolution, args.part_resolution)),
        T.ToTensor(),
        norm
    ])

    # For object-level CLIP
    raw_transform = T.Compose([
        T.ToTensor(),
        norm
    ])

    dino_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    mobilesamv2, ObjAwareModel, predictor = torch.hub.load("/home/user/LZY/RLA/MobileSAMV2", args.mobilesamv2_encoder_name, source='local')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobilesamv2.to(device=device)
    mobilesamv2.eval()

    samclip = []
    print(f"Computing features for {len(image_paths)} images.")

    clip_model = MaskCLIPFeaturizer(args.clip_model_name).cuda().eval()

    # ======================
    for i in trange(len(image_paths)):
        image_file_path = str(image_paths[i])

        image = cv2.imread(image_file_path)
        # resize to longest edge
        if max(image.shape[:2]) > args.sam_size:
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (int(args.sam_size * image.shape[1] / image.shape[0]), args.sam_size))
            else:
                image = cv2.resize(image, (args.sam_size, int(args.sam_size * image.shape[0] / image.shape[1])))
        image = image[:, :, ::-1]  # BGR to RGB

        raw_input_image = raw_transform(Image.fromarray(image))
        whole_image_feature = clip_model(raw_input_image[None].cuda())[0]
        clip_feat_dim = whole_image_feature.shape[0]

        raw_img_H, raw_img_W = image.shape[:2]

        # part level
        small_W = args.part_feat_res
        small_H = raw_img_H * small_W // raw_img_W

        # obj level
        object_W = args.obj_feat_res
        object_H = raw_img_H * object_W // raw_img_W

        final_W = args.final_feat_res
        final_H = raw_img_H * final_W // raw_img_W

        # ===== Object-aware Model =====
        obj_results = ObjAwareModel(image, device=device, imgsz=args.sam_size, conf=yolo_conf, iou=yolo_iou, verbose=False)
        if not obj_results:
            # Add an all-zero tensor if no object is detected
            samclip.append(torch.zeros((clip_feat_dim, final_H, final_W)))
            continue

        predictor.set_image(image)
        input_boxes1 = obj_results[0].boxes.xyxy
        input_boxes = input_boxes1.cpu().numpy()
        input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda()
        sam_mask = []
        image_embedding = predictor.features
        image_embedding = torch.repeat_interleave(image_embedding, 320, dim=0)
        prompt_embedding = mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding = torch.repeat_interleave(prompt_embedding, 320, dim=0)
        for (boxes,) in batch_iterator(320, input_boxes):
            with torch.no_grad():
                image_embedding = image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embedding = prompt_embedding[0:boxes.shape[0],:,:,:]
                sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,)
                low_res_masks, _ = mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                low_res_masks = predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)
                sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold) * 1.0
                sam_mask.append(sam_mask_pre.squeeze(1))

        sam_mask = torch.cat(sam_mask)
        # Visualize SAM mask
        annotation = sam_mask
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        show_img = annotation[sorted_indices]
        ann_img = show_anns(show_img)
        save_img_path = ('./_mask.png')
        Image.fromarray((ann_img * 255).astype(np.uint8)).save(save_img_path)

        # ===== Object-level CLIP feature =====
        # Interpolate CLIP features to image size
        resized_clip_feat_map_bchw = torch.nn.functional.interpolate(whole_image_feature.unsqueeze(0).float(),
                                                                size=(raw_img_H, raw_img_W),
                                                                mode='bilinear',
                                                                align_corners=False)

        mask_tensor_bchw = sam_mask.unsqueeze(1)

        resized_mask_tensor_bchw = torch.nn.functional.interpolate(mask_tensor_bchw.float(),
                                                                size=(raw_img_H, raw_img_W),
                                                                mode='nearest').bool()

        aggregated_feat_map = torch.zeros((clip_feat_dim, raw_img_H, raw_img_W), dtype=torch.float32, device=device)
        aggregated_feat_cnt_map = torch.zeros((raw_img_H, raw_img_W), dtype=int, device=device)

        for mask_idx in range(resized_mask_tensor_bchw.shape[0]):
            aggregared_clip_feat = resized_clip_feat_map_bchw[0, :, resized_mask_tensor_bchw[mask_idx, 0]]
            aggregared_clip_feat = aggregared_clip_feat.mean(dim=1)

            aggregated_feat_map[:, resized_mask_tensor_bchw[mask_idx, 0]] += aggregared_clip_feat[:, None]
            aggregated_feat_cnt_map[resized_mask_tensor_bchw[mask_idx, 0]] += 1
            
        aggregated_feat_map = aggregated_feat_map / (aggregated_feat_cnt_map[None, :, :] + 1e-6)
        # aggregated_feat_map = F.interpolate(aggregated_feat_map[None], (final_H, final_W), mode='bilinear', align_corners=False)[0]
        samclip.append(aggregated_feat_map.permute(1,2,0).detach().cpu())

    reshaped_tensors = [t.view(-1, 768) for t in samclip]
    stacked_tensor = torch.cat(reshaped_tensors, dim=0)
    U, S, V = torch.pca_lowrank(stacked_tensor, q=32)
    pca_result = torch.mm(stacked_tensor, V[:, :32])

    import pdb
    pdb.set_trace()        

    for i, image_path in tqdm(enumerate(sorted(os.listdir(os.path.join(args.image_root, 'images_8'))))):
        torch.save(pca_clip_features[i], os.path.join(OUTPUT_DIR, image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt')))

    data = {}
    data['proj_v'] = V[:, :32].cpu()
    torch.save(data, os.path.join(args.image_root, 'global_pca_32.pt'))

if __name__ == "__main__":
    parser = ArgumentParser("Compute reference features for feature splatting")
    parser.add_argument("--image_root", "-s", required=True, type=str)
    parser.add_argument("--part_batch_size", type=int, default=32, help="Part-level CLIP inference batch size")
    parser.add_argument("--part_resolution", type=int, default=224, help="Part-level CLIP input image resolution")
    parser.add_argument("--sam_size", type=int, default=1024, help="Longest edge for MobileSAMV2 segmentation")
    parser.add_argument("--obj_feat_res", type=int, default=100, help="Intermediate (for MAP) SAM-enhanced Object-level feature resolution")
    parser.add_argument("--part_feat_res", type=int, default=300, help="Intermediate (for MAP) SAM-enhanced Part-level feature resolution")
    parser.add_argument("--final_feat_res", type=int, default=64, help="Final hierarchical CLIP feature resolution")
    parser.add_argument("--dino_resolution", type=int, default=800, help="Longest edge for DINOv2 feature generation")
    parser.add_argument("--dinov2_model_name", type=str, default='dinov2_vits14')
    parser.add_argument("--mobilesamv2_encoder_name", type=str, default='mobilesamv2_sam_vit_h')
    parser.add_argument("--clip_model_name", type=str, default='ViT-L/14@336px')

    args = parser.parse_args()

    image_paths = [os.path.join(os.path.join(args.image_root, 'images_8'), fn) for fn in os.listdir(os.path.join(args.image_root, 'images_8'))]

    batch_extract_feature(image_paths, args)
    
    