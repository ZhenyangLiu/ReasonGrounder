from tqdm import tqdm
import torch, torchvision
from .clip_utils import OpenCLIPNetwork
import numpy as np
import os

default_template = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

@torch.no_grad()



def get_features_from_image_and_masks(clip_model: OpenCLIPNetwork, image: np.array, masks: torch.tensor, background = 1.):
    image_shape = image.shape[:2]
    masks = torch.nn.functional.interpolate(masks.unsqueeze(0).float(), image_shape, mode='bilinear').squeeze(0)
    masks[masks > 0.5] = 1
    masks[masks != 1] = 0
    masks = masks.cpu()
    original_image = torch.from_numpy(image)[None]
    masked_images = masks[:,:,:,None] * original_image + (1 - masks[:,:,:,None]) * 255. * background
    bboxes = torchvision.ops.masks_to_boxes(masks)

    bbox_heights = bboxes[:, 2] - bboxes[:, 0]
    bbox_widths = bboxes[:, 3] - bboxes[:, 1]

    bboxes = bboxes.int().tolist()
    cropped_seg_image_features1x = []


    for seg_idx in range(len(bboxes)):
        with torch.no_grad():
            tmp_image = masked_images[seg_idx][bboxes[seg_idx][1]:bboxes[seg_idx][3], bboxes[seg_idx][0]:bboxes[seg_idx][2], :]
            import matplotlib.pyplot as plt
            plt.imshow(tmp_image.cpu().numpy() / 255.0)
            tmp_image = tmp_image.cuda()
            masked_image_clip_features = clip_model.encode_image(tmp_image[None,...].permute([0,3,1,2]) / 255.0)
            cropped_seg_image_features1x.append(masked_image_clip_features.cpu())
    cropped_seg_image_features1x = torch.cat(cropped_seg_image_features1x, dim=0)

    return cropped_seg_image_features1x

def get_features_from_image_and_bboxes(clip_model: OpenCLIPNetwork, image: np.array, masks: torch.tensor, background=1.0, idx=0, expansion_factor=0.1, save_dir='cropped_images', min_size=1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_shape = image.shape[:2]
    masks = torch.nn.functional.interpolate(masks.unsqueeze(0).float(), image_shape, mode='bilinear').squeeze(0)
    masks = (masks > 0.5).float()
    
    masks = masks.cpu()

    original_image = torch.from_numpy(image)[None]

    masked_images = masks[:, :, :, None] * original_image + (1 - masks[:, :, :, None]) * 255. * background

    bboxes = torchvision.ops.masks_to_boxes(masks)

    bbox_heights = bboxes[:, 2] - bboxes[:, 0]
    bbox_widths = bboxes[:, 3] - bboxes[:, 1]
    
    expand_h = (bbox_heights * expansion_factor).int()
    expand_w = (bbox_widths * expansion_factor).int()

    bboxes[:, 0] = (bboxes[:, 0] - expand_h).clamp(min=0)
    bboxes[:, 1] = (bboxes[:, 1] - expand_w).clamp(min=0)
    bboxes[:, 2] = (bboxes[:, 2] + expand_h).clamp(max=image_shape[0])
    bboxes[:, 3] = (bboxes[:, 3] + expand_w).clamp(max=image_shape[1])

    bboxes = bboxes.int().tolist()
    cropped_seg_image_features1x = []

    for seg_idx in range(len(bboxes)):
        with torch.no_grad():
            y1, x1, y2, x2 = bboxes[seg_idx]

            if y2 - y1 <= 0 or x2 - x1 <= 0:
                tmp_image = original_image[0] * masks[seg_idx][:, :, None] + (1 - masks[seg_idx][:, :, None]) * 255. * background
            else:
                y2 = min(y2, image_shape[0])
                x2 = min(x2, image_shape[1])

                tmp_image = original_image[0][x1:x2, y1:y2, :]


            tmp_image = tmp_image.cuda()
            masked_image_clip_features = clip_model.encode_image(tmp_image[None, ...].permute([0, 3, 1, 2]) / 255.0)

            cropped_seg_image_features1x.append(masked_image_clip_features.cpu())

    cropped_seg_image_features1x = torch.cat(cropped_seg_image_features1x, dim=0)

    return cropped_seg_image_features1x




def get_relevancy_scores(clip_model, images_features, prompt):
    with torch.no_grad():
        clip_model.set_positives([prompt])
        r_scores = []
        relevancy_score = clip_model.get_relevancy(images_features, 0)
        r_score = relevancy_score[:,0]

    return r_score

def get_scale_based_clip(clip_model, images_features, mask_scales, prompt, pca_mat):
    with torch.no_grad():
        text_embeds = clip_model.set_scale_positives([prompt])
        r_scores = []
        features = torch.matmul(images_features, pca_mat.cpu().t())        
        relevancy_score = clip_model.get_scale_relevancy(features, 0)
        r_score = relevancy_score[:,0:1]
        max_index = torch.argmax(r_score)
    return mask_scales[max_index]

def get_relevancy_gaussians(clip_model, images_features, prompt):
    with torch.no_grad():
        
        clip_model.set_positives([prompt])
        r_scores = []
        relevancy_score = clip_model.get_relevancy(images_features, 0)
        r_score = relevancy_score[:,0]

    return r_score


# def get_scores_with_template(clip_model, images_features, prompt, template = default_template):
#     with torch.no_grad():
#         clip_model.set_positives([t.format(prompt) for t in template])
#     r_scores = []
#     for i,f in enumerate(images_features):
#         with torch.no_grad():
#             # N_image_features x N_pos x 2
#             relevancy_scores = clip_model.get_relevancy_with_template(f)
#             # N_image_features x N_pos
#             r_score = relevancy_scores[...,0]
#         r_scores.append(r_score)
#     return r_scores

def get_scores_with_template(clip_model, images_features, prompt, template = default_template):
    with torch.no_grad():
        clip_model.set_positives([t.format(prompt) for t in template])
        r_scores = []

        for i,f in enumerate(images_features):

            # N_image_features x N_pos x 2
            relevancy_scores = clip_model.get_relevancy_with_template(f)
            # N_image_features x N_pos
            r_score = relevancy_scores[...,0]
            r_scores.append(r_score)

    return r_score

def get_scores_with_template(clip_model, images_features, prompt, template = default_template):
    with torch.no_grad():
        # clip_model.set_positives([t.format(prompt) for t in template])
        from time import time
        start_time = time()
        clip_model.set_positive_with_template(prompt, template)
        # print('set_positive_with_template', time() - start_time)
        # N_image_features x N_pos x 2
        start_time = time()
        relevancy_scores = clip_model.get_relevancy_with_template(images_features)
        # print('get_relevancy_with_template', time() - start_time)
        # N_image_features x N_pos
        r_score = relevancy_scores[...,0]

    return r_score

def get_segmentation(clip_model, images_features, images_masks, prompts = []):
    with torch.no_grad():
        clip_model.set_positives(prompts)
    images_scores = []
    for i,f in tqdm(enumerate(images_features)):
        with torch.no_grad():
            # f = torch.nn.functional.normalize(f, dim = -1)
            # k, n_p
            segmentation_score = clip_model.get_segmentation(f)
            # n_p, h, w
            image_score = torch.einsum('kp,khw->kphw', segmentation_score, images_masks[i]).sum(dim = 0) / (images_masks[i].sum(dim = 0, keepdim = True)+1e-9)
        images_scores.append(image_score)
    return images_scores


import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)

def get_3d_mask(args, pipeline, scene_gaussians, cameras, image_names, images_scores, save_path = None, filtered_views = None):
    tmp_mask = torch.zeros_like(scene_gaussians.get_mask).float().detach().clone()
    tmp_mask.requires_grad = True

    for it, view in tqdm(enumerate(cameras)):
        if filtered_views is not None and it not in filtered_views:
            continue
        image_idx = None
        try:
            image_idx = image_names.index(view.image_name+'.jpg')
        except:
            continue

        background = torch.zeros(tmp_mask.shape[0], 3, device = 'cuda')
        rendered_mask_pkg = gaussian_renderer.render_mask(view, scene_gaussians, pipeline.extract(args), background, precomputed_mask=tmp_mask)

        gt_score = images_scores[image_idx]

        tmp_target_mask = torch.nn.functional.interpolate(gt_score.unsqueeze(0).unsqueeze(0).float(), size=rendered_mask_pkg['mask'].shape[-2:] , mode='bilinear').squeeze(0)

        loss = -(tmp_target_mask * rendered_mask_pkg['mask']).sum()

        loss.backward()
        grad_score = tmp_mask.grad.clone()

        tmp_mask.grad.detach_()
        tmp_mask.grad.zero_()
        with torch.no_grad():
            tmp_mask = tmp_mask - grad_score

        tmp_mask.requires_grad = True

    with torch.no_grad():
        tmp_mask[tmp_mask <= 0] = 0
        tmp_mask[tmp_mask != 0] = 1
    if save_path is not None:
        torch.save(tmp_mask.bool(), save_path)
    # torch.save(tmp_mask.bool(), './segmentation_res/final_mask.pt')
    # final_mask = tmp_mask
    return tmp_mask


def load_multi_lvl_features_and_masks(image_path = './data/3dovs/bed/images/', feature_path = './data/3dovs/bed/language_features/'):



    images_features = [[],[],[],[]]
    images_masks = [[],[],[],[]]

    for image_name in sorted(os.listdir(image_path)):
        name = image_name.split('.')[0]
        feature_name = name + '_f.npy'
        mask_name = name + '_s.npy'

        tmp_f = np.load('./data/3dovs/bed/language_features/'+feature_name)
        tmp_s = np.load('./data/3dovs/bed/language_features/'+mask_name)

        all_features = 0
        for lvl, mask in enumerate(tmp_s):
            idxes = list(np.unique(mask))
            try:
                idxes.remove(-1)
            except:
                pass
            num_masks_in_lvl = len(idxes)
            images_features[lvl].append(tmp_f[all_features:all_features+num_masks_in_lvl])
            all_features += num_masks_in_lvl

            this_lvl_masks = []
            for idx in sorted(idxes):
                idx = int(idx)
                this_lvl_masks.append((mask == idx).astype(np.float32))
            images_masks[lvl].append(this_lvl_masks)


def get_multi_lvl_scores(clip_model, images_features, images_masks, prompt):
    with torch.no_grad():
        clip_model.set_positives([prompt])
    images_scores = [[],[],[],[]]
    for lvl, fs_in_lvl in tqdm(enumerate(images_features)):
        if lvl == 0:
            continue
        for i,f in enumerate(fs_in_lvl):

            with torch.no_grad():
                relevancy_score = clip_model.get_relevancy(torch.from_numpy(f).cuda(), 0)
                r_score = relevancy_score[:,0]

                stacked_image_mask = torch.from_numpy(np.stack(images_masks[lvl][i])).cuda()
                image_score = (r_score[:,None, None] * stacked_image_mask).sum(dim = 0)
                
            images_scores[lvl].append(image_score)
    final_images_scores = []
    for i in range(len(images_scores[1])):
        final_images_scores.append(torch.stack([images_scores[1][i], images_scores[2][i], images_scores[3][i]], dim = 0))
    return final_images_scores