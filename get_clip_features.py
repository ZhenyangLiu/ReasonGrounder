import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from clip_utils.clip_utils import load_clip
from clip_utils import get_features_from_image_and_masks, get_features_from_image_and_bboxes



if __name__ == '__main__':
    
    parser = ArgumentParser(description="Get CLIP features with SAM masks")
    
    parser.add_argument("--image_root", default='./data/360_v2/garden/', type=str)
    parser.add_argument("--downsample", default=1, type=int)

    args = parser.parse_args()

    clip_model = load_clip()
    clip_model.eval()

    OUTPUT_DIR = os.path.join(args.image_root, 'clip_features')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clip_features = []
    pca_clip_features = []
    with torch.no_grad():
        for i, image_path in tqdm(enumerate(sorted(os.listdir(os.path.join(args.image_root, 'images'))))):
            if image_path == '.ipynb_checkpoints':
                print(f"Skipping {image_path}")
                continue
            image = cv2.imread(os.path.join(os.path.join(args.image_root, 'images', image_path)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = torch.load(os.path.join(os.path.join(args.image_root, 'sam_masks'), image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt')))
            features = get_features_from_image_and_bboxes(clip_model, image, masks, background = 0., idx = i).float()
            clip_features.append(features)


        reshaped_tensors = [t.view(-1, 512) for t in clip_features]
        stacked_tensor = torch.cat(reshaped_tensors, dim=0)
        U, S, V = torch.pca_lowrank(stacked_tensor, q=64)
        pca_result = torch.mm(stacked_tensor, V[:, :64])
        split_sizes = [t.shape[0] for t in clip_features]
        pca_result_split = torch.split(pca_result, split_sizes)
        pca_clip_features = [t.view(size, -1) for t, size in zip(pca_result_split, split_sizes)]
            
        clip_index = 0 
        image_files = sorted(os.listdir(os.path.join(args.image_root, 'images')))
        for image_path in tqdm(image_files):
            if image_path == '.ipynb_checkpoints' or not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Skipping {image_path}")
                continue
            if clip_index >= len(pca_clip_features):
                print("Warning: clip_features index exceeds available features.")
                break

            torch.save(
                pca_clip_features[clip_index], 
                os.path.join(OUTPUT_DIR, image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt'))
            )
            clip_index += 1           

        data = {}
        data['proj_v'] = V[:, :64].cpu()
        torch.save(data, os.path.join(args.image_root, 'pca_64.pt'))

    torch.cuda.empty_cache()