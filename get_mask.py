import argparse
import sys
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
from dataset import pklDataset, FeatureDataset, BenchmarkDataset
from sam2.sam2_video_predictor import SAM2VideoPredictor
import torch.nn.functional as F

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
    
def save_images_as_video(images, output_file):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 10.0, (width, height))

    for image in images:
        video_writer.write(image)

    video_writer.release()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    parser.add_argument('--idx', type=int, default=0, help='dataset idx')
    args = parser.parse_args()
    
    
    # os.makedirs(video_dir,exist_ok=True)
    # for i in range(len(image_data)):
    #     frame = image_data[i][:,:,[2,1,0]]
    #     cv2.imwrite(os.path.join(video_dir,f'{i:05d}.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY, 100])
    device = torch.device("cuda")
    
    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    inference_state = predictor.init_state('raw_video/bmx_trees')
    
    point = np.array([0.5,0.5])
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=17,
                obj_id=0,
                points=point,
                labels=labels,
            )
    
    mask = out_mask_logits[0] > 0.
    import pdb;pdb.set_trace()