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
    
    image_data = BenchmarkDataset(args.data_dir,'./kinetics_dino')
    image_data.switch_to(args.idx)
    sample = image_data[0]
    H,W = sample.shape[-3:-1]
    
    
    dino_out_dir = os.path.join('./kinetics_dino',image_data.curname(),'dino_embeddings')
    mask_dir = os.path.join(dino_out_dir,'sam2_mask')
    os.makedirs(mask_dir,exist_ok=True)
    
    video_dir = f'./kinetics_dino/{image_data.curname()}/video'
    resize_dir = f'./kinetics_dino/{image_data.curname()}/video_resize'
    os.makedirs(resize_dir,exist_ok=True)
    jpg_files = glob.glob(os.path.join(video_dir, '*.jpg'))
    H,W = image_data.H,image_data.W
    # import pdb;pdb.set_trace()
    for jpg_file in tqdm(jpg_files):
        # 读取图像
        img = cv2.imread(jpg_file)
        # H, W = img.shape[:2]
        
        # resize图像
        img_resized = cv2.resize(img, (480*W//H, 480), interpolation = cv2.INTER_LINEAR)
        
        # 获取图像的文件名
        base_name = os.path.basename(jpg_file)
        
        # 创建保存resize图像的路径
        save_path = os.path.join(resize_dir, base_name)
        
        # 保存resize后的图像，设置JPEG质量为100，几乎无损
        cv2.imwrite(save_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    # os.makedirs(video_dir,exist_ok=True)
    # for i in range(len(image_data)):
    #     frame = image_data[i][:,:,[2,1,0]]
    #     cv2.imwrite(os.path.join(video_dir,f'{i:05d}.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY, 100])
    device = torch.device("cuda")
    
    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    inference_state = predictor.init_state(resize_dir)
    
    
    # import pdb;pdb.set_trace()
    for frame in range(0,len(image_data),5):
        image_data.start_frame = frame
        image_data.switch_to(args.idx)
        points, occluded = image_data.get_gt()
        valid_points = (torch.from_numpy(points[:,0]))*torch.tensor([W,H]).float()
        occ_mask = torch.from_numpy(occluded[:,0,])
        valid_points = valid_points[occ_mask == False,:].cuda()
        
        ann_frame_idx = frame  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # import pdb;pdb.set_trace()
        all_masks = []
        predictor.reset_state(inference_state)
        # import pdb;pdb.set_trace()
        point_map = torch.ones((valid_points.shape[0]),dtype=torch.int32).cuda()*-1
        mask_count = 0
        for i in range(valid_points.shape[0]):
            if point_map[i] != -1:
                continue
            points = valid_points[i:i+1].cpu().numpy()
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1], np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=i,
                points=points,
                labels=labels,
            )
            # sampled_logits = F.grid_sample(out_mask_logits[-1:].cuda(),(valid_points/torch.tensor([W,H]).cuda().float()*2-1)[None,None,...])[0,0,0]
            # sampled_logits = sampled_logits > 0.0
            # chosen_points = torch.logical_and(sampled_logits, point_map == -1)
            # point_map[chosen_points] = mask_count
            # mask_count += 1
            point_map[i] = i
            
        # import pdb;pdb.set_trace()
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
                
        predictor.reset_state(inference_state)
            
        mask_count = 0
        for i in range(valid_points.shape[0]):
            if point_map[i] < mask_count:
                continue
            labels = np.array([1], np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=i,
                points=points,
                labels=labels,
            )
            mask_count += 1
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
        
        images = []
        for out_frame_idx in range(0, len(image_data)):
            masks = []
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                masks.append(out_mask)
                # import pdb;pdb.set_trace()
                # images.append((image_data[out_frame_idx]/4.+out_mask[0][...,None]*100).astype(np.uint8))
            masks = np.stack(masks)
            all_masks.append(masks)
        # save_images_as_video(images,os.path.join(mask_dir,f'{i}.mp4'))
        

        try:
            # import pdb;pdb.set_trace()
            all_masks = np.stack(all_masks,axis=1)
            torch.save(torch.from_numpy(all_masks),os.path.join(mask_dir,f'all_mask_{frame}.pt'))
            torch.save(point_map.cpu(),os.path.join(mask_dir,f'point_map_{frame}.pt'))
            print('saved to ',os.path.join(mask_dir,f'all_mask_{frame}.pt'),' with shape ',all_masks.shape)
        except:
            pass
    
    
