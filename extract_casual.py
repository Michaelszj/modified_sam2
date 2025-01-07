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
from PIL import Image
import ffmpeg

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
    
    
def load_video(path):
    # 获取目录中所有的 .jpg 文件并排序
    files = sorted([f for f in os.listdir(path) if f.endswith('.jpg')])
    frames = []
    
    # 依次读取每个 .jpg 文件并转换为 tensor
    for file in files:
        img_path = os.path.join(path, file)
        img = Image.open(img_path)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # 将通道维度放到前面，符合 Pytorch 的格式
        frames.append(img_tensor)
    
    # 将所有帧合并为一个 4D tensor
    video_tensor = torch.stack(frames)  # (T, C, H, W)
    return video_tensor

def save_video(video_tensor, output_path, fps=10):
    # 检查 tensor 格式是否正确
    if video_tensor.dim() != 4 or video_tensor.size(1) != 3:
        raise ValueError("Expected video_tensor to have shape (帧数, 通道数, 高度, 宽度) with 3 channels.")

    # 提取视频的帧数、高度和宽度
    num_frames, channels, height, width = video_tensor.shape
    
    # 将通道维度从 (帧数, 通道数, 高度, 宽度) 转换为 (帧数, 高度, 宽度, 通道数)
    video_tensor = video_tensor.permute(0, 2, 3, 1)

    # 转换为 uint8 格式，以符合视频保存的要求
    video_tensor = video_tensor.clamp(0, 255).byte().numpy()

    # 初始化视频写入器
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
        .output(output_path, pix_fmt='yuv420p', vcodec='libx264')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )



    # 将每一帧写入 ffmpeg
    for frame in video_tensor:
        process.stdin.write(frame.tobytes())
    
    # 关闭管道
    process.stdin.close()
    process.wait()
    print(f"视频已保存至 {output_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    args = parser.parse_args()
    
    
    mask_dir = os.path.join(args.data_dir,'sam2_mask')
    jpg_dir = os.path.join(args.data_dir,'video')
    os.makedirs(mask_dir,exist_ok=True)
    device = torch.device("cuda")
    
    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    video = load_video(jpg_dir)
    inference_state = predictor.init_state(jpg_dir)
    H,W = video.shape[-2:]
    
    
    # ***** Generate the mask for the object based on a target point *****
    # ***** Feel free to modify this *****
    target = torch.tensor([500,650]).float().cuda()
    
    
    
    
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # import pdb;pdb.set_trace()
    all_masks = []
    predictor.reset_state(inference_state)
    # import pdb;pdb.set_trace()
    mask_count = 0
    
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=0,
        points=target[None,:],
        labels=labels,
    )
    # sampled_logits = F.grid_sample(out_mask_logits[-1:].cuda(),(valid_points/torch.tensor([W,H]).cuda().float()*2-1)[None,None,...])[0,0,0]
    # sampled_logits = sampled_logits > 0.0
    # chosen_points = torch.logical_and(sampled_logits, point_map == -1)
    # point_map[chosen_points] = mask_count
    # mask_count += 1
        
    # import pdb;pdb.set_trace()
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
            
    images = []
    for out_frame_idx in range(0, len(video)):
        masks = []
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            masks.append(out_mask)
            # import pdb;pdb.set_trace()
            # images.append((image_data[out_frame_idx]/4.+out_mask[0][...,None]*100).astype(np.uint8))
        masks = np.stack(masks)
        all_masks.append(masks)
    # save_images_as_video(images,os.path.join(mask_dir,f'{i}.mp4'))
    all_masks = np.stack(all_masks,axis=1) # (1,T,1,H,W)
    # import pdb;pdb.set_trace()
    mesh = torch.stack(torch.meshgrid([torch.arange(0,H),torch.arange(0,W)]),dim=-1)
    density = 12
    mod_mask_sparse = torch.logical_and((mesh[:,:,0] % density)==0,(mesh[:,:,1] % density)==0)
    obj_mask_sparse = torch.logical_and(torch.from_numpy(all_masks[0,0,0]),mod_mask_sparse)
    valid_points_sparse = torch.nonzero(obj_mask_sparse)[...,[1,0]]*torch.tensor([1/W,1/H]).float()
    # import pdb;pdb.set_trace()
    density = 4
    mod_mask_dense = torch.logical_and((mesh[:,:,0] % density)==0,(mesh[:,:,1] % density)==0)
    obj_mask_dense = torch.logical_and(torch.from_numpy(all_masks[0,0,0]),mod_mask_dense)
    valid_points_dense = torch.nonzero(obj_mask_dense)[...,[1,0]]*torch.tensor([1/W,1/H]).float()
    
    frame = 0
    torch.save(torch.from_numpy(all_masks),os.path.join(mask_dir,f'all_mask.pt'))
    torch.save(valid_points_sparse,os.path.join(mask_dir,f'query_sparse.pt'))
    torch.save(valid_points_dense,os.path.join(mask_dir,f'query_dense.pt'))
    np.save(os.path.join(mask_dir,f'query_sparse.npy'),valid_points_sparse.numpy())
    np.save(os.path.join(mask_dir,f'query_dense.npy'),valid_points_dense.numpy())
    vis = video//2+all_masks[0].astype(np.uint8)*100
    save_video(vis,os.path.join(mask_dir,'vis.mp4'))
    print('saved to ',os.path.join(mask_dir,f'all_mask.pt'),' with shape ',all_masks.shape)
    
    
    
