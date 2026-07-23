import os
import json
import numpy as np
import cv2
import random
import onnxruntime as ort
import torch
from torchvision import transforms
import argparse

OUTPUT_DIR = "cached_3d_features"
BASE_DIR = "dataset/scannet/scans/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The core classes we actually care about evaluating
ALLOWED_CLASSES = {
    "table", "chair", "sofa", "couch", "bed", "window", 
    "door", "doorframe", "desk", "cabinet", "monitor", 
    "bookshelf", "kitchen counter", "tv", "refrigerator",
    "coffee table"
}

onnx_session = ort.InferenceSession("models/spa3r_encoder_psfm_fp32.onnx")
input_name = onnx_session.get_inputs()[0].name

color_aug = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
])

def get_latents(img_tensor):
    raw_tokens = onnx_session.run(None, {input_name: img_tensor})[0]
    return raw_tokens[0].tolist()

def load_real_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((1, 1, 3, 224, 224), dtype=np.float32)
    img_resized = cv2.resize(img, (224, 224))
    
    img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    img_t = color_aug(img_t)
    
    img_tensor = np.expand_dims(np.expand_dims(img_t.numpy(), axis=0), axis=0)
    return img_tensor

def load_axis_alignment(meta_file_path):
    if not os.path.exists(meta_file_path):
        return np.eye(4)
    with open(meta_file_path, 'r') as f:
        for line in f:
            if line.startswith('axisAlignment'):
                vals = [float(x) for x in line.strip().split('=')[1].split()]
                return np.array(vals).reshape(4, 4)
    return np.eye(4)

def world_to_pixel(x, y, z, pose_matrix, inv_align_matrix, intrinsic):
    p_world_aligned = np.array([x, y, z, 1.0])
    p_world_unaligned = inv_align_matrix @ p_world_aligned
    p_cam = np.linalg.inv(pose_matrix) @ p_world_unaligned
    x_c, y_c, z_c = p_cam[:3]
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = (x_c * fx / z_c) + cx if z_c != 0 else 0
    v = (y_c * fy / z_c) + cy if z_c != 0 else 0
    return u, v, z_c

def generate_spatial_prompts(gt_boxes):
    # gt_boxes: dict mapping obj_id (str) to dict with "label", "center", "size"
    label_to_ids = {}
    for obj_id, obj_data in gt_boxes.items():
        label = obj_data["label"]
        if label not in label_to_ids:
            label_to_ids[label] = []
        label_to_ids[label].append(obj_id)
        
    final_prompts = {}
    for label, ids in label_to_ids.items():
        if len(ids) == 1:
            final_prompts[ids[0]] = f"{label} in room"
        else:
            # Sort instances of the same label by their X-coordinate (left to right)
            ids_sorted_by_x = sorted(ids, key=lambda i: gt_boxes[i]["center"][0])
            for index, obj_id in enumerate(ids_sorted_by_x):
                if index == 0:
                    modifier = "on the left"
                elif index == len(ids_sorted_by_x) - 1:
                    modifier = "on the right"
                else:
                    modifier = "in the middle"
                final_prompts[obj_id] = f"{label} {modifier} in room"
    return final_prompts

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default=None, help="Specific scene to process")
args = parser.parse_args()

if args.scene:
    all_scenes = [args.scene]
else:
    all_scenes = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

output_file = os.path.join(OUTPUT_DIR, "scannet_3d_cached_FULL.json")
all_cached_data = []
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        all_cached_data = json.load(f)

print(f"[3D CACHE] Caching multi-view spatial latents using generated GT boxes...")

for scene_id in all_scenes:
    print(f"Caching features for {scene_id}...")
    scene_dir = os.path.join(BASE_DIR, scene_id)
    color_dir = os.path.join(scene_dir, "color")
    pose_dir = os.path.join(scene_dir, "pose")
    intrinsic_path = os.path.join(scene_dir, "intrinsic/intrinsic_color.txt")
    meta_file = os.path.join(scene_dir, f"{scene_id}.txt")
    gt_boxes_file = os.path.join(scene_dir, "gt_boxes.json")
    
    if not (os.path.exists(color_dir) and os.path.exists(pose_dir) and os.path.exists(intrinsic_path)):
        print(f"  -> Skipping {scene_id}: missing extracted frames/poses")
        continue
        
    if not os.path.exists(gt_boxes_file):
        print(f"  -> Skipping {scene_id}: missing gt_boxes.json (run generate_gt_boxes.py first)")
        continue
        
    all_frames = [f for f in os.listdir(color_dir) if f.endswith('.jpg')]
    if not all_frames:
        continue
        
    intrinsic = np.loadtxt(intrinsic_path)
    align_matrix = load_axis_alignment(meta_file)
    inv_align_matrix = np.linalg.inv(align_matrix)
    
    with open(gt_boxes_file, 'r') as f:
        gt_boxes = json.load(f)
        
    # Filter out rare classes
    gt_boxes = {k: v for k, v in gt_boxes.items() if v["label"].lower() in ALLOWED_CLASSES}
        
    # Apply spatial modifiers to the labels
    final_prompts = generate_spatial_prompts(gt_boxes)
    
    cached_for_scene = 0
    for obj_id, obj_data in gt_boxes.items():
        prompt = final_prompts.get(obj_id, f"object in room")
        center = obj_data["center"]
        size = obj_data["size"]
        gt_box_3d = center + size # [x, y, z, dx, dy, dz]
    
        target_frame = None
        u, v, d = 0, 0, 0
        valid = False
        
        # Try to find a frame where this object is visible
        for _ in range(100):
            target_frame = random.choice(all_frames)
            frame_id = target_frame.split('.')[0]
            pose_path = os.path.join(pose_dir, f"{frame_id}.txt")
            if not os.path.exists(pose_path):
                continue
            pose = np.loadtxt(pose_path)
            u, v, d = world_to_pixel(center[0], center[1], center[2], pose, inv_align_matrix, intrinsic)
            # Object must be in front of the camera (d > 0.1) and within the 1296x968 image plane
            if d > 0.1 and 0 <= u <= 1296 and 0 <= v <= 968:
                valid = True
                break
                
        if not valid:
            continue
            
        gt_cam_box = [float(u)/1296.0, float(v)/968.0, float(d)/10.0, gt_box_3d[3], gt_box_3d[4], gt_box_3d[5]]
        
        # Pick 2 random context frames
        context_frames = random.sample(all_frames, 2)
        
        img1 = load_real_image(os.path.join(color_dir, context_frames[0]))
        img2 = load_real_image(os.path.join(color_dir, context_frames[1]))
        img3 = load_real_image(os.path.join(color_dir, target_frame))
    
        all_cached_data.append({
            "scene_id": scene_id,
            "object_id": obj_id,
            "prompt": prompt,
            "target_frame_id": target_frame.split('.')[0],
            "spatial_latents_c1": get_latents(img1),
            "spatial_latents_c2": get_latents(img2),
            "spatial_latents_t": get_latents(img3),
            "gt_box_cam": gt_cam_box,
            "gt_box_3d": gt_box_3d
        })
        cached_for_scene += 1
            
    print(f"  -> Cached {cached_for_scene} features for {scene_id}")

with open(output_file, "w") as f:
    json.dump(all_cached_data, f)

print(f"[SUCCESS] Saved {len(all_cached_data)} cached 3D features!")
