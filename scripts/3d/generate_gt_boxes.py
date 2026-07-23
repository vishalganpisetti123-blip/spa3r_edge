import os
import json
import numpy as np
from plyfile import PlyData

import argparse

BASE_DIR = "dataset/scannet/scans"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None, help="Specific scene to process")
    args = parser.parse_args()

    print("[GT GENERATOR] Starting to compute ground-truth 3D bounding boxes from meshes...")
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for scene in scenes:
        scene_dir = os.path.join(BASE_DIR, scene)
        ply_path = os.path.join(scene_dir, f"{scene}_vh_clean_2.ply")
        segs_path = os.path.join(scene_dir, f"{scene}_vh_clean_2.0.010000.segs.json")
        agg_path = os.path.join(scene_dir, f"{scene}.aggregation.json")
        
        if not (os.path.exists(ply_path) and os.path.exists(segs_path) and os.path.exists(agg_path)):
            print(f"Skipping {scene} (missing mesh or annotation files)")
            continue
            
        print(f"Processing {scene}...")
        
        # 1. Load vertices
        plydata = PlyData.read(ply_path)
        vertices = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        
        # 2. Load segment indices for each vertex
        with open(segs_path, 'r') as f:
            segs_data = json.load(f)
        seg_indices = np.array(segs_data['segIndices']) # map from vertex index to segment ID
        
        # 3. Load aggregation data
        with open(agg_path, 'r') as f:
            agg_data = json.load(f)
            
        gt_boxes = {}
        for group in agg_data.get('segGroups', []):
            obj_id = str(group['objectId'])
            label = group['label']
            segments = set(group['segments'])
            
            # Find all vertices belonging to these segments
            mask = np.isin(seg_indices, list(segments))
            obj_vertices = vertices[mask]
            
            if len(obj_vertices) == 0:
                continue
                
            vmin = obj_vertices.min(axis=0)
            vmax = obj_vertices.max(axis=0)
            
            center = (vmin + vmax) / 2.0
            size = vmax - vmin
            
            gt_boxes[obj_id] = {
                "label": label,
                "center": center.tolist(),
                "size": size.tolist()
            }
            
        out_path = os.path.join(scene_dir, "gt_boxes.json")
        with open(out_path, 'w') as f:
            json.dump(gt_boxes, f, indent=2)
            
        print(f"Saved {len(gt_boxes)} GT boxes for {scene} -> {out_path}")

if __name__ == "__main__":
    main()
