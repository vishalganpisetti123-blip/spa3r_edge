#!/bin/bash

# Ensure we have the scannetv2_train.txt file
if [ ! -f "scannetv2_train.txt" ]; then
    echo "Error: scannetv2_train.txt not found!"
    exit 1
fi

while read SCENE; do
    # Skip empty lines
    if [ -z "$SCENE" ]; then continue; fi

    echo "========================================"
    echo "Processing $SCENE"
    echo "========================================"
    
    mkdir -p dataset/scannet/scans/$SCENE
    
    # 1. Download .sens, .ply, .segs.json
    echo "Downloading $SCENE.sens..."
    until curl -sS -L -C - --retry 999 -o dataset/scannet/scans/$SCENE/$SCENE.sens https://kaldir.vc.cit.tum.de/scannet/v1/scans/$SCENE/$SCENE.sens; do echo "Reconnecting..."; sleep 2; done
    
    echo "Downloading meshes and segments..."
    echo "" | python download-scannet.py -o dataset/scannet --id $SCENE --type _vh_clean_2.ply
    echo "" | python download-scannet.py -o dataset/scannet --id $SCENE --type _vh_clean_2.0.010000.segs.json
    
    # 2. Extract Color Frames, Poses, and Intrinsics
    echo "Extracting frames and poses..."
    python reader.py --filename dataset/scannet/scans/$SCENE/$SCENE.sens --output_path dataset/scannet/scans/$SCENE/ --export_color_images --export_poses --export_intrinsics
    
    # 3. Generate GT Boxes
    echo "Generating GT boxes..."
    python scripts/3d/generate_gt_boxes.py --scene $SCENE
    
    # 4. CACHE the spatial features IMMEDIATELY
    echo "Caching 3D features..."
    python scripts/3d/cache_3d_features.py --scene $SCENE
    
    # 5. DESTROY the heavy raw files
    echo "Cleaning up heavy files for $SCENE..."
    rm -f dataset/scannet/scans/$SCENE/$SCENE.sens
    rm -f dataset/scannet/scans/$SCENE/${SCENE}_vh_clean_2.ply
    rm -rf dataset/scannet/scans/$SCENE/color/
    
    echo "$SCENE processing complete. Storage reclaimed!"
done < scannetv2_train.txt
