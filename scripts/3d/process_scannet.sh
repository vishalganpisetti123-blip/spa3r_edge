#!/bin/bash

# Let's start by scaling up to 4 more scenes (you can add all 1200+ later)
SCENES=("scene0001_00" "scene0002_00" "scene0003_00" "scene0004_00")

for SCENE in "${SCENES[@]}"; do
    echo "========================================"
    echo "Processing $SCENE"
    echo "========================================"
    
    mkdir -p dataset/scannet/scans/$SCENE
    
    # 1. Download the .sens file with the invincible loop
    until curl -L -C - --retry 999 --retry-delay 2 --retry-max-time 0 -o dataset/scannet/scans/$SCENE/$SCENE.sens https://kaldir.vc.cit.tum.de/scannet/v1/scans/$SCENE/$SCENE.sens; do 
        echo "Reconnecting..."; sleep 2; 
    done
    
    # Also download the metadata text files required for bounding boxes
    echo "" | python download-scannet.py -o dataset/scannet --id $SCENE --type .txt
    echo "" | python download-scannet.py -o dataset/scannet --id $SCENE --type .aggregation.json
    
    # 2. Extract Color Frames, Poses, and Intrinsics
    python reader.py --filename dataset/scannet/scans/$SCENE/$SCENE.sens --output_path dataset/scannet/scans/$SCENE/ --export_color_images --export_poses --export_intrinsics
    
    echo "$SCENE fully extracted!"
done
