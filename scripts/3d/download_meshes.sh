#!/bin/bash

SCENES=("scene0000_00" "scene0001_00" "scene0002_00" "scene0003_00" "scene0004_00")

for SCENE in "${SCENES[@]}"; do
    echo "========================================"
    echo "Downloading meshes for $SCENE"
    echo "========================================"
    
    mkdir -p dataset/scannet/scans/$SCENE
    
    echo "" | python download-scannet.py -o dataset/scannet --id $SCENE --type _vh_clean_2.ply
    echo "" | python download-scannet.py -o dataset/scannet --id $SCENE --type _vh_clean_2.0.010000.segs.json
    
    echo "$SCENE meshes downloaded!"
done
