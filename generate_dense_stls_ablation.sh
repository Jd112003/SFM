#!/usr/bin/env bash
set -euo pipefail

export SFM_UID=$(id -u)
export SFM_GID=$(id -g)

for obj in buda estatua leon; do
    echo "============================================="
    echo "Processing ${obj} (Dense Reconstruction using Ablation n80_r1)..."
    echo "============================================="
    
    sparse_bin_dir="outputs/${obj}/ablation/n80_r1/sparse/0"
    dense_dir="outputs/${obj}/ablation_dense"
    
    if [ ! -d "${sparse_bin_dir}" ]; then
        echo "Missing sparse model for ${obj} at ${sparse_bin_dir}, skipping..."
        continue
    fi

    mkdir -p "${dense_dir}"
    
    # 0. Fix relative paths in sparse model
    echo "0. Fixing paths in sparse model database..."
    mkdir -p "outputs/${obj}/ablation_dense/fixed_txt"
    mkdir -p "outputs/${obj}/ablation_dense/fixed_bin"
    
    docker compose run -T --user root --rm colmap model_converter \
        --input_path "/workspace/${sparse_bin_dir}" \
        --output_path "/workspace/outputs/${obj}/ablation_dense/fixed_txt" \
        --output_type TXT
        
    # Replace relative path garbage
    docker compose run -T --user root --rm --entrypoint sed sfm \
        -i "s|../../../../../${obj}/images/||g" \
        "/workspace/outputs/${obj}/ablation_dense/fixed_txt/images.txt"
        
    docker compose run -T --user root --rm colmap model_converter \
        --input_path "/workspace/outputs/${obj}/ablation_dense/fixed_txt" \
        --output_path "/workspace/outputs/${obj}/ablation_dense/fixed_bin" \
        --output_type BIN
        
    echo "1. Undistorting images from fixed n80 sparse model..."
    docker compose run -T --user root --rm colmap image_undistorter \
        --image_path "/workspace/data/${obj}/images" \
        --input_path "/workspace/outputs/${obj}/ablation_dense/fixed_bin" \
        --output_path "/workspace/${dense_dir}" \
        --output_type COLMAP \
        --max_image_size 800
        
    echo "2. Patch Match Stereo (Dense Reconstruction)..."
    docker compose run -T --user root --rm colmap-gpu patch_match_stereo \
        --workspace_path "/workspace/${dense_dir}" \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.max_image_size 800
        
    echo "3. Stereo Fusion..."
    docker compose run -T --user root --rm colmap-gpu stereo_fusion \
        --workspace_path "/workspace/${dense_dir}" \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path "/workspace/${dense_dir}/fused.ply"
        
    echo "4. Poisson Meshing..."
    docker compose run -T --user root --rm colmap poisson_mesher \
        --input_path "/workspace/${dense_dir}/fused.ply" \
        --output_path "/workspace/${dense_dir}/meshed-poisson.ply"
        
    echo "5. Converting PLY to STL..."
    docker compose run -T --rm --user root --entrypoint bash sfm -c "\
        pip install trimesh scipy networkx rtree && \
        python -c \"
import trimesh
mesh = trimesh.load('/workspace/${dense_dir}/meshed-poisson.ply')
mesh.export('/workspace/outputs/${obj}/reconstruction/model_dense_n80.stl')
        \"
    "
    
    # Fix ownership of everything created in this folder
    docker compose run -T --rm --user root --entrypoint chown sfm \
        -R ${SFM_UID}:${SFM_GID} /workspace/outputs/${obj}/ablation_dense /workspace/outputs/${obj}/reconstruction/model_dense_n80.stl || true
        
    echo "Finished ${obj}!"
done
