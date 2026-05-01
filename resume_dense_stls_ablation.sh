#!/usr/bin/env bash
set -euo pipefail

export SFM_UID=$(id -u)
export SFM_GID=$(id -g)

for obj in estatua leon; do
    echo "============================================="
    echo "Processing ${obj} from scratch (Dense Reconstruction n160_r1)..."
    echo "============================================="
    
    sparse_bin_dir="outputs/${obj}/ablation/n160_r1/sparse/0"
    dense_dir="outputs/${obj}/ablation_dense"
    
    if [ ! -d "${sparse_bin_dir}" ]; then
        echo "Missing sparse model for ${obj} at ${sparse_bin_dir}, skipping..."
        continue
    fi

    mkdir -p "${dense_dir}"
    
    echo "0. Fixing paths in sparse model database for ${obj}..."
    mkdir -p "${dense_dir}/fixed_txt" "${dense_dir}/fixed_bin"
    
    docker compose run -T --user root --rm colmap model_converter \
        --input_path "/workspace/${sparse_bin_dir}" \
        --output_path "/workspace/${dense_dir}/fixed_txt" \
        --output_type TXT
        
    docker compose run -T --user root --rm --entrypoint sed sfm \
        -i "s|../../../../../${obj}/images/||g" "/workspace/${dense_dir}/fixed_txt/images.txt"
        
    docker compose run -T --user root --rm colmap model_converter \
        --input_path "/workspace/${dense_dir}/fixed_txt" \
        --output_path "/workspace/${dense_dir}/fixed_bin" \
        --output_type BIN
        
    echo "1. Undistorting images for ${obj}..."
    docker compose run -T --user root --rm colmap image_undistorter \
        --image_path "/workspace/data/${obj}/images" \
        --input_path "/workspace/${dense_dir}/fixed_bin" \
        --output_path "/workspace/${dense_dir}" \
        --output_type COLMAP \
        --max_image_size 1200

    echo "2. Patch Match Stereo (Resuming Dense Reconstruction)..."
    docker compose run -T --user root --rm colmap-gpu patch_match_stereo \
        --workspace_path "/workspace/${dense_dir}" \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.max_image_size 1200
        
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
mesh.export('/workspace/outputs/${obj}/reconstruction/model_dense_n160.stl')
        \"
    "
    
    # 6. Fix ownership
    docker compose run -T --rm --user root --entrypoint chown sfm \
        -R ${SFM_UID}:${SFM_GID} /workspace/${dense_dir} /workspace/outputs/${obj}/reconstruction/model_dense_n160.stl || true
        
    echo "Finished ${obj}!"
done
