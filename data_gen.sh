#!/bin/bash
#SBATCH --job-name=oversight_gen
#SBATCH --gres=gpu:1
#SBATCH --account=nils
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd /shared/home/bilal.ashfaq/Oversight\ Proj

source /shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh
conda activate dm_proj

mkdir -p logs outputs

echo "===== GPU INFO ====="
nvidia-smi

echo "===== CUDA CHECK ====="
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
python -c "import torch; print('device:', torch.cuda.get_device_name(0))"

echo "===== START GENERATION ====="
python -u dataset_gen_pipeline.py

echo "===== GENERATION FINISHED ====="
