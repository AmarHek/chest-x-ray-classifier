#!/bin/bash
# Activate the virtual environment
source /home/ls6/hekalo/Git/chest-x-ray-classifier/venv/bin/activate

# add project root to PYTHONPATH
PYTHONPATH="${PYTHONPATH}:/home/ls6/hekalo/Git/chest-x-ray-classifier"
export PYTHONPATH

log_output="/home/ls6/hekalo/job_output/baseline-%j.out"
error_output="/home/ls6/hekalo/job_output/baseline-%j.err"

config="/home/ls6/hekalo/Git/chest-x-ray-classifier/configs/baseline.yaml"

# Define model names
backbones_small=(
  "resnet101"
  "densenet121"
  "densenet201"
  "efficientnet_b2"
  "efficientnet_v2_m"
)

backbones_big=(
  "swin_v2_b"
  "swin_v2_t"
  "vit_b_16"
  "vit_b_32"
  "efficientnet_v2_l"
)

# Iterate through model names
for backbone in "${backbones_small[@]}"; do
  # Submit the script as a SLURM job with the current combination of model_name_or_path and dataset_names_or_paths
  sbatch -p ls6 --gres=gpu:rtx2080:1 --wrap="python ../Scripts/train.py $config $backbone" -o $log_output -e $error_output
done

for backbone in "${backbones_big[@]}"; do
  # Submit the script as a SLURM job with the current combination of model_name_or_path and dataset_names_or_paths
  sbatch -p ls6 --gres=gpu:rtx4090ti:1 --wrap="python ../Scripts/train.py $config $backbone" -o $log_output -e $error_output
done

# Deactivate the virtual environment
deactivate
