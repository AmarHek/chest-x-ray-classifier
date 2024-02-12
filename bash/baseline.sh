#!/bin/bash
# Activate the virtual environment
source /home/ls6/hekalo/Git/chest-x-ray-classifier/venv/bin/activate

# add project root to PYTHONPATH
PYTHONPATH="${PYTHONPATH}:/home/ls6/hekalo/Git/chest-x-ray-classifier"
export PYTHONPATH

log_output="/home/ls6/hekalo/job_output/baseline/baseline-%j.out"
error_output="/home/ls6/hekalo/job_output/baseline/baseline-%j.err"

config="/home/ls6/hekalo/Git/chest-x-ray-classifier/configs/trainconfig_baseline.yaml"

# Define model names
backbones_bigger=(
  "efficientnet_v2_m"
  "vit_l_16"
  "densenet169"
  "densenet201"
)

for backbone in "${backbones_bigger[@]}"; do
  # Submit the script as a SLURM job with the current combination of model_name_or_path and dataset_names_or_paths
  sbatch -p ls6prio --gres=gpu:rtx3090:1 --wrap="python src/Scripts/train.py $config --backbone=$backbone" -o $log_output -e $error_output
done

# Deactivate the virtual environment
deactivate
