#!/bin/bash
# Activate the virtual environment
source /home/ls6/hekalo/Git/chest-x-ray-classifier/venv/bin/activate

# add project root to PYTHONPATH
PYTHONPATH="${PYTHONPATH}:/home/ls6/hekalo/Git/chest-x-ray-classifier"
export PYTHONPATH

log_output="/home/ls6/hekalo/job_output/baseline/baseline-%j.out"
error_output="/home/ls6/hekalo/job_output/baseline/baseline-%j.err"

config="/home/ls6/hekalo/Git/chest-x-ray-classifier/configs/trainconfig_baseline_huggingface.yaml"

# Define model names
backbones=(
"facebook/dinov2-base"
"facebook/convnext-base-224"
"facebook/convnextv2-base-1k-224"
"snap-research/efficientformer-l1-300"
"Xrenya/pvt-medium-224"
)

for backbone in "${backbones_bigger[@]}"; do
  # Submit the script as a SLURM job with the current combination of model_name_or_path and dataset_names_or_paths
  sbatch -p ls6prio --gres=gpu:rtx4090:1 --wrap="python src/Scripts/train.py $config --backbone=$backbone" -o $log_output -e $error_output
done

# Deactivate the virtual environment
deactivate
