#!/bin/bash
# Activate the virtual environment
source /home/ls6/hekalo/Git/chest-x-ray-classifier/venv/bin/activate

# add project root to PYTHONPATH
PYTHONPATH="${PYTHONPATH}:/home/ls6/hekalo/Git/chest-x-ray-classifier"
export PYTHONPATH

log_output="/home/ls6/hekalo/job_output/slurm-%j.out"

# Define model names
architectures=(
  "densenet121"
  "efficientnet_v2_s"
  "efficientnet_b1"
  "resnet50"
)

classes_sets=(
  "pneumonia"
  "chexternal"
  "chexternal_pneumo"
  "chexpert"
)

model_path="/scratch/hekalo/Experiments/labels_chexpert/bce/"

csv_path="/scratch/hekalo/Datasets/CheXpert-v1.0-small/"
img_path="/scratch/hekalo/Datasets/"
image_size=320
loss="bce"
optimizer="adam"
learning_rate=0.01
batch_size=32
epochs=100

# Iterate through model names
for architecture in "${architectures[@]}"; do
  # Iterate through classes
  for classes in "${classes_sets[@]}"; do
    # Submit the script as a SLURM job with the current combination of model_name_or_path and dataset_names_or_paths
    sbatch -p ls6 --gres=gpu:rtx3090:1 --wrap="python ../Scripts/train_one_loss.py $architecture $classes $model_path$classes $csv_path $img_path --image_size=$image_size --loss=$loss --optimizer=$optimizer --learning_rate=$learning_rate --batch_size=$batch_size --epochs=$epochs --lr_scheduler='plateau' --es_patience=5" -o $log_output
  done
done

# Deactivate the virtual environment
deactivate
