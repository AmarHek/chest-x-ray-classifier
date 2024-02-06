#!/bin/bash
# Activate the virtual environment
source /home/ls6/hekalo/Git/chest-x-ray-classifier/venv/bin/activate

# add project root to PYTHONPATH
PYTHONPATH="${PYTHONPATH}:/home/ls6/hekalo/Git/chest-x-ray-classifier"
export PYTHONPATH

log_output="/home/ls6/hekalo/job_output/label_dependency/job-%j.out"
error_output="/home/ls6/hekalo/job_output/label_dependency/job-%j.err"

config="/home/ls6/hekalo/Git/chest-x-ray-classifier/configs/trainconfig_label_dependency.yaml"

# Define label sets
label_sets=(
"'Cardiomegaly'"
"'Cardiomegaly,Edema'"
"'Cardiomegaly,Edema,Consolidation'"
"'Cardiomegaly,Edema,Consolidation,Pleural Effusion'"
"'Cardiomegaly,Edema,Consolidation,Pleural Effusion,Atelectasis'"
"'Edema'"
"'Edema,Consolidation'"
"'Edema,Consolidation,Pleural Effusion'"
"'Edema,Consolidation,Pleural Effusion,Atelectasis'"
"'Consolidation'"
"'Consolidation,Pleural Effusion'"
"'Consolidation,Pleural Effusion,Atelectasis'"
"'Pleural Effusion'"
"'Pleural Effusion,Atelectasis'"
"'Atelectasis'"
)

for label_set in "${label_sets[@]}"; do
  sbatch -p ls6 --gres=gpu:1 --wrap="python src/Scripts/train.py $config --labels=$label_set" -o $log_output -e $error_output
done


# Deactivate the virtual environment
deactivate
