import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from models import PretrainedModel
from models import load_model
import yaml
from params import PretrainedModelParams
import os

def get_all_modules(module):
    all_modules = []

    # If the module has children, recursively collect modules from each child
    for child_module in module.children():
        all_modules.extend(get_all_modules(child_module))

    # Append the current module to the list of all modules
    all_modules.append(module)

    return all_modules

def save_pruned_model(model,modelParams, optimizer, current_epoch, validation_metric,current_score, best_score,save_path):
    torch.save({"model": model.state_dict(),
                "modelParams": modelParams,
                "optimizer": optimizer,
                "epoch": current_epoch,
                "validation_metric": validation_metric
                #"score": current_score,
                #"best_score": best_score},
                },save_path)
    print("saved succesfully")

# instantiate given model
model_path = "C:\\Users\\Finn\\Desktop\\Informatik\\4. Semester\\Bachelor-Arbeit\\Framework new\\chest-x-ray-classifier\\experiments\\Baseline_2024-03-25_11-17-07\\Baseline_2024-03-25_11-17-07_1_best.pth"

#yaml_file_path = model_path+"\\params.yaml"

# Read YAML file
#with open(yaml_file_path, 'r') as file:
#    yaml_data = yaml.safe_load(file)


model_params = PretrainedModelParams()
model_params.backbone = "densenet121"
#model_params.backbone = yaml_data['modelParams']['backbone']

model_params.num_heads = 4
model_params.lam = 0.1
model_params.num_classes = 14

loaded_dict = torch.load(model_path)
model_params = loaded_dict["modelParams"]

#current_epoch = loaded_dict["current_epoch"]
#train_params =  loaded_dict["trainParams"]

#optimizer =  train_params.optimizer,
optimizer = 'adam'
validation_metric = 'loss'
#"score": current_score,
#"best_score": best_score



model = load_model(model_params)
modules = get_all_modules(model)

#test pruning
pruning_ratio = .5
pruning_level = "Global"
pruning_structured = False
pruning_random = False
pruning_counter = 0

if pruning_level == "Global":
    if not pruning_structured and not pruning_random:
        modules_global = tuple([(m,"weight") for m in modules if hasattr(m,'weight')])
        
        prune.global_unstructured(modules_global, pruning_method=prune.L1Unstructured, amount=pruning_ratio)
        for i in range(len(modules)):
            if hasattr(modules[i],"weight"):
                prune.remove(modules[i],"weight")
    else:
        print("Selected invalid pruning parameters. Global Pruning only allows not-random, unstructured")

        
if pruning_level == "Local":
    for i in range(len(modules)):
        try:
            if pruning_structured and pruning_random:
                prune.random_structured(modules[i], name="weight", amount=pruning_ratio,dim=0)
            
            elif pruning_structured and not pruning_random:
                prune.ln_structured(modules[i], name="weight", amount=pruning_ratio,dim=0)
            
            elif not pruning_structured and pruning_random:
                prune.random_unstructured(modules[i], name="weight", amount=pruning_ratio)
            elif not pruning_structured and not pruning_random:
                prune.l1_unstructured(modules[i], "weight", pruning_ratio, importance_scores=None)
            else:
                print("Selected invalid pruning paramters. Local Prunign only allows...")


            pruning_counter += 1
            print(pruning_counter)
            modules[i]._forward_pre_hooks
            prune.remove(modules[i],'weight')
        except AttributeError as e:
            print(e)

pruning_structured_str = "structured" if pruning_structured else "unstructured"
pruning_random_str = "random" if pruning_random else "l1"
#print(model.state_dict())
model_name = f"{model_path[-39:-4]}_{int(pruning_ratio*100)}_{pruning_level}_{pruning_structured_str}_{pruning_random_str}"
pruned_model_path = f"C:\\Users\\Finn\\Desktop\\Informatik\\4. Semester\\Bachelor-Arbeit\\Framework new\\chest-x-ray-classifier\\experiments\\Pruned Models\\{model_name}\\{model_name}.pth"
print(model_params)

os.makedirs(f"C:\\Users\\Finn\\Desktop\\Informatik\\4. Semester\\Bachelor-Arbeit\\Framework new\\chest-x-ray-classifier\\experiments\\Pruned Models\\{model_name}")
save_pruned_model(model,model_params,optimizer,None,validation_metric,None,None,pruned_model_path)