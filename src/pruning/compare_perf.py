import pandas as pd
import argparse
import re
import platform
import yaml
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your script.')

# Add arguments
parser.add_argument('yaml_file', type=str, default="configs/compare_perf_config.yaml", help='Specifies config file which contains files to be compared')


# Parse the command-line arguments
args = parser.parse_args()
yaml_file_path = args.yaml_file
with open(yaml_file_path, "r") as file:
    yaml_data = yaml.safe_load(file)

paths = yaml_data['compParams']
# Access the arguments
#print("Argument 1:", args.arg1)
#print("Argument 2:", args.arg2)

def get_change_in_prob_row(l1,l2):
    sum = 0
    for i in range(len(l1)):
        sum+=abs(l1[i]-l2[i])
    return sum
def get_change_in_prob_overall(df,no,dfs):
    l1cols = [f"{i}_{dfs[0]}" for i in range(no)]
    l2cols = [f"{i}_{dfs[1]}" for i in range(no)]
    res = []
    for index, row in df.iterrows():
        l1 = df.loc[index,:][l1cols].tolist()
        l2 = df.loc[index,:][l2cols].tolist()
        #print(l1,l2)
        res.append(get_change_in_prob_row(l1,l2))
    return res
def compare_performances(path_normal,
                        path_pruned,
                        path_ground_truth,
                        export_result, 
                        pruning_method):
    if platform.system() == 'Windows':
        df1 = pd.read_csv(f"{path_normal}\\chexpert\\outputs.csv")
        df2 = pd.read_csv(f"{path_pruned}\\chexpert\\outputs.csv")
    else:
        df1 = pd.read_csv(f"{path_normal}/chexpert/outputs.csv")
        df2 = pd.read_csv(f"{path_pruned}/chexpert/outputs.csv")

    illness = yaml_data['compParams']['illness']
    no_illness = len(illness)

    tdf = pd.read_csv(path_ground_truth)
    for col in illness:
        tdf[col].replace(-1, 1, inplace=True)
    # Concatenate based on 'filename'
    result = pd.merge(df1, df2, on='filename', how='outer')
    columns_to_transform = result.columns.drop("filename").tolist()

    #Difficulty measure 1: calculate pruning impact as difference in probabilities
    result[f'PI_{pruning_method}_prob'] = get_change_in_prob_overall(result,no_illness,['x','y'])
    #convert output to boolean values
    result.loc[:, columns_to_transform] = result.loc[:, columns_to_transform].map(lambda x: 1 if x >= 0.5 else 0)
    result[f'PI_{pruning_method}_hamming'] = get_change_in_prob_overall(result,no_illness,['x','y'])
    #shorten filenames to match format of tdf
    result.filename = result.filename.map(lambda x: x[x.find("CheXpert-v1.0-small"):])

    #merge result and truth
    result = pd.merge(result,tdf, left_on='filename',right_on='Path',how='outer')

    #get performance for both dfs for each illness
    #illness = tdf.columns.tolist()[6:]
    #illness = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    
    for i in range(no_illness):
        for df in ['x','y']:
            result[f"perf_{df}_{i}"] = (result[f'{i}_{df}'] == result[illness[i]]).astype(int)

            #get overall performance
            if i == no_illness-1:
                result[f"perf_{df}_total"] = result[[f"perf_{df}_{k}" for k in range(no_illness)]].sum(axis=1)

    #get pruning impact
    result[f'PI_{pruning_method}_acc'] = result[f"perf_x_total"] - result[f"perf_y_total"]
    print('succesfully inferred pruning impact')
    
    #add pruning impact to difficulties file

    ddf = pd.read_csv(export_result)
    #3. difficult measure: difference in accuracy
    exp_df = pd.merge(ddf,result[['filename',f'PI_{pruning_method}_prob',f'PI_{pruning_method}_acc',f'PI_{pruning_method}_hamming']],left_on = 'Path',right_on='filename',how="outer")
    #
    cols_to_drop = []
    for c in exp_df.columns: 
        if "filename" in c or "Unnamed" in c:
            cols_to_drop.append(c)
    exp_df = exp_df.drop(columns=cols_to_drop)
    exp_df.to_csv(export_result)

path_normal = paths['path_normal']
path_pruned = paths['path_pruned']
path_ground_truth = paths['path_ground_truth']
export_result = paths['export_result']
pruning_method = paths['pruning_method']

for i,p in enumerate(path_pruned):
    print(f"{i+1}/{len(path_pruned)}: {p}")
    compare_performances(path_normal,p , path_ground_truth,export_result,pruning_method[i])