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

    tdf = pd.read_csv(path_ground_truth)
    # Concatenate based on 'filename'
    result = pd.merge(df1, df2, on='filename', how='outer')

    columns_to_transform = result.columns.drop("filename").tolist()
    #convert output to boolean values
    result.loc[:, columns_to_transform] = result.loc[:, columns_to_transform].applymap(lambda x: 1 if x >= 0.5 else 0)
    #shorten filenames to match format of tdf
    result.filename = result.filename.map(lambda x: x[36:])

    #merge result and truth
    result = pd.merge(result,tdf, left_on='filename',right_on='Path',how='outer')

    #get performance for both dfs for each illness
    illness = tdf.columns.tolist()[7:]
    for i in range(12):
        for df in ['x','y']:
            result[f"perf_{df}_{i}"] = (result[f'{i}_{df}'] == result[illness[i]]).astype(int)

            #get overall performance
            if i == 11:
                result[f"perf_{df}_total"] = result[[f"perf_{df}_{k}" for k in range(12)]].sum(axis=1)

    #get pruning impact
    result[f'PI_{pruning_method}'] = result[f"perf_x_total"] - result[f"perf_y_total"]
    print('succesfully inferred pruning impact')
    
    #add pruning impact to difficulties file

    ddf = pd.read_csv(export_result)
    exp_df = pd.merge(ddf,result[['filename',f'PI_{pruning_method}']],left_on = 'Path',right_on='filename',how="outer")
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

compare_performances(path_normal, path_pruned, path_ground_truth,export_result,pruning_method)