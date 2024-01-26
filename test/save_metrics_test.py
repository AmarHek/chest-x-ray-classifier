import pandas as pd

# Your performance metrics dictionary
performance_metrics = {
    'f1_class': [0.82, 0.78],
    'auc': 0.91,
    'precision': 0.85,
    'recall': 0.75,
    'f1': 0.80,
    'auc_class': [0.84, 0.97],
    'precision_class': [0.82, 0.88]
    # Add other metrics as needed
}

labels = ["Cardiomegaly", "Effusion"]

# Separate average and class-specific metrics
average_metrics = {}
class_metrics = {}

for metric, value in performance_metrics.items():
    if metric.endswith('_class'):
        class_metrics[metric.replace('_class', '')] = value
    else:
        average_metrics[metric] = value

# Create DataFrames for average and class-specific metrics
# average is straightforward
df_average = pd.DataFrame(list(average_metrics.items()), columns=['Metric', 'Average'])
# for class we first create an empty DataFrame
df_class = pd.DataFrame(columns=['Metric'] + labels)
# populate the DataFrame
for metric, values in class_metrics.items():
    df_class = df_class.append({'Metric': metric, **dict(zip(labels, values))}, ignore_index=True)

# Concatenate DataFrames along columns
# result_df = pd.concat([df_average, df_class], axis=1)
result_df = pd.merge(left=df_average, right=df_class, on='Metric', how='outer')

# Replace NaN with -1
result_df = result_df.fillna(-1)

print(result_df)
