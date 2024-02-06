import pandas as pd

# create dummy metrics dict
performance_metrics = {
    'f1_class': [0.82, 0.78],
    'auc_class': [0.84, 0.97],
    'precision_class': [0.82, 0.88],
    'rec_class': [0.75, 0.82]
}
labels = ["Cardiomegaly", "Effusion"]
df_class = pd.DataFrame(columns=['Metric'] + labels)

for metric, values in performance_metrics.items():
    temp_df = pd.DataFrame({'Metric': metric, **dict(zip(labels, values))}, index=[0])
    df_class = pd.concat([df_class, temp_df], ignore_index=True)

print(df_class)
