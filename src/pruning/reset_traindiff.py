import pandas as pd
from datetime import datetime
import platform
if platform.system() == 'Windows':
    filepath = r"C:\Users\Finn\Desktop\Informatik\4. Semester\Bachelor-Arbeit\Framework new\chest-x-ray-classifier\configs\local_train_difficulties.csv"
    base_path = "C:/Users/Finn/Desktop/Informatik/4. Semester/Bachelor-Arbeit/Framework new/chest-x-ray-classifier/configs/old_pi_files/"
else:
    filepath = "./configs/train_with_difficulties.csv"
    base_path = "./configs/old_pi_files/"
df = pd.read_csv(filepath)

d = f"{str(datetime.now().date())}_{str(datetime.now().hour)}_{str(datetime.now().minute)}"

df.to_csv(f"{base_path}/pi_file_{d}.csv")
print(f"exported old to {d}")

cols = "Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices".split(",")

df[cols].to_csv(filepath)