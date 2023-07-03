import os

from src.datasets.chexpert import CheXpert
from src.makers.tester import Tester

if __name__ == "__main__":

    model_file = "/home/ls6/hekalo/models.txt"
    csv_path = "/scratch/hekalo/Datasets/CheXpert-v1.0-small/"
    img_path = "/scratch/hekalo/Datasets/"

    image_size = 320

    base_path = "/scratch/hekalo/Models/labels_chexpert/bce/"

    classes = ["pneumonia", "chexternal", "chexternal_pneumo", "chexpert"]

    classes_dict = {
        "pneumonia": ["Pneumonia"],
        "chexternal": ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
        "chexternal_pneumo": ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'Pneumonia'],
        "chexpert": ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
                     "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                     "Pleural Other", "Fracture", "Support Devices"]
    }

    for cls in classes:
        labels = classes_dict[cls]
        model_base_path = os.path.join(base_path, cls)
        models = []
        for (dir_path, dir_names, filenames) in os.walk(model_base_path):
            for file in filenames:
                if file.endswith("_best.pt"):
                    models.append(os.path.join(dir_path, file))

        testSet = CheXpert(csv_path=csv_path, image_root_path=img_path, use_upsampling=False,
                           image_size=image_size, mode='test', train_cols=labels)

        tester = Tester(test_set=testSet, classes=labels, model_paths=models, batch_size=1)

        tester.test()
        tester.save_metrics("/scratch/hekalo/Evaluations/labels_chexpert/bce/metrics_%s.json" % cls)
        tester.save_raw_results("/scratch/hekalo/Evaluations/labels_chexpert/bce/output_raw_%s.json" % cls)

        del testSet, tester
