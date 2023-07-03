import os

from src.datasets.chexpert import CheXpert
from src.makers.tester import Tester

if __name__ == "__main__":

    model_file = "/home/ls6/hekalo/models.txt"
    csv_path = "/scratch/hekalo/Datasets/CheXpert-v1.0-small/"
    img_path = "/scratch/hekalo/Datasets/"

    image_size = 320

    base_path = "scratch/hekalo/Models/labels_chexpert/bce/"

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
        print(model_base_path)
        models = []
        print("hello")
        print(os.path.exists(model_base_path))
        for (dirpath, dirnames, filenames) in os.walk(model_base_path):
            print("hello")
            print(dirpath, dirnames, filenames)
            for file in filenames:
                print(file)
                if file.endswith("_best.pt"):
                    models.append(os.path.join(dirpath, file))

        print(models)

        # testSet = CheXpert(csv_path=csv_path, image_root_path=img_path, use_upsampling=False,
        #                    image_size=image_size, mode='test', train_cols=labels)
#
        # tester = Tester(test_set=testSet, classes=labels, models_file=model_file, batch_size=1, metrics=["f1"])
#
        # tester.test()
        # tester.save_metrics("/scratch/hekalo/Evaluations/labels_chexpert/bce/metrics_%s.json" % cls)
        # tester.save_raw_results("/scratch/hekalo/Evaluations/labels_chexpert/bce/output_raw_%s.json" % cls)
#
        # del testSet, tester
