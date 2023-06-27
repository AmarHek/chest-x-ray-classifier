from src.datasets.chexpert import CheXpert
from src.makers.tester import Tester

if __name__ == "__main__":

    model_file = "/home/ls6/hekalo/models.txt"

    csv_path = "/scratch/hekalo/Datasets/CheXpert-v1.0-small/"
    img_path = "/scratch/hekalo/Datasets/"
    image_size = 320

    labels = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
              "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture", "Support Devices"]

    testSet = CheXpert(csv_path=csv_path + 'test.csv', image_root_path=img_path, use_upsampling=False,
                       image_size=image_size, mode='test', train_cols=labels)

    tester = Tester(test_set=testSet, classes=labels, models_file=model_file, batch_size=1)

    tester.test()
    tester.save_metrics("/home/ls6/hekalo/metrics.json")
    tester.save_raw_results("/home/ls6/hekalo/output.json")
