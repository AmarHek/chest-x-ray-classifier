from lib.dataset import CheXpert
from lib.trainer import Trainer
from lib import models

if __name__ == '__main__':

    csv_path = '//hastur/scratch/hekalo/Datasets/CheXpert-v1.0-small/'
    img_path = '//hastur/scratch/hekalo/Datasets/'

    classes = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation",
               "Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]

    trainSet = CheXpert(csv_path=csv_path + 'train.csv', image_root_path=img_path, use_upsampling=False,
                        use_frontal=True, image_size=100, mode='train', train_cols=classes)
    valSet = CheXpert(csv_path=csv_path + 'valid.csv', image_root_path=img_path, use_upsampling=False,
                      use_frontal=True, image_size=100, mode='valid', train_cols=classes)

    model = models.sequential_model("densenet121", trainSet.num_classes)

    trainer = Trainer(model=model,
                      train_set=valSet,
                      valid_set=valSet,
                      loss="bce",
                      optimizer="adam",
                      learning_rate=0.01,
                      epochs=100,
                      batch_size=50,
                      update_steps=1,
                      early_stopping_patience=10,
                      lr_scheduler="reduce",
                      plateau_patience=5)

    model_path = "//hastur/scratch/hekalo/Models/test/"
    model_name_base = "densenet121_bce_adam"

    trainer.train(model_path, model_name_base)
