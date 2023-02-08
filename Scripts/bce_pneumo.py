from lib.dataset import CheXpert
from lib.trainer import Trainer
from lib import models

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("architecture", help="Input backbone model to train.", type=str)
    args = parser.parse_args()

    architecture = args.architecture

    csv_path = '//hastur/scratch/hekalo/Datasets/CheXpert-v1.0-small/'
    img_path = '//hastur/scratch/hekalo/Datasets/'

    classes = ["Pneumonia"]

    trainSet = CheXpert(csv_path=csv_path + 'train.csv', image_root_path=img_path, use_upsampling=False,
                        use_frontal=True, image_size=320, mode='train', train_cols=classes)
    valSet = CheXpert(csv_path=csv_path + 'valid.csv', image_root_path=img_path, use_upsampling=False,
                      use_frontal=True, image_size=320, mode='valid', train_cols=classes)

    model = models.sequential_model(architecture, trainSet.num_classes)

    trainer = Trainer(model=model,
                      train_set=trainSet,
                      valid_set=valSet,
                      loss="bce",
                      optimizer="adam",
                      learning_rate=0.01,
                      epochs=100,
                      batch_size=32,
                      early_stopping_patience=10,
                      lr_scheduler="reduce",
                      plateau_patience=5)

    model_path = "//hastur/scratch/hekalo/Models/labels_chexpert/bce/pneumonia/"
    model_name_base = architecture

    trainer.train(model_path, model_name_base)
