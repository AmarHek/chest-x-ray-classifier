from src.datasets.chexpert import CheXpert
from src.makers.trainer import Trainer
from src.models import cnn_models as models

import argparse

"""
Simple script to train a model end-to-end with a single type of loss
"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("architecture", help="Input backbone model to train.", type=str)
    parser.add_argument("classes", type=str)

    parser.add_argument("model_path", type=str)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("img_path", type=str)
    parser.add_argument("--image_size", type=int, default=320)

    parser.add_argument("--loss", type=str, default="bce")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--update_steps", type=int, default=1000)
    parser.add_argument("--es_patience", type=int, default=10)

    # scheduler options
    parser.add_argument("--lr_scheduler", type=str, default="plateau")
    parser.add_argument("--plateau_patience", type=int, default=5)
    parser.add_argument("--exponential_gamma", type=float, default=0.01)
    parser.add_argument("--cyclic_lr_min", type=float, default=0.001)
    parser.add_argument("--cyclic_lr_max", type=float, default=0.01)

    args = parser.parse_args()

    classes_dict = {
        "pneumonia": ["Pneumonia"],
        "chexternal": ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
        "chexternal_pneumo": ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'Pneumonia'],
        "chexpert": ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
                     "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                     "Pleural Other", "Fracture", "Support Devices"]
    }

    print(args)
    assert args.model_path is not None, "Specify model path!"
    assert args.classes in classes_dict.keys(), "Invalid class config!"

    classes = classes_dict[args.classes]

    trainSet = CheXpert(csv_path=args.csv_path + 'train.csv', image_root_path=args.img_path, use_upsampling=False,
                        use_frontal=True, image_size=args.image_size, mode='train', train_cols=classes)
    valSet = CheXpert(csv_path=args.csv_path + 'valid.csv', image_root_path=args.img_path, use_upsampling=False,
                      use_frontal=True, image_size=args.image_size, mode='valid', train_cols=classes)

    model = models.sequential_model(args.architecture, trainSet.num_classes)

    trainer = Trainer(model=model,
                      train_set=trainSet,
                      valid_set=valSet,
                      loss=args.loss,
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      update_steps=args.update_steps,
                      early_stopping_patience=args.es_patience,
                      lr_scheduler=args.lr_scheduler,
                      plateau_patience=args.plateau_patience,
                      exponential_gamma=args.exponential_gamma,
                      cyclic_lr=(args.cyclic_lr_min, args.cyclic_lr_max))

    trainer.train(args.model_path, args.architecture)
