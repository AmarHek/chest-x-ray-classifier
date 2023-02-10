from lib.dataset import CheXpert
from lib.trainer import Trainer
from lib import models

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--architecture", help="Input backbone model to train.", type=str)
    parser.add_argument("--classes", type=str)

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--csv_path", type=str, default='/scratch/hekalo/Datasets/CheXpert-v1.0-small/')
    parser.add_argument("--img_path", type=str, default='/scratch/hekalo/Datasets/')
    parser.add_argument("--image_size", type=int, default=320)

    parser.add_argument("--loss", type=str, default="bce")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--update_steps", type=int, default=1000)
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--lr_scheduler", type=str, default="plateau")

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

    architecture = args.architecture
    classes = classes_dict[args.classes]

    model_path = args.model_path
    csv_path = args.csv_path
    img_path = args.img_path
    image_size = args.image_size

    loss = args.loss
    optimizer = args.optimizer
    learning_rate = args.learning_rate

    batch_size = args.batch_size
    epochs = args.epochs

    update_steps = args.update_steps
    es_patience = args.es_patience
    lr_scheduler = args.lr_scheduler

    trainSet = CheXpert(csv_path=csv_path + 'train.csv', image_root_path=img_path, use_upsampling=False,
                        use_frontal=True, image_size=image_size, mode='train', train_cols=classes)
    valSet = CheXpert(csv_path=csv_path + 'valid.csv', image_root_path=img_path, use_upsampling=False,
                      use_frontal=True, image_size=image_size, mode='valid', train_cols=classes)

    model = models.sequential_model(architecture, trainSet.num_classes)

    trainer = Trainer(model=model,
                      train_set=trainSet,
                      valid_set=valSet,
                      loss=loss,
                      optimizer=optimizer,
                      learning_rate=learning_rate,
                      epochs=epochs,
                      batch_size=batch_size,
                      update_steps=update_steps,
                      early_stopping_patience=es_patience,
                      lr_scheduler=lr_scheduler,
                      plateau_patience=5)

    trainer.train(model_path, architecture)
