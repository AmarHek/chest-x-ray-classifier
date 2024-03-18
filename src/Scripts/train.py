import argparse

import params
from makers import Trainer
from utils import load_yaml, dir_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_yaml",
                        help="Path to config yaml with all necessary settings.", type=dir_path)
    parser.add_argument("--learning_rate",
                        help="Learning rate for the optimizer.", type=float)
    parser.add_argument("--backbone",
                        help="Backbone for the model.", type=str)
    parser.add_argument("--labels",
                        help="Labels to train on.", type=str)
    parser.add_argument("--batch_size",
                        help="Batch size for the training.", type=int)
    args = parser.parse_args()
    return args


def merge_args(args,
               trainParams,
               modelParams,
               trainDataParams,
               valDataParams,
               augmentParams):
    # Learning Rate
    if args.learning_rate is not None:
        trainParams.learning_rate = args.learning_rate
        trainParams.exp_name = trainParams.exp_name + f"_lr{args.learning_rate}"
        print(f"Set learning rate to {args.learning_rate} and exp_name "
              f"to {trainParams.exp_name}")
    if args.backbone is not None:
        modelParams.backbone = args.backbone
        trainParams.exp_name = trainParams.exp_name + f"_{args.backbone.replace('/', '_')}"
        print(f"Set backbone to {args.backbone} and exp_name "
              f"to {trainParams.exp_name}")
    if args.labels is not None:
        label_num = len(args.labels.split(","))
        trainDataParams.train_labels = args.labels.split(",")
        valDataParams.train_labels = args.labels.split(",")
        trainParams.exp_name = trainParams.exp_name + f"_{args.labels}"
        # update the number of classes in the model
        modelParams.num_classes = label_num
        print(f"Set labels to {args.labels} and updated num_classes to {label_num}")
    if args.batch_size is not None:
        trainParams.batch_size = args.batch_size
        print(f"Set batch size to {args.batch_size}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config_yaml)

    # since we offer multiple datasets and models, we need to select the correct params class
    datasetParamsClass = params.dataset_params_selector[cfg["trainDataParams"]["dataset"]]
    modelParamsClass = params.model_param_selector[cfg["modelParams"]["model_type"]]

    trainParams = params.TrainParams()
    trainParams.load_from_dict(cfg["trainParams"])
    modelParams = modelParamsClass()
    modelParams.load_from_dict(cfg["modelParams"])
    trainDataParams = datasetParamsClass()
    trainDataParams.load_from_dict(cfg["trainDataParams"])
    valDataParams = datasetParamsClass()
    valDataParams.load_from_dict(cfg["valDataParams"])
    augmentParams = params.AugmentationParams()
    augmentParams.load_from_dict(cfg["augmentParams"])

    merge_args(args, trainParams, modelParams, trainDataParams, valDataParams, augmentParams)

    trainer = Trainer(trainParams,
                      modelParams,
                      trainDataParams,
                      valDataParams,
                      augmentParams)

    trainer.print_params()
    trainer.save_params()

    trainer.train()
