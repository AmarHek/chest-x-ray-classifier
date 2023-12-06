"""
Training script for CycleGAN, CycleMedGAN and Pix2Pix
"""
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
        trainParams.exp_name = trainParams.exp_name + f"_{args.backbone}"
        print(f"Set backbone to {args.backbone} and exp_name "
              f"to {trainParams.exp_name}")


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
    testDataParams = datasetParamsClass()
    testDataParams.load_from_dict(cfg["testDataParams"])
    augmentParams = params.AugmentationParams()
    augmentParams.load_from_dict(cfg["augmentParams"])

    merge_args(args, trainParams, modelParams, trainDataParams, valDataParams, augmentParams)

    trainer = Trainer(trainParams,
                      modelParams,
                      trainDataParams,
                      valDataParams,
                      augmentParams)

    trainer.print_params()
    trainer.save_params(addTestParams=True, testDataParams=testDataParams)

    trainer.train()
