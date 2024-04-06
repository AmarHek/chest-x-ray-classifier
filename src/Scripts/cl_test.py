import argparse

import params
from makers import ClTester
from utils import load_yaml, dir_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_yaml",
                        help="Path to config yaml with all necessary settings.", type=dir_path)
    args = parser.parse_args()
    return args


def merge_args(args,
               testParams,
               testDataParams,
               augmentParams):
    pass


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config_yaml)

    # since we offer multiple datasets, we need to select the correct params class
    datasetParamsClass = params.dataset_params_selector[cfg["testDataParams"]["dataset"]]

    testParams = params.TestParams()
    testParams.load_from_dict(cfg["testParams"])
    testDataParams = datasetParamsClass()
    testDataParams.load_from_dict(cfg["testDataParams"])
    if "augmentParams" in cfg.keys():
        augmentParams = params.AugmentationParams()
        augmentParams.load_from_dict(cfg["augmentParams"])
    else:
        augmentParams = None

    merge_args(args, testParams, testDataParams, augmentParams)

    tester = Tester(testParams, testDataParams, augmentParams)

    tester.test()
