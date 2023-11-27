from chexpert import CheXpert


def load_dataset(datasetParams, augmentationParams):
    if datasetParams.dataset == "chexpert":
        dataset = CheXpert(datasetParams, augmentationParams)
    else:
        raise NotImplementedError(f"Dataset {datasetParams.dataset} not implemented yet")

    return dataset
