from typing import Tuple, List
from dataclasses import dataclass, field

from params.base_params import BaseParams


@dataclass
class DatasetParams(BaseParams):
    name = "Dataset Parameters"

    dataset: str = "chexpert"  # [chexpert | ...]

    image_root_path: str = "./"
    csv_path: str = "./"
    assert_images_exist: bool = False
    verbose: bool = True

    image_format: str = "jpeg"  # dicom or jpeg
    image_size: Tuple[int] = (320, 320)
    normalization: str = "imagenet"  # imagenet, None or custom one depending on dataset

    augment: bool = True
    mode: str = 'train'
    shuffle: bool = True

    # label smoothing
    lsr_method: str = "dam"  # [dam | pham | None]

    # lsr dam params
    lsr_one_cols: List[str] = field(default_factory=lambda: [])
    lsr_zero_cols: List[str] = field(default_factory=lambda: [])

    # lsr pham params
    lsr_lower: float = 0.1
    lsr_upper: float = 0.9


@dataclass
class CheXpertParams(DatasetParams):
    name = "CheXpert Parameters"

    scan_orientation: str = "frontal"  # [frontal | lateral | all]
    scan_projection: str = "all"  # [all | ap | pa]

    # labels
    train_labels: List[str] = field(default_factory=lambda:
                                    ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])

    # upsampling
    use_upsampling: bool = False
    upsample_labels: List[str] = field(default_factory=lambda: ['Cardiomegaly', 'Consolidation'])

    # lsr dam params
    lsr_one_cols: List[str] = field(default_factory=lambda:
                                    ['Edema', 'Atelectasis'])
    lsr_zero_cols: List[str] = field(default_factory=lambda:
                                     ['Cardiomegaly', 'Consolidation', 'Pleural Effusion'])

    # lsr pham params
    lsr_lower: float = 0.55
    lsr_upper: float = 0.85


dataset_params_selector = {
    "chexpert": CheXpertParams
}


@dataclass
class AugmentationParams(BaseParams):
    name = "Augmentation Parameters"

    randomAffine: bool = True
    degrees: Tuple[int, int] = (-15, 15)
    translate: Tuple[float, float] = (0.05, 0.05)
    scale: Tuple[float, float] = (0.95, 1.05)
    fill: int = 128

    randomResizedCrop: bool = False
    crop_scale: Tuple[float, float] = (0.9, 1.0)

    horizontalFlip: bool = True
    verticalFlip: bool = False
