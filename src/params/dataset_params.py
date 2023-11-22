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
    image_size: int = 320

    augment: bool = True
    mode: str = 'train'
    shuffle: bool = True

    # label smoothing
    use_lsr_random: bool = True
    use_lsr_dam: bool = False


@dataclass
class ChexpertParams(DatasetParams):
    name = "CheXpert Parameters"

    use_frontal: bool = True
    use_ap: bool = False
    use_pa: bool = False

    # labels
    train_cols: List[str] = field(default_factory=lambda:
                                  ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])

    # upsampling
    use_upsampling: bool = True
    upsampling_cols: List[str] = field(default_factory=lambda: ['Cardiomegaly', 'Consolidation'])


@dataclass
class AugmentationParams(BaseParams):
    name = "Augmentation Parameters"

    randomAffine: bool = True
    degrees: Tuple[int, int] = (-15, 15)
    translate: Tuple[float, float] = (0.05, 0.05)
    scale: Tuple[float, float] = (0.95, 1.05)
    fill: int = 128

    randomCrop: bool = False
    crop_size: Tuple[int, int] = (256, 256)

    horizontalFlip: bool = False
    verticalFlip: bool = False
