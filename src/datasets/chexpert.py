import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
import os

from params.dataset_params import ChexpertParams, AugmentationParams
from preprocessing import means, stds
from util.util import check_files


class CheXpert(Dataset):

    def __init__(self,
                 params: ChexpertParams,
                 augmentationParams: AugmentationParams = None,
                 seed: int = 42069):

        # Assertions
        assert os.path.isdir(params.image_root_path), 'Need a valid path for the images!'
        assert os.path.exists(params.csv_path), 'Need a valid path for the csv!'

        # set parameters
        self.mode = params.mode
        self.image_size = params.image_size
        self._images_list = []
        self._labels_list = []
        self.df = None

        self.value_counts_dict = {}
        self.train_labels = params.train_labels
        self.image_ratio = None
        self.image_ratio_list = None

        # get normalization parameters
        if params.normalization is not None:
            assert params.normalization in means.keys(), 'Invalid normalization!'
            self.__mean__ = means[params.normalization]
            self.__std__ = stds[params.normalization]
        else:
            self.__mean__ = None
            self.__std__ = None

        self.image_root_path = params.image_root_path
        print('Number of classes: [%d]' % len(self.train_labels))

        # load data from csv
        self.df = pd.read_csv(params.csv_path)

        # check if chosen columns are valid
        self._assert_train_labels()

        # filter dataframe, only filter for train and val
        if self.mode != 'test':
            self._filter_scan_type(params.scan_orientation, params.scan_projection)

        # get number of images in dataset
        self.num_images = len(self.df)

        # up-sample selected cols, but only for train and val
        if self.mode != "test" and params.use_upsampling:
            self._upsample_images(params.upsample_labels)

        # label smoothing
        self._apply_lsr(params.lsr_method, **params.to_dict())

        # shuffle data
        if params.shuffle:
            self._shuffle(seed)

        # get class counts
        self.count_classes()

        # quick fix: if class contains no labels 0 or 1, set to 0
        self._fix_faulty_labels()

        # get list of images and labels
        self._labels_list = self.df[self.train_labels].values.tolist()
        self._images_list = [os.path.join(params.image_root_path, path) for path in self.df['Path'].tolist()]

        # check if all images exist
        if params.assert_images_exist:
            assert self._assert_images_exist(params.image_root_path)

        # print class counts
        if params.verbose:
            self.print_class_counts()

        # setup image processing and data augmentation
        self.aug_params = augmentationParams
        self.augment = params.augment
        self.process_image = self.get_transforms()

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # load the image in RGB format
        image = Image.open(self._images_list[idx],).convert('RGB')
        # apply image processing
        image = self.process_image(image)

        label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)

        return image, label

    @property
    def chexpert_classes(self):
        if self.mode == "test":
            cheXpert_classes = self.df.columns[1:]
        else:
            cheXpert_classes = self.df.columns[5:]

        return cheXpert_classes

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.image_ratio

    @property
    def num_classes(self):
        return len(self.train_labels)

    @property
    def data_size(self):
        return self.num_images

    def _filter_scan_type(self, scan_orientation, scan_projection):
        assert scan_orientation in ['frontal', 'lateral', 'all'], 'Invalid scan orientation!'
        if scan_orientation != 'all':
            self.df = self.df[self.df['Frontal/Lateral'] == scan_orientation.capitalize()]
        assert scan_projection in ['all', 'ap', 'pa'], 'Invalid scan projection!'
        if scan_projection != 'all':
            self.df = self.df[self.df['AP/PA'] == scan_projection.upper()]

    def _assert_train_labels(self):
        # check if all classes are valid
        assert all(col in self.chexpert_classes for col in self.train_labels), \
            'One or more classes in train_cols are incorrect!'

    def _upsample_images(self, upsample_labels):
        assert isinstance(upsample_labels, list), 'Input should be list!'
        assert all(col in self.train_labels for col in upsample_labels), \
            'Discrepancy between upsample_labels and train_labels!'

        sampled_df_list = []
        for col in upsample_labels:
            print('Upsampling %s...' % col)
            sampled_df_list.append(self.df[self.df[col] == 1])
        self.df = pd.concat([self.df] + sampled_df_list, axis=0)

    def _shuffle(self, seed):
        data_index = list(range(self.num_images))
        np.random.seed(seed)
        np.random.shuffle(data_index)
        self.df = self.df.iloc[data_index]

    def _apply_lsr(self, lsr_method, **kwargs):
        # this is for label smooth regularization based on Deep AUC paper
        if lsr_method == 'dam':
            for col in self.train_labels:
                if col in kwargs['lsr_one_cols']:
                    self.df[col].replace(-1, 1, inplace=True)
                elif col in kwargs['lsr_zero_cols']:
                    self.df[col].replace(-1, 0, inplace=True)
                else:
                    self.df[col].replace(-1, 1, inplace=True)
        # this is for label smooth regularization based on Pham et al.
        elif lsr_method == 'pham':
            for col in self.train_labels:
                a = kwargs['lsr_lower']
                b = kwargs['lsr_upper']
                self.df[col] = self.df[col].apply(self.label_smoothing_regularization,
                                                  args=(a, b))
        else:
            raise NotImplementedError('Invalid LSR method!')

    def _assert_images_exist(self, image_root_path) -> bool:
        # check if all images exist
        # use multi-threading to speed up
        image_path_list = [os.path.join(image_root_path, path) for path in self._images_list]
        print('Checking if all images exist...')
        file_results = check_files(image_path_list)

        # check if all files exist
        if all(file_results):
            print('...done')
            return True
        else:
            # print all files that do not exist
            for idx, result in enumerate(file_results):
                if not result:
                    print('File %s does not exist!' % image_path_list[idx])
            return False

    def _fix_faulty_labels(self):
        for class_key, select_col in enumerate(self.train_labels):
            class_dict = self.value_counts_dict[class_key]
            if 0.0 not in class_dict.keys():
                class_dict[0.0] = 0
            if 1.0 not in class_dict.keys():
                class_dict[1.0] = 0

    def count_classes(self):
        for class_key, train_label in enumerate(self.train_labels):
            class_value_counts_dict = self.df[train_label].value_counts().to_dict()
            self.value_counts_dict[class_key] = class_value_counts_dict

    def print_class_counts(self):
        print('-' * 30)
        image_ratio_list = []
        for class_key, select_col in enumerate(self.train_labels):
            image_ratio = self.value_counts_dict[class_key][1] / (
                    self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
            image_ratio_list.append(image_ratio)
            print('Found %s images in total, %s positive images, %s negative images' % (
                self.num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
            print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, image_ratio))
            print()
        self.image_ratio = np.mean(image_ratio_list)
        self.image_ratio_list = image_ratio_list
        print('-' * 30)

    def get_transforms(self):
        transforms = []
        if self.augment:
            if self.aug_params.randomAffine:
                transforms.append(tfs.RandomAffine(degrees=self.aug_params.degrees,
                                                   scale=self.aug_params.scale,
                                                   translate=self.aug_params.translate,
                                                   fill=self.aug_params.fill))
            if self.aug_params.randomResizedCrop:
                transforms.append(tfs.RandomResizedCrop(size=self.aug_params.crop_scale, ratio=(1, 1)))
            if self.aug_params.horizontalFlip:
                transforms.append(tfs.RandomHorizontalFlip(0.5))
            if self.aug_params.verticalFlip:
                transforms.append(tfs.RandomVerticalFlip(0.5))

        transforms.append(tfs.Resize(size=self.image_size))
        transforms.append(tfs.ToTensor())
        transforms.append(tfs.Normalize(mean=self.__mean__, std=self.__std__))

        return tfs.Compose(transforms)

    @staticmethod
    def label_smoothing_regularization(value, a=0.55, b=0.85):
        assert a < b, 'a must be smaller than b!'
        assert a >= 0 and b <= 1, 'a and b must be between 0 and 1!'

        if value == -1:
            return round(np.random.uniform(a, b), 5)
        else:
            return value


if __name__ == '__main__':
    dataset_params = ChexpertParams()
    dataset_params.image_root_path = "F:/"
    dataset_params.csv_path = "F:/CheXpert-v1.0/train.csv"
    dataset_params.use_upsampling = True
    augmentation_params = AugmentationParams()

    print(dataset_params.upsample_labels, dataset_params.train_labels)

    dataset = CheXpert(dataset_params, augmentation_params)

    import torch

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=1, drop_last=True, shuffle=True)

    for batch in dataloader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
