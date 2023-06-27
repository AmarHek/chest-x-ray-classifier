import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
from PIL import Image
import pandas as pd
import os


class CheXpert(Dataset):

    def __init__(self,
                 csv_path: str,
                 image_root_path: str,
                 image_size: int = 320,
                 use_frontal: bool = True,
                 use_pa: bool = False,
                 use_ap: bool = False,
                 use_upsampling: bool = True,
                 use_lsr_random: bool = True,
                 use_lsr_dam: bool = False,
                 shuffle: bool = True,
                 seed: int = 42069,
                 verbose: bool = True,
                 assert_images: bool = False,
                 upsampling_cols: list[str] = None,
                 train_cols: list[str] = None,
                 mode: str = 'train'):

        # Assertions
        assert os.path.isdir(image_root_path), 'Need a valid path for the images!'
        assert os.path.exists(csv_path), 'Need a valid path for the csv!'
        assert not (use_pa is True and use_ap is True), 'Cannot filter out both ap and pa!'
        if use_lsr_random is True and use_lsr_dam is True:
            print('Warning: Both LSR methods set to True. Only lsr_dam will be used.')

        if upsampling_cols is None:
            upsampling_cols = ['Cardiomegaly', 'Consolidation']
        if train_cols is None:
            train_cols = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

        self.mode = mode
        self.image_size = image_size
        self._images_list = []
        self._labels_list = []
        self.value_counts_dict = {}
        self.select_cols = train_cols
        self.imratio = None
        self.imratio_list = None
        self.df = None

        print('Number of classes: [%d]' % len(train_cols))

        # load data from csv
        self.df = pd.read_csv(csv_path)

        if mode == "test":
            cheXpert_classes = self.df.columns[1:]
        else:
            cheXpert_classes = self.df.columns[5:]
        assert all(col in cheXpert_classes for col in train_cols), \
            'One or more classes in train_cols are incorrect!'
        if use_upsampling:
            assert all(col in cheXpert_classes for col in upsampling_cols), \
                'One or more classes in upsampling_cols are incorrect!'
            assert all(col in upsampling_cols for col in train_cols), \
                'Discrepancy between upsampling_cols and train_cols!'

        # filter dataframe
        # only use frontal scans
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        # only use PA scans
        if use_pa:
            self.df = self.df[self.df['AP/PA'] == 'PA']
        # only use AP scans
        if use_ap:
            self.df = self.df[self.df['AP/PA'] == 'AP']

        self._num_images = len(self.df)

        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # this is for label smooth regularization based on Deep AUC paper
        if use_lsr_dam:
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']:
                    self.df[col].replace(-1, 1, inplace=True)
                elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                    self.df[col].replace(-1, 0, inplace=True)
                else:
                    self.df[col].replace(-1, 1, inplace=True)

        # this is for label smooth regularization based on Pham et al.
        if use_lsr_random:
            np.random.seed(seed)
            for col in train_cols:
                self.df[col] = self.df[col].apply(CheXpert.label_smoothing_regularization)

        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        for class_key, select_col in enumerate(train_cols):
            class_value_counts_dict = self.df[select_col].value_counts().to_dict()
            self.value_counts_dict[class_key] = class_value_counts_dict

        # quick fix: if class contains no labels 0 or 1, set to 0
        for class_key, select_col in enumerate(train_cols):
            class_dict = self.value_counts_dict[class_key]
            if 0.0 not in class_dict.keys():
                class_dict[0.0] = 0
            if 1.0 not in class_dict.keys():
                class_dict[1.0] = 0

        self._labels_list = self.df[train_cols].values.tolist()
        self._images_list = [os.path.join(image_root_path, path) for path in self.df['Path'].tolist()]
        if assert_images:
            print("Checking if all images exist...")
            assert self.assert_images_exist(image_root_path), "One or more image paths are invalid!"
            print("...done")

        if verbose:
            print('-' * 30)
            imratio_list = []
            for class_key, select_col in enumerate(train_cols):
                imratio = self.value_counts_dict[class_key][1] / (
                        self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                imratio_list.append(imratio)
                print('Found %s images in total, %s positive images, %s negative images' % (
                    self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                print()
            self.imratio = np.mean(imratio_list)
            self.imratio_list = imratio_list
            print('-' * 30)

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def assert_images_exist(self, image_root_path) -> bool:
        for image_path in tqdm.tqdm(self._images_list):
            if not os.path.isfile(os.path.join(image_root_path, image_path)):
                print("%s does not exist!" % os.path.join(image_root_path, image_path))
                return False

        return True

    @staticmethod
    def label_smoothing_regularization(value, a=0.55, b=0.85):
        if value == -1:
            return round(np.random.uniform(a, b), 5)
        else:
            return value

    @staticmethod
    def image_augmentation(image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05),
                                                fill=128)])  # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)

        if self.mode == 'train':
            image = CheXpert.image_augmentation(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = image / 255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        image = (image - __mean__) / __std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        return image, label


if __name__ == '__main__':
    csv_path = '\\\\hastur\\scratch\\hekalo\\Datasets\\CheXpert-v1.0-small\\'
    img_path = '\\\\hastur\\scratch\\hekalo\\Datasets\\'
    trainSet = CheXpert(csv_path=csv_path + 'train.csv', image_root_path=img_path, use_upsampling=False, use_frontal=True,
                        image_size=320, mode='train', train_cols=["Pneumonia"])
    testSet = CheXpert(csv_path=csv_path + 'valid.csv', image_root_path=img_path, use_upsampling=False, use_frontal=True,
                       image_size=320, mode='valid', train_cols=["Pneumonia"])

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=32, num_workers=2, drop_last=True, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, drop_last=False, shuffle=False)


