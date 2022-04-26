import glob
import logging
import random

import numpy as np
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DatasetLoader(Dataset):

    def __init__(self, X, y, length=160, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.sequence_length = length


    @staticmethod
    def load_dataset(data_dir_train: str, data_fir_val:str):
        input_paths_train = []

        logger.debug(f"load_dataset: Loading dataset from {data_dir_train}")
        class_dirs_train = sorted(glob.glob(data_dir_train + '/*'))

        for idx, class_dir in enumerate(class_dirs_train):
            data_paths = [image_path for image_path in glob.glob(class_dir + '/**/*', recursive=True)
                           if image_path.split('.')[-1].lower() in ['npy']]
            input_paths_train.append(data_paths)
            logger.debug(f"load_dataset: {idx + 1} out of {len(class_dirs_train)}; "
                         f"n_images is {len(data_paths)}; class_dir is {class_dir}")

        input_paths_test = []
        class_dirs_test = sorted(glob.glob(data_fir_val + '/*'))


        logger.debug(f"load_dataset: Loading dataset from {data_fir_val}")

        for idx, class_dir in enumerate(class_dirs_test):
            data_paths = [image_path for image_path in glob.glob(class_dir + '/**/*', recursive=True)
                           if image_path.split('.')[-1].lower() in ['npy']]
            input_paths_test.append(data_paths)
            logger.debug(f"load_dataset: {idx + 1} out of {len(class_dirs_test)}; "
                         f"n_images is {len(data_paths)}; class_dir is {class_dir}")

        X_train, y_train = [], []
        X_test, y_test = [], []
        for class_idx, class_images in enumerate(input_paths_train):
            idxs = [idx for idx in range(len(class_images))]
            random.shuffle(idxs)

            X_train.extend([class_images[idx] for idx in idxs])
            y_train.extend([class_idx for _ in idxs])

        for class_idx, class_images in enumerate(input_paths_test):
            idxs = [idx for idx in range(len(class_images))]
            random.shuffle(idxs)

            X_test.extend([class_images[idx] for idx in idxs])
            y_test.extend([class_idx for _ in idxs])

        return X_train, y_train, X_test, y_test



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        X = self.X[idx]
        y = self.y[idx]

        data = np.load(X).astype(np.float32).T

        data, data = sample_fixed_length_data_aligned(data, data, self.sequence_length)

        return data, y

    @property
    def classes(self):
        return 2



def sample_fixed_length_data_aligned(data_a, data_b, sample_length):

    frames_total = data_a.shape[1]

    if frames_total < sample_length:
        data_a = np.pad(data_a, ((0, 0), (0, sample_length - frames_total+1)), mode='constant')
        data_b = np.pad(data_a, ((0, 0), (0, sample_length - frames_total+1)), mode='constant')
        frames_total = data_a.shape[1]

    start = np.random.randint(frames_total - sample_length + 1)
    end = start + sample_length

    return data_a[:, start:end], data_b[:, start:end]


def _load_datasets(data_dir, data_fir_val):
    X_train, y_train, X_test, y_test = DatasetLoader.load_dataset(data_dir, data_fir_val)

    train_data = DatasetLoader(X_train, y_train)
    test_data = DatasetLoader(X_test, y_test)

    logger.debug(f"load_datasets: (train_data, test_data, eval_data) sizes = "
                 f"{len(train_data), len(test_data)}")
    return train_data, test_data,


def create_dataloaders(data_dir,  data_fir_val, batch_size=64):
    train_data, test_data = _load_datasets(data_dir, data_fir_val)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    return trainloader, testloader
