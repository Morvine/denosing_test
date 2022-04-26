import glob
import logging

import numpy as np
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DatasetLoader(Dataset):

    def __init__(self, clear, noise, cls, length=160, transform=None):
        self.clear = clear
        self.noise = noise
        self.cls = cls
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
        labels_train = [1 for _ in range(len(input_paths_train[0]))]
        labels_train.extend([0 for _ in range(len(input_paths_train[1]))])

        labels_test = [1 for _ in range(len(input_paths_test[0]))]
        labels_test.extend([0 for _ in range(len(input_paths_test[1]))])
        input_paths_train[1].extend(input_paths_train[0])
        input_paths_train[0].extend(input_paths_train[0])

        input_paths_test[1].extend(input_paths_test[0])
        input_paths_test[0].extend(input_paths_test[0])



        return input_paths_train[0], input_paths_train[1], labels_train, input_paths_test[0], input_paths_test[1], labels_test

    def __len__(self):
        return len(self.clear)

    def __getitem__(self, idx):

        noise_path = self.noise[idx]
        clear_path = self.clear[idx]
        noise_data = np.load(noise_path).astype(np.float32).T
        clear_data = np.load(clear_path).astype(np.float32).T

        if noise_data is None or clear_path is clear_data:
            raise ValueError(f"\nbroken image path: {noise_path}")

        noise_data, clear_data = sample_fixed_length_data_aligned(noise_data, clear_data, self.sequence_length)
        cls = self.cls[idx]

        return clear_data, noise_data, cls



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


def _load_datasets(data_dir,data_fir_val):
    X_train, y_train, cls_train, X_test, y_test, cls_test = DatasetLoader.load_dataset(data_dir, data_fir_val)

    train_data = DatasetLoader(X_train, y_train, cls_train)
    test_data = DatasetLoader(X_test, y_test, cls_test)

    logger.debug(f"load_datasets: (train_data, test_data) sizes = "
                 f"{len(train_data), len(test_data)}")
    return train_data, test_data


def create_dataloaders(data_dir,  data_fir_val, batch_size=64):
    train_data, test_data = _load_datasets(data_dir, data_fir_val)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
    return trainloader, testloader
