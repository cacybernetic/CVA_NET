import os
import logging
import random
import typing as t

import numpy as np
import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader as PytorchDataLoader

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import Compose

LOGGER = logging.getLogger(__name__)


class HDF5Writer:
    def __init__(self, h5_group):
        self._h5_group = h5_group
        self._count = len(list(self._h5_group.keys()))

    def set_attr(self, name, value):
        """Method to set attribute

        :param name: The name of the attribute.
        :param value: The value of this attribute.

        :type name: `str`
        :type value: `typing.Any`
        :rtype: `None`
        """
        self._h5_group.attrs[name] = value

    def write(self, sample):
        """
        Function of samples writing into dataset

        :param sample: The sample row wille be writen.
        :type sample: `tuple` of `numpy.ndarray`| `numpy.ndarray`
        :rtype: `None`
        """
        if not isinstance(sample, tuple):
            if isinstance(sample, np.ndarray):
                sample = (sample,)
            else:
                raise TypeError(
                    "Tuple of np.ndarray is expected. "
                    "But another type is received"
                )

        new_index = self._count
        index_str = str(new_index)
        sample_group = self._h5_group.create_group(index_str)
        for ind, array in enumerate(sample):
            if not isinstance(array, np.ndarray):
                raise TypeError(
                    "Tuple of sample must contain the ndarray data type."
                )
            sample_group.create_dataset(f"{ind}", data=array)
        self._count += 1

    def clear(self):
        """Function to clear this dataset"""
        if self._count == 0:
            return
        self._h5_group.clear()
        self._count = 0

    def __len__(self):
        return self._count


class HDF5Reader:
    def __init__(self, h5_group, name=None):
        self._h5_group = h5_group
        self._name = name if name is not None else ""
        self._count = len(list(self._h5_group.keys()))
        self._description = {}

    @property
    def class_names(self):
        return self.get_attr('class_names', [])

    def get_attr(self, name, default=None):
        """Method to set attribute

        :param name: The name of the attribute.
        :param default: The default value to return
          when no value found for this attribute.

        :type name: `str`
        :type default: `typing.Any`
        :rtype: `typing.Any`
        """
        attr_keys = self._h5_group.attrs.keys()
        if name not in attr_keys:
            return default
        return self._h5_group.attrs[name]

    def read(self, index):
        """
        Function of samples reading from dataset

        :param index: The index of the sample.
        :type index: `int`
        :rtype: `tuple` of `numpy.ndarray` | `numpy.ndarray`
        """
        if index < 0 or index >= self._count:
            raise IndexError(f"The index {index} is out of range")
        index_str = str(index)
        sample_group = self._h5_group[index_str]
        sample_keys = list(sample_group.keys())
        sample_columns = []

        for key_str in sample_keys:
            column = np.asarray(sample_group[key_str])
            sample_columns.append(column)
        # print(sample_columns)
        # breakpoint()

        return tuple(sample_columns) if len(sample_columns) > 1 \
            else sample_columns[0]

    def describe(self):
        """
        Method to return a description of this dataset
        formatted in string.

        :rtype: `dict` of `typing.Any`
        """
        if self._description:
            return self._description

        # Count number of sample for each label.
        label_counts = {}
        class_names = self.class_names
        for i in range(self._count):
            _, cls_id = self.read(i)
            cls_name = class_names[cls_id]
            if cls_name in label_counts:
                label_counts[cls_name] += 1
            else:
                label_counts[cls_name] = 1

        self._description['label_counts'] = label_counts
        self._description['class_names'] = class_names
        self._description['num_samples'] = self._count
        return self._description

    def summary(self):
        """
        Method to return a summary about this dataset
        formatted in string.

        :rtype: `str`
        """
        if not self:
            return ""
        desc = self.describe()
        num_samples = desc['num_samples']
        # class_names = desc['class_names']
        label_counts = desc['label_counts']

        summary_str = f"Dataset {self._name}:\n"
        summary_str += f"\t* Number of samples: \033[48m{num_samples}\033[0m"
        summary_str += "\n"

        # sorted_cls_names = sorted(
        #     class_names, key=lambda x: len(x), reverse=True)
        # max_length = len(sorted_cls_names[0])
        summary_str += f"\t* Class names list and its number of occurrences:"
        summary_str += "\n"
        for class_name, count in label_counts.items():
            summary_str += f"\t\t\t- {class_name:24s} {count}\n"

        index = random.randint(0, len(self))
        sample = self.read(index)
        summary_str += "\t* Sample:\n"
        if isinstance(sample, tuple):
            feature = sample[0]
            target = sample[1]
            summary_str += "\t\t* Feature:\n" + str(feature) + "\n"
            summary_str += "\t\t* Target:\n" + str(target) + "\n"
            summary_str += "\t\t* Feature shape:" + str(feature.shape) + "\n"
            summary_str += "\t\t* Target shape:" + str(target.shape) + "\n"

        return summary_str

    def __len__(self):
        return self._count

    def __getitem__(self, item):
        sample = self.read(item)
        return sample

    def __str__(self):
        return self.summary()


class HDF5DatasetWriter:
    """
    HDF5 dataset writing implementation
    ===================================

    :type file_path: `str`
    """
    def __init__(self, file_path):
        self._file_path = file_path
        self._h5file = None
        self._metadata = None
        self._datasets = None

    def open(self):
        """Function of HDF5 file opening

        :rtype: `None`
        """
        assert self._file_path is not None, \
            "The file path value is not defined"
        self._h5file = h5py.File(self._file_path, mode='a')

        data_keys = list(self._h5file.keys())
        if 'metadata' not in data_keys:
            self._metadata = self._h5file.create_group('metadata')
        self._metadata = self._h5file['metadata']
        attr_names = list(self._metadata.attrs.keys())
        if 'class_names' not in attr_names:
            self._metadata.attrs['class_names'] = []

        if 'datasets' not in data_keys:
            self._datasets = self._h5file.create_group('datasets')
        self._datasets = self._h5file['datasets']

    def _check_file_opened(self):
        if not isinstance(self._h5file, h5py.File):
            raise TypeError(
                f"The file located at {self._file_path}"
                " is not opened yet. Please open the file"
                " calling open() method."
            )

    def new_dataset(self, name):
        """
        Function of creation of new dataset named the name provided.

        :param name: The name of the new dataset.
        :type name: `str`
        :rtype: `HDF5Writer`
        """
        self._check_file_opened()
        # if name in self._datasets:
        #     raise FileExistsError(
        #         f"The dataset named {name} is already exists")
        group = self._datasets.create_group(name)
        writer = HDF5Writer(group)
        # self._datasets.append(name)
        return writer

    def exists(self, name):
        """Function to check if the dataset named `name` is exists

        :type name: `str`
        :rtype: `bool`
        """
        self._check_file_opened()
        return name in list(self._datasets.keys())

    def get_dataset(self, name):
        """
        Function to return an instance of dataset writer
        for an existing dataset.

        :type name: `str`
        :rtype: `HDF5Writer`
        """
        # if name in self._datasets:
        #     raise FileExistsError(
        #         f"The dataset named {name} is already exists")
        group = self._datasets[name]
        writer = HDF5Writer(group)
        return writer

    def mark_completed(self):
        """
        Method that allows to mark that the writing
        of dataset into h5 file is completed.

        :rtype: `bool`
        """
        metadata_keys = list(self._metadata.attrs.keys())
        if 'completed' in metadata_keys:
            return
        self._metadata.attrs['completed'] = True

    def close(self):
        """Function of HDF5 file closing"""
        self._check_file_opened()
        self._h5file.close()
        # self._datasets.clear()


class HDF5DatasetReader:
    """
    HDF5 dataset writing implementation
    ===================================

    :type file_path: `str`
    """
    def __init__(self, file_path):
        self._file_path = file_path
        self._h5file = None
        self._metadata = None
        self._datasets = None
        self._class_names = None

    @staticmethod
    def completed(file_path):
        """
        Function to verify if the h5 file is completed or not.

        :rtype: `bool`
        """
        if not os.path.isfile(file_path):
            return False
        with h5py.File(file_path, mode='r') as h5_file:
            h5_keys = list(h5_file.keys())
            if 'metadata' not in h5_keys:
                return False
            metadata = h5_file['metadata']
            metadata_keys = list(metadata.attrs.keys())
            if 'completed' not in metadata_keys:
                return False
            is_completed = bool(metadata.attrs['completed'])
            return is_completed

    @property
    def dataset_names(self):
        """List of dataset names

        :rtype: `tuple` of `str`
        """
        self._check_file_opened()
        return list(self._datasets.keys())

    def open(self):
        """Function of HDF5 file opening

        :rtype: `None`
        """
        assert self._file_path is not None, \
            "The file path value is not defined"
        self._h5file = h5py.File(self._file_path, mode='r')
        # self._datasets = tuple(self._h5file.keys())
        data_keys = list(self._h5file.keys())
        if 'datasets' not in data_keys or 'metadata' not in data_keys:
            raise TypeError(f"This HDF5 file is not corrupted.")
        self._metadata = self._h5file['metadata']
        self._datasets = self._h5file['datasets']

    def _check_file_opened(self):
        if not isinstance(self._h5file, h5py.File):
            raise TypeError(
                f"The file located at {self._file_path}"
                " is not opened yet. Please open the file"
                " calling open() method."
            )

    def exists(self, name):
        """Function to check if the dataset named `name` is exists

        :type name: `str`
        :rtype: `bool`
        """
        self._check_file_opened()
        return name in list(self._datasets.keys())

    def get_dataset(self, name):
        """
        Function to return an instance of dataset writer
        for an existing dataset.

        :type name: `str`
        :rtype: `HDF5Reader` | `None`
        """
        if not self.exists(name):
            return None
        group = self._datasets[name]
        writer = HDF5Reader(group, name=name)
        return writer

    def close(self):
        """Function of HDF5 file closing"""
        self._check_file_opened()
        self._h5file.close()

###############################################################################
# DATA TRANSFORMATION
###############################################################################
from PIL import Image, ImageEnhance
from torchvision.transforms import Compose


class CustomRotation:
    """Custom rotation with random angle for license plate characters."""

    def __init__(self, degrees: t.Tuple[float, float] = (-7, 7), fill: int=0):
        self.degrees = degrees
        self.fill = fill

    def __call__(self, img):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        img = TF.rotate(img, angle, fill=self.fill)
        #: fill for missing pixels with fill value between 0-255.
        return img


class AdaptiveContrast:
    """Adaptive contrast enhancement for license plate characters."""

    def __init__(self, factor_range: t.Tuple[float, float] = (0.8, 1.5)):
        self.factor_range = factor_range

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL for enhancement
            img = TF.to_pil_image(img)

        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        enhancer = ImageEnhance.Contrast(img)
        enhanced_img = enhancer.enhance(factor)
        return enhanced_img


class ToNumpy:
    """Conversion into numpy array."""

    def __init__(self) -> None:
        ...

    def __call__(self, img: torch.Tensor):
        img = img.cpu().detach()
        img = img.numpy()
        return img


def get_training_transforms() -> Compose:
    """
    Training transforms pipeline optimized for license plate character
    recognition. Includes data augmentation techniques suitable
    for alphanumeric characters.
    """
    return Compose([
        # Resize to AlexNet input size while maintaining aspect ratio:
        # transforms.Resize(img_size),

        # Random crop to final AlexNet input size (224x224):
        # transforms.RandomCrop(224, padding=16, fill=255),

        # Custom rotation with limited angle to preserve character
        # readability:
        CustomRotation(degrees=(-7, 7)),

        # Random horizontal flip with low probability (characters should
        # remain readable):
        transforms.RandomHorizontalFlip(p=0.1),

        # Color jittering to simulate different lighting conditions:
        transforms.ColorJitter(
            brightness=0.3,    # Simulate different lighting;
            contrast=0.3,      # Vary contrast levels;
            saturation=0.2,    # Slight saturation changes;
            hue=0.05           # Minor hue variations;
        ),

        # Adaptive contrast enhancement:
        AdaptiveContrast(factor_range=(0.9, 1.4)),

        # Random perspective transformation (subtle):
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=255),

        # Random affine transformations:
        transforms.RandomAffine(
            degrees=0,               # No additional rotation (handled above);
            translate=(0.05, 0.05),  # Slight translation;
            scale=(0.95, 1.05),      # Minor scaling;
            shear=(-2, 2),           # Subtle shearing;
            fill=255,
        ),

        # Gaussian blur to simulate camera focus issues:
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2,
        ),

        # Convert to tensor:
        transforms.ToTensor(),

        # Normalize using ImageNet statistics (AlexNet was pre-trained
        # on ImageNet):
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ),

        # Random erasing to improve robustness:
        transforms.RandomErasing(
            p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0,
        ),
        ToNumpy(),
    ])


def get_transforms(
    img_size: t.Tuple[int, int] = (224, 224)
) -> Compose:
    """
    Validation/test transforms pipeline.
    Only includes necessary preprocessing without augmentation.
    """
    return Compose([
        # Resize to slightly larger than target size:
        transforms.Resize(img_size),

        # Center crop to AlexNet input size:
        # transforms.CenterCrop(224),

        # Convert to tensor:
        transforms.ToTensor(),

        # Normalize using ImageNet statistics:
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        ToNumpy(),
    ])


###############################################################################
# DATASET BUILDING
###############################################################################


def build(
    data_dir: str,
    dataset_file: str,
    transform: t.Union[Compose, t.Callable[[Image.Image], np.ndarray]]=None,
    img_size: t.Tuple[int, int]=(224, 224),
    train: bool=True,
) -> HDF5Writer:
    """
    Class method to build a dataset from data samples
    contained on data dir.

    :param data_dir: The path to the data directory.
    :param img_size: The image size will be used to resize all images.
    :param transform: The pipeline of data transformation.
    :returns: An instance of the Dataset.
    """
    paths = []
    classes = []
    class_names = []
    folder_names = os.listdir(data_dir)
    folder_names = sorted(folder_names)
    
    ## Image files loading with its label from data directory.
    iterator = tqdm(folder_names, total=len(folder_names), unit=' label(s)')
    for index, folder_name in enumerate(iterator):
        class_name = folder_name
        class_idx = index

        class_names.append(class_name)
        folder_path = os.path.join(data_dir, folder_name)
        file_names = os.listdir(folder_path)
        file_count = len(file_names)
        iterator.set_description(f"Loading of {file_count} from {folder_name}")
        for fid, file_name in enumerate(file_names):
            is_image_file = (
                file_name.endswith(".png")
                or file_name.endswith(".jpg")
                or file_name.endswith(".jpeg")
                or file_name.endswith(".webp")
            )
            if not is_image_file:
                continue
            file_path = os.path.join(folder_path, file_name)
            #fix_image_rotation(file_path, file_path)
            paths.append(file_path)
            classes.append(class_idx)
            iterator.set_postfix(files=f"{(fid + 1)}/{file_count}")

        iterator.write(f"Class {class_name} of {file_count} is processed.")
    iterator.set_description("Done")
    LOGGER.info("Image files loading from data directory is done.\n")

    ## Image transformation.
    if transform is None:
        transform = get_training_transforms() if train is True \
            else get_transforms()
    ds_writer = HDF5DatasetWriter(dataset_file)
    ds_writer.open()
    dataset_name = 'train' if train is True else 'test'
    h5_dataset = ds_writer.new_dataset(dataset_name)
    iterator = tqdm(
        iterable=zip(paths, classes), total=len(paths), unit=' image(s)'
    )
    for image_file, class_idx in iterator:
        image = Image.open(image_file).convert('RGB')
        image = image.resize(img_size)
        image = transform(image)
        class_idx = np.array(class_idx, dtype=np.int64)
        h5_dataset.write((image, class_idx))
    LOGGER.info("Image transformation and dataset building is done.")

    ## Dataset metadata adding.
    h5_dataset.set_attr('num_channels', image.shape[0])
    h5_dataset.set_attr('image_size', img_size)
    h5_dataset.set_attr('class_names', class_names)
    LOGGER.info("Dataset metadata is added.")

    ## Dataset closing.
    # ds_writer.update_classes(class_names)
    ds_writer.mark_completed()
    ds_writer.close()
    LOGGER.info("Dataset file completed and closed.")
    return h5_dataset


###############################################################################
# DATASET LOADING
###############################################################################

class Dataset(BaseDataset):

    def __init__(
        self,
        dataset_source: HDF5Reader,
        start_index: int=0,
        end_index: int=-1
    ) -> None:
        super().__init__()
        self._dataset_source = dataset_source
        self._start = start_index
        assert end_index >= -1, \
            "The value of the end of index must be between -1, 0, +inf."
        self._end = end_index if end_index != -1 else len(dataset_source)
        self._count = self._end - self._start

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int):
        abs_index = self._start + index
        if abs_index < self._start or abs_index > self._end:
            raise IndexError(
                "The index of the sample is out of range of %d and %d."
                % (0, self._count - 1)
            )
        feature, label_idx = self._dataset_source.read(index)
        feature = torch.tensor(feature, dtype=torch.float32)
        label_idx = torch.tensor(label_idx, dtype=torch.int64)
        return feature, label_idx


def get_dataloader(
    dataset_source: HDF5DatasetReader,
    batch_size: int = 16,
    num_workers: int = 4,
    drop_last: bool=False,
    pin_memory: bool=False,
    val: int=0.2,
) -> t.List[Dataset]:
    # ds_reader = HDF5DatasetReader(dataset_file)
    # ds_reader.open()
    train_dataset_source = dataset_source.get_dataset("train")
    test_dataset_source = dataset_source.get_dataset("test")
    class_names = train_dataset_source.get_attr('class_names')

    ## Create datasets.
    train_dataset = Dataset(train_dataset_source)
    test_dataset = Dataset(test_dataset_source)
    num_val_samples = int(val * len(test_dataset_source))
    val_dataset = Dataset(test_dataset_source, end_index=num_val_samples)

    ## Create data loaders.
    train_loader = PytorchDataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
    )
    val_loader = PytorchDataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = PytorchDataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, class_names

###############################################################################
# MAIN IMPLEMENTATION
###############################################################################

def _get_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['building', 'analysis'])
    parser.add_argument(
        '-d', '--data-dir', type=str, help="The path to directory of data."
    )
    parser.add_argument(
        '--image-size', nargs=2, type=int,
        help=(
            "Allows to define the size which will be taked by all images "
            "contained in the final dataset."
        )
    )
    parser.add_argument(
        '--train', action="store_true",
        help=(
            "Specify if the dataset which we want to build "
            "is the training set to allow the dataset builder "
            "to use the good data transformer."
        )
    )
    parser.add_argument(
        '--dataset', type=str, default='dataset.h5',
        help="The path to HDF5 file where you want to save the dataset."
    )
    return parser.parse_args()


def main() -> None:
    import sys

    # Set up logging:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s \t %(message)s',
        handlers=[
            logging.FileHandler("classification_dataset.log"),
            logging.StreamHandler()
        ]
    )

    args = _get_arguments()

    if args.action == 'building':
        dataset_file = build(
            data_dir=args.data_dir, dataset_file=args.dataset,
            img_size=args.image_size, train=args.train
        )
        LOGGER.info(str(dataset_file))
    
    elif args.action == 'analysis':
        ds_reader = HDF5DatasetReader(args.dataset)
        ds_reader.open()
        train_dataset_source = ds_reader.get_dataset("train")
        test_dataset_source = ds_reader.get_dataset("test")

        LOGGER.info("=" * 80)
        LOGGER.info("TRAIN DATASET:")
        LOGGER.info("=" * 80)
        LOGGER.info(str(train_dataset_source))
        LOGGER.info("=" * 80)
        LOGGER.info("TEST DATASET:")
        LOGGER.info("=" * 80)
        LOGGER.info(str(test_dataset_source))

        ret = get_dataloader(ds_reader)
        train_loader, val_loader, test_loader, class_names = ret
        LOGGER.info("Testing of dataloader (val_loader):")
        for idx, (features, target_ids) in enumerate(val_loader):
            LOGGER.info("features shape: " + str(features.shape))
            LOGGER.info("target_ids shape: " + str(target_ids.shape))
            if idx >= 3:
                break
        LOGGER.info("class_names: " + str(class_names))
        LOGGER.info("class_names: " + str(len(class_names)) + " classes.")

        ds_reader.close()

    sys.exit(0)


if __name__ == '__main__':
    main()
