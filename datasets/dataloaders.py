import numpy as np
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
from datasets.datasets import SegDataset

def split_ids(len_ids,seed):
    """
    The function `split_ids` takes the length of IDs and a seed value, then splits the IDs into
    training, validation, and test sets based on specified percentages and returns the indices for each
    set.
    
    :param len_ids: The `len_ids` parameter represents the total number of IDs or samples that you want
    to split into training, validation, and testing sets. This function is designed to split these IDs
    into three sets based on the specified percentages for training, validation, and testing sizes
    :param seed: The `seed` parameter in the `split_ids` function is used to set the random seed for
    reproducibility. By setting a specific seed value, you can ensure that the random splitting of data
    into training, validation, and test sets remains consistent across different runs of the function.
    This is particularly
    :return: The function `split_ids` returns three sets of indices: `train_indices`, `test_indices`,
    and `val_indices`. These indices are used to split a given number of IDs into training, testing, and
    validation sets based on the specified sizes and random seed.
    """

    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=seed,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=seed
    )

    return train_indices, test_indices, val_indices

def get_dataloaders(input_paths, target_paths, batch_size, num_workers,input_dims=(352, 352),seed = 42):
    """
    The function `get_dataloaders` prepares data loaders for training a model with specified input and
    target paths, batch size, number of workers, input dimensions, and seed, along with data
    transformations.
    
    :param input_paths: The `input_paths` parameter is typically a list of file paths where your input
    data is stored. This could be the paths to your training images, for example. You would pass this
    list to your data loader to load the input data during training
    :param target_paths: The `target_paths` parameter typically refers to the paths where the target
    data (labels) are stored. In the context of your function `get_dataloaders`, it is likely the file
    paths where the target data corresponding to the input data is located. This target data is usually
    the ground truth
    :param batch_size: The `batch_size` parameter specifies the number of samples in each batch of data
    that will be processed during training. It is a hyperparameter that can affect the training process
    and model performance. A larger batch size can lead to faster training times but may require more
    memory, while a smaller batch size may
    :param num_workers: The `num_workers` parameter in the `get_dataloaders` function specifies the
    number of subprocesses to use for data loading. It controls the number of parallel processes that
    will be used to load the data. Increasing the `num_workers` can speed up data loading as the data
    loading process
    :param input_dims: The `input_dims` parameter specifies the dimensions to which the input images
    will be resized. In this case, the input images will be resized to a height of 352 pixels and a
    width of 352 pixels
    :param seed: The `seed` parameter in the `get_dataloaders` function is used to set the seed for
    random number generation. Setting a seed ensures that the random operations in your code are
    reproducible. By using the same seed value, you can get the same random numbers each time you run
    your, defaults to 42 (optional)
    """
    
    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(input_dims, antialias=True),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(input_dims, antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(input_dims), transforms.Grayscale()]
    )

    train_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4train,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=True,
    )

    test_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    val_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    train_indices, test_indices, val_indices = split_ids(len(input_paths),seed=seed)

    train_dataset = data.Subset(train_dataset, train_indices)
    val_dataset = data.Subset(val_dataset, val_indices)


    # test_kvasir = [[i for i in test_indices if input_paths[i].endswith("jpg")][3]]
    # test_cvc = [[i for i in test_indices if input_paths[i].endswith("tif")][3]]
    # test_dataset_kvasir = data.Subset(test_dataset, test_kvasir)
    # test_dataset_cvc = data.Subset(test_dataset, test_cvc)
    # print(input_paths[test_cvc[0]])
    test_kvasir = [i for i in test_indices if input_paths[i].endswith("jpg")]
    test_cvc = [i for i in test_indices if input_paths[i].endswith("tif")]
    test_dataset_kvasir = data.Subset(test_dataset, test_kvasir)
    test_dataset_cvc = data.Subset(test_dataset, test_cvc)


    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_kvasir_dataloader = data.DataLoader(
        dataset=test_dataset_kvasir,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    test_cvc_dataloader = data.DataLoader(
        dataset=test_dataset_cvc,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )


    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader,test_kvasir_dataloader, test_cvc_dataloader, val_dataloader
