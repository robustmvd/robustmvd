from .registry import get_dataset


def create_dataset(dataset_name, dataset_type=None, split=None, **kwargs):
    """Create a dataset.

    Args:
        dataset_name (str): The name of the dataset to create. Can optionally contain the dataset_type and split in the
            format base_dataset_name.split.dataset_type.
        dataset_type (str): The type of the dataset to create. Can optionally be provided within the dataset_name.
        split (str): The split of the dataset to create. Can optionally be provided within the dataset_name.

    Keyword Args:
        **kwargs: Arguments for the dataset.
    """
    dataset_cls = get_dataset(dataset_name=dataset_name, dataset_type=dataset_type, split=split)
    dataset = dataset_cls(**kwargs)
    return dataset


# TODO: create compounddataset; createupdateddataset; create dataset from cfg file; create dataloader
