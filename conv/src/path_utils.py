from pathlib import Path


def get_conv_data_dir(dataset):
    dataset_dir = Path(__file__).resolve().parents[1] / "data" / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            f"Expected data under conv/data/{dataset}."
        )
    return dataset_dir


def get_root_data_dir(dataset):
    dataset_dir = Path(__file__).resolve().parents[2] / "data" / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            f"Expected data under data/{dataset} at the repository root."
        )
    return dataset_dir


def require_data_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return path
