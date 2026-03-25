import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T


class MRISliceDataset(Dataset):
    """
    PyTorch Dataset for knee MRI slices stored as JPEG files.

    Images are identified by their DICOM SOPInstanceUID, loaded as grayscale,
    and replicated across three channels to match the input expected by
    ImageNet-pretrained models.

    Args:
        image_folder: Directory containing JPEG files named as {SOPInstanceUID}.jpeg.
        dataframe: Pandas DataFrame with at least the columns SOPInstanceUID
                   and the column specified by label_col.
        label_col: Name of the column holding the binary class label (0 or 1).
        transform: Optional torchvision transform applied to the PIL image.
    """

    def __init__(self, image_folder: str, dataframe, label_col: str = "labelTA", transform=None) -> None:
        self.image_folder = image_folder
        self.data = dataframe.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sop_uid = self.data.at[idx, "SOPInstanceUID"]
        image_path = os.path.join(self.image_folder, f"{sop_uid}.jpeg")
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.data.at[idx, self.label_col]
        return np.repeat(image, 3, axis=0), label, sop_uid


def compute_normalization_stats(
    image_folder: str,
    dataframe,
    label_col: str = "labelTA",
    n_samples: int = 500,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[float, float]:
    """
    Estimate pixel mean and standard deviation from a random subset of images.

    Args:
        image_folder: Directory containing JPEG files.
        dataframe: Metadata DataFrame used to build the dataset.
        label_col: Label column name passed to MRISliceDataset.
        n_samples: Number of images to sample.
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader worker processes.

    Returns:
        (mean, std) as plain Python floats.
    """
    base_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    dataset = MRISliceDataset(image_folder, dataframe, label_col=label_col, transform=base_transform)

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_pixels, pixel_sum, pixel_sq_sum = 0, 0.0, 0.0
    for images, _, _ in loader:
        b, _, h, w = images.shape
        n_pixels += b * h * w
        pixel_sum += images[:, 0].sum().item()
        pixel_sq_sum += (images[:, 0] ** 2).sum().item()

    mean = pixel_sum / n_pixels
    std = ((pixel_sq_sum / n_pixels) - mean ** 2) ** 0.5
    return mean, std


def build_transforms(mean: float, std: float, augment: bool = False) -> T.Compose:
    """
    Build the image transform pipeline.

    Args:
        mean: Pixel mean for normalization.
        std: Pixel standard deviation for normalization.
        augment: If True, applies training-time augmentations (flip, rotation,
                 color jitter). Set to False for validation and inference.

    Returns:
        A torchvision Compose transform.
    """
    ops = [T.Resize((224, 224)), T.ToTensor()]
    if augment:
        ops += [
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomHorizontalFlip(),
            T.RandomRotation(30),
        ]
    ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)
