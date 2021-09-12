import numpy as np
import pandas as pd
import rasterio
import torch


class FloodDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def _get_paths_by_chip(self, image_level_df):
        """
        Returns a chip-level dataframe with pivoted columns
        for vv_path and vh_path.

        Args:
            image_level_df (pd.DataFrame): image-level dataframe

        Returns:
            chip_level_df (pd.DataFrame): chip-level dataframe
        """
        paths = []
        for chip, group in image_level_df.groupby("chip_id"):
            vv_path = group[group.polarization == "vv"]["feature_path"].values[
                0
            ]
            vh_path = group[group.polarization == "vh"]["feature_path"].values[
                0
            ]
            paths.append([chip, vv_path, vh_path])
        return pd.DataFrame(paths, columns=["chip_id", "vv_path", "vh_path"])

    def __init__(
        self,
        root,
        split="train",
        data_augmentation=True,
        split_name="",
        transforms=None,
    ):
        # !TODO if training/testing load appropriate data,
        # and set the x_paths, and y_paths themselves.
        self.root = root
        self.split = split
        self.split_name = split_name
        self.data_augmentation = data_augmentation
        # self.data = x_paths
        # self.label = y_paths
        self.transforms = transforms

        self.fns = []
        if self.split == "train":
            data = pd.read_csv(f"{self.split_name}chipId_train.csv")
        elif self.split == "val":
            data = pd.read_csv(f"{self.split_name}chipId_val.csv")
        else:
            print("Not a valid split type")

        # data in, out
        self.data_x = self._get_paths_by_chip(data)
        self.data_y = (
            data[["chip_id", "label_path"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if self.data_y is not None:
            assert self.data_x.shape[0] == self.data_y.shape[0]

    def __getitem__(self, idx):
        # Loads a 2-channel image from a chip-level dataframe
        img = self.data_x.loc[idx]
        with rasterio.open(img.vv_path) as vv:
            vv_path = vv.read(1)
        with rasterio.open(img.vh_path) as vh:
            vh_path = vh.read(1)
        x_arr = np.stack([vv_path, vh_path], axis=-1)

        # Min-max normalization
        # !TODO understand if min/max_norm here is suitable.
        min_norm = -77
        max_norm = 26
        x_arr = np.clip(x_arr, min_norm, max_norm)
        x_arr = (x_arr - min_norm) / (max_norm - min_norm)

        if self.data_y is not None:
            label_path = self.data_y.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1)

            # Apply same data augmentations to sample and label
            if self.transforms:
                transformed = self.transforms(image=x_arr, mask=y_arr)
                x_arr = transformed["image"]
                y_arr = transformed["mask"]

            x_arr = np.transpose(x_arr, [2, 0, 1])

            sample = {"chip_id": img.chip_id, "chip": x_arr, "label": y_arr}
        else:  # No labels - test set only
            if self.transforms:
                x_arr = self.transforms(image=x_arr)["image"]

            x_arr = np.transpose(x_arr, [2, 0, 1])
            sample = {"chip_id": img.chip_id, "chip": x_arr}

        return sample

    def __len__(self):
        return len(self.data_x)
