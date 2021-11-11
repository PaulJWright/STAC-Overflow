import numpy as np
import pandas as pd
import rasterio
import torch

new = 1


def normalize(array, min_val=None, max_val=None):

    if not min_val:
        min_val = np.nanmin(array)

    if not max_val:
        max_val = np.nanmax(array)

    if min_val == max_val:
        min_val, max_val = 0, 1

    array = np.clip(array, min_val, max_val)
    array = (array - min_val) / (max_val - min_val)
    return np.nan_to_num(array)


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
            occurrence_path = group[group.polarization == "vv"][
                "occurrence"
            ].values[0]
            nasadem_path = group[group.polarization == "vv"]["nasadem"].values[
                0
            ]
            seasonality_path = group[group.polarization == "vv"][
                "seasonality"
            ].values[0]
            extent_path = group[group.polarization == "vv"]["extent"].values[0]
            change_path = group[group.polarization == "vv"]["change"].values[0]
            recurrence_path = group[group.polarization == "vv"][
                "recurrence"
            ].values[0]
            transitions_path = group[group.polarization == "vv"][
                "transitions"
            ].values[0]
            paths.append(
                [
                    chip,
                    vv_path,
                    vh_path,
                    occurrence_path,
                    nasadem_path,
                    seasonality_path,
                    extent_path,
                    change_path,
                    recurrence_path,
                    transitions_path,
                ]
            )
        return pd.DataFrame(
            paths,
            columns=[
                "chip_id",
                "vv_path",
                "vh_path",
                "occurrence_path",
                "nasadem_path",
                "seasonality_path",
                "extent_path",
                "change_path",
                "recurrence_path",
                "transitions_path",
            ],
        )

    def __init__(
        self,
        root,
        fsv=None,
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
            data = pd.read_csv(
                f"{self.split_name}chipId_{fsv}_train.csv"  # _weighted_withoutna.csv"
                # f"{self.split_name}chipId_0_train.csv"  # weighted_withoutna.csv"
            )
        elif self.split == "val":
            data = pd.read_csv(f"{self.split_name}chipId_{fsv}_val.csv")
        else:
            print("Not a valid split type")

        # data in, out
        self.data_x = self._get_paths_by_chip(data)
        self.data_y = (
            data[["chip_id", "label_path"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.data_labels = (
            data[["location", "chip_id"]]
            .drop_duplicates(["chip_id"])
            .reset_index(drop=True)
        )

        if self.split == "train":
            # creating weighting labels
            self.weighting_labels = (
                data[
                    [
                        "location",
                        "chip_id",
                        # "weight_0",
                        # "weight_1",
                        # "weight_2",
                        # "weight_3",
                        # "weight_4",
                    ]
                ]
                .drop_duplicates(["chip_id"])
                .reset_index(drop=True)
            )

            # self.weighting_labels["median_weight"] = self.weighting_labels[
            #     ["weight_0", "weight_1", "weight_2", "weight_3", "weight_4"]
            # ].median(axis=1)
            # self.weightings = self.weighting_labels["median_weight"].to_numpy()

        if self.data_y is not None:
            assert self.data_x.shape[0] == self.data_y.shape[0]

    def get_labels(self):
        """
        Method to get the labels for the specified dataset.

        Returns
        -----------
        target_labels : np.ndarray
            Array of the corresponding target labels for the specified
            dataset.
        """

        return self.data_labels["location"].to_numpy()

    def __getitem__(self, idx):
        # Loads a 2-channel image from a chip-level dataframe

        img = self.data_x.loc[idx]

        with rasterio.open(img.vv_path) as vv:
            vv_path = vv.read(1)
        with rasterio.open(img.vh_path) as vh:
            vh_path = vh.read(1)
        # x_arr = np.stack([vv_path, vh_path], axis=-1)
        with rasterio.open(img.nasadem_path) as nasadem:
            nasadem_path = nasadem.read(1)
        with rasterio.open(img.occurrence_path) as occurrence:
            occurrence_path = occurrence.read(1)
        with rasterio.open(img.seasonality_path) as seasonality:
            seasonality_path = seasonality.read(1)
        with rasterio.open(img.extent_path) as extent:
            extent_path = extent.read(1)
        with rasterio.open(img.change_path) as change:
            change_path = change.read(1)
        with rasterio.open(img.recurrence_path) as recurrence:
            recurrence_path = recurrence.read(1)
        with rasterio.open(img.transitions_path) as transitions:
            transitions_path = transitions.read(1)
        x_arr = np.stack(
            [
                vv_path,
                vh_path,
                occurrence_path,
                nasadem_path,
                seasonality_path,
                extent_path,
                change_path,
                recurrence_path,
                transitions_path,
            ],
            axis=-1,
        )

        # Min-max normalization
        # !TODO understand if min/max_norm here is suitable.
        # !TODO missing values should go to zero (as easier to predict)
        mean_norm = -10.8662
        stdv_norm = 5.1265
        x_arr[:, :, 0:1] = (x_arr[:, :, 0:1] - mean_norm) / (stdv_norm)
        mean_norm = -17.6854
        stdv_norm = 5.7039
        x_arr[:, :, 1:2] = (x_arr[:, :, 1:2] - mean_norm) / (stdv_norm)
        #
        # --- occurence
        x_arr[:, :, 2:3] = x_arr[:, :, 2:3] / 255
        #
        # --- nasadem
        # just dividing because this data is useful
        # assuming no flooding > 200m above sealevel
        x_arr[:, :, 3:4] = x_arr[:, :, 3:4] / 750.0
        x_arr[:, :, 3:4] = np.clip(x_arr[:, :, 3:4], -1, 1)
        mean_norm = 0.1977
        stdv_norm = 0.1635
        x_arr[:, :, 3:4] = (x_arr[:, :, 3:4] - mean_norm) / (stdv_norm)
        #
        # --- seasonality
        x_arr[:, :, 4:5][x_arr[:, :, 4:5] == 255] = 0
        mean_norm = 0.6490
        stdv_norm = 2.5721
        x_arr[:, :, 4:5] = (x_arr[:, :, 4:5] - mean_norm) / (stdv_norm)
        #
        # --- extent
        x_arr[:, :, 5:6][x_arr[:, :, 5:6] == 255] = 0
        mean_norm = 0.1194
        stdv_norm = 0.3243
        x_arr[:, :, 5:6] = (x_arr[:, :, 5:6] - mean_norm) / (stdv_norm)
        #
        # --- change
        # x_arr[:,:,6:7][x_arr[:,:,6:7] == 255] = 0
        # mean_norm = 233.1316
        # stdv_norm = 55.7812
        x_arr[:, :, 6:7] = x_arr[:, :, 6:7] / 255
        #
        # --- recurrence
        x_arr[:, :, 7:8][x_arr[:, :, 7:8] == 255] = 0
        mean_norm = 9.2239
        stdv_norm = 26.7654
        x_arr[:, :, 7:8] = (x_arr[:, :, 7:8] - mean_norm) / (stdv_norm)
        #
        # --- transitions\
        x_arr[:, :, 8:9][x_arr[:, :, 8:9] == 255] = 0
        mean_norm = 0.5780
        stdv_norm = 1.9358
        x_arr[:, :, 8:9] = (x_arr[:, :, 8:9] - mean_norm) / (stdv_norm)

        if self.data_y is not None:
            label_path = self.data_y.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1)
                # taking values of 255, and putting to zero (no water)
                # for x_arr (vv, vh), 255 -> 1, which seems like no water?
                min_norm = 0
                max_norm = 1
                y_arr[y_arr == 255] = 0
                y_arr = np.clip(y_arr, min_norm, max_norm)

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
