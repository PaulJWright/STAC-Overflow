from __future__ import print_function

import copy
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

import albumentations
import numpy as np
import scipy
import scipy.ndimage as scind
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import yaml
from dataset import FloodDataset
from torchsampler import ImbalancedDatasetSampler

EXPERIMENT = 0


def make_the_weight_image(data, opt):
    y = copy.deepcopy(data["label"].int())
    weights = np.zeros_like(y.float())

    for id2 in range(0, y.shape[0]):
        temp = y[id2].detach().numpy().astype(int)
        a = np.abs(scipy.ndimage.sobel(temp, axis=-1))
        b = np.abs(scipy.ndimage.sobel(temp, axis=0))
        temp = temp.astype(float)
        temp[temp == 0.0] = 1 / np.log(
            opt["segmentation_weights_heuristic"] + 1.0 - opt["water_perc"]
        )
        temp[temp == 1.0] = 1 / np.log(
            opt["segmentation_weights_heuristic"] + opt["water_perc"]
        )

        temp[np.where((a + b) > 0)] = float(opt["edge_val"])

        weights[id2, :, :] = temp

        assert np.unique(temp).shape[0] <= 3

    return weights


def unet_weight_map(y, wc=None, w0=10, sigma=5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    labels = y
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = scind.distance_transform_edt(
                labels != label_id
            )

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


class XEDiceLoss(torch.nn.Module):
    """
    Computes (0.5 * CrossEntropyLoss) + (0.5 * DiceLoss).
    """

    def __init__(self, wts):
        super().__init__()
        self.wts = wts
        if self.wts is None:
            self.xe = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            self.xe = torch.nn.CrossEntropyLoss(
                reduction="none", weight=self.wts
            )

    def forward(self, pred, true, xe):
        valid_pixel_mask = true.ne(255)  # valid pixel mask

        # Cross-entropy loss
        temp_true = torch.where(
            (true == 255), 0, true
        )  # cast 255 to 0 temporarily
        xe_loss = self.xe(pred, temp_true)
        xe_loss = xe_loss.masked_select(valid_pixel_mask).mean()

        # Dice loss
        pred = torch.softmax(pred, dim=1)[:, 1]
        pred = pred.masked_select(valid_pixel_mask)
        true = true.masked_select(valid_pixel_mask)
        dice_loss = 1 - (2.0 * torch.sum(pred * true)) / (
            torch.sum(pred + true) + 1e-7
        )

        # print(xe_loss, dice_loss)
        # print(xe)

        return (xe * xe_loss) + ((1 - xe) * dice_loss)


def intersection_and_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum(), union.sum()


# from collections import OrderedDict

# Start time
start_time = datetime.now()
print(">>> Started at: `{0}`\n\n".format(start_time))

if EXPERIMENT == 1:
    with open("experiment_5.yml") as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)
else:
    # --- Load parameters for training
    with open("best_model_9th.yml") as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

print(opt)

# --- Ensure torch is using deterministic algorithms for reproducibility
torch.use_deterministic_algorithms(opt["torch_deterministic"])
# --- Set buffer size to force deterministic results from cuDNN
if opt["torch_deterministic"]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("https://github.com/pytorch/pytorch/issues/46024")


if EXPERIMENT == 1:
    from datetime import datetime

    opt["outf"] = opt["outf"] + str(
        start_time.strftime("%d_%m_%Y_%H-%M-%S") + "/"
    )
    opt["date"] = start_time.strftime("%d_%m_%Y_%H-%M-%S")
    # !TODO should change to not be hardcoded
    # 2 choices
    opt["augmentation"] = opt["augmentation"][random.randint(0, 1)]
    opt["balance_dataset_option"] = opt["balance_dataset_option"][
        random.randint(0, 1)
    ]
    # opt["balance_dataset_type"]   = opt["balance_dataset_type"] [random.randint(0, 1)]
    opt["optimizer"] = opt["optimizer"][random.randint(0, 2)]
    opt["stop_patience_fraction"] = opt["stop_patience_fraction"][
        random.randint(0, 1)
    ]
    # opt["sgd_momentum"]           = opt["sgd_momentum"][random.randint(0, 2)]
    opt["sgd_lr"] = opt["sgd_lr"][random.randint(0, 4)]
    opt["edge_val"] = opt["edge_val"][random.randint(0, 4)]
    opt["weights"] = opt["weights"][random.randint(0, 3)]

    opt["segmentation_weights_heuristic"] = opt[
        "segmentation_weights_heuristic"
    ][random.randint(0, 2)]
    # opt["crop_size_square"] = opt["crop_size_square"][random.randint(0, 1)]
    opt["model"] = opt["model"][random.randint(0, 2)]
    opt["w0"] = random.uniform(opt["w0"][0], opt["w0"][1])
    opt["batchSize"] = opt["batchSize"][random.randint(0, 1)]
    opt["PretrainedUNet_backbone"] = opt["PretrainedUNet_backbone"][
        random.randint(0, 4)
    ]
    opt["PretrainedUNet_weights"] = opt["PretrainedUNet_weights"][
        random.randint(0, 3)
    ]

    # opt["data_version"] = opt["data_version"][random.randint(0, 1)]
    opt["weights_sigma"] = opt["weights_sigma"][random.randint(0, 2)]

    # opt["gamma"] = opt["gamma"][random.randint(0, 3)]
    opt["gamma"] = random.uniform(opt["gamma"][0], opt["gamma"][1])
    opt["water_perc_by_batch"] = opt["water_perc_by_batch"][
        random.randint(0, 1)
    ]

    opt["patience"] = opt["patience"][random.randint(0, 3)]
    opt["dropout_val"] = opt["dropout_val"][random.randint(0, 5)]

    if opt["dropout_val"] == 0.0:
        opt["model"] = "PretrainedUNet"

    # opt["xentropy"] = opt["xentropy"][random.randint(0, 6)]
    opt["xentropy"] = random.uniform(opt["xentropy"][0], opt["xentropy"][1])

    if opt["weights"] == "mask":
        opt["xentropy"] = 1.0

    opt["five_sets_value"] = opt["five_sets_value"][random.randint(0, 4)]
    # opt["lr"] = opt["lr"][random.randint(0, 4)]
    opt["lr"] = random.uniform(opt["lr"][0], opt["lr"][1])
    if opt["optimizer"] == "sgd":
        print(opt["sgd_lr"], opt["lr"])
        opt["lr"] = opt["sgd_lr"]
        print(opt["sgd_lr"], opt["lr"])

    if opt["model"] == "FPN":
        opt["torch_deterministic"] = False
        torch.use_deterministic_algorithms(opt["torch_deterministic"])

# --- Set random seed
if opt["manualSeed"] == 0:
    # Chosen randomly if not set in parameter file
    opt["manualSeed"] = random.randint(0, 10000)

print("Random Seed: ", opt["manualSeed"])
random.seed(opt["manualSeed"])
np.random.seed(opt["manualSeed"])
torch.manual_seed(opt["manualSeed"])

print("\n \n \n >>>> starting")
for keys, values in opt.items():
    print(f"{keys}: {values}")
print("\n \n \n")

# --- Check for GPU, if there isn't one, run on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Make output directory if it doesn't already exist
os.makedirs(opt["outf"])
# --- Create logfile
print(os.path.join(opt["outf"], opt["logfile"]))
log_fout = open(os.path.join(opt["outf"], opt["logfile"]), "a")
# - Starting time
log_fout.write(">>> Started at: `{0}`\n".format(start_time))
# --- Write parameters to logfile
log_fout.write(str(opt) + "\n")

# --- Write git hash to logfile
label = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
log_fout.write(f"Git repo: {label}\n")

# --- Copy parameter file to output directory for posterity
if EXPERIMENT == 1:
    pass
else:
    shutil.copyfile(
        "training_parameters.yml",
        os.path.join(opt["outf"], "training_parameters.yml"),
    )

# --- Set dataset type depending on problem
if opt["dataset_type"] == "FloodDataset":

    # https://github.com/albumentations-team/albumentations#pixel-level-transforms

    transformations = [
        albumentations.RandomCrop(height=512, width=512),
        albumentations.RandomRotate90(),
        albumentations.HorizontalFlip(),
        albumentations.VerticalFlip(),
        # albumentations.HorizontalFlip(),
        # albumentations.VerticalFlip(),
        # albumentations.Rotate(),
        # albumentations.Transpose(),
        # albumentations.ShiftScaleRotate(),
        # albumentations.Affine(),
        # albumentations.Perspective(),
        # albumentations.Downscale(),
    ]

    # Converting integer opt["albumnttns_value"] into a binary of the same length
    # as the transformations array and then indexes of the array.
    #
    # e.g. 6 -> 0110 -> [1,2] -> [  albumentations.RandomRotate90(),
    #                               albumentations.HorizontalFlip(),
    #                            ]
    #
    transformation = bin(opt["albumnttns_value"])[2:].zfill(
        len(transformations)
    )
    # ensure that the length of the augmentation array and the binary are the same
    assert len(transformation) == len(transformations)
    transformation_yes_no = [
        int(i)
        for i in range(0, len(transformation))
        if transformation[i] == "1"
    ]
    print("int: ", opt["albumnttns_value"], "| binary:", transformation)

    transformations = [transformations[i] for i in transformation_yes_no]
    training_transformations = albumentations.Compose(transformations)
    #
    # print
    print(training_transformations)

    # --- Write augmentation parameters to logfile
    log_fout.write(">>> Augmentations:")
    log_fout.write(str(training_transformations) + "\n")

    # --- Load modelnet training set
    dataset = FloodDataset(
        fsv=opt["five_sets_value"],
        root=opt["dataset"],
        split="train",
        data_augmentation=opt["augmentation"],
        split_name=opt["split_name"],
        transforms=training_transformations,
    )

    # --- Load modelnet validation set
    # --- Data_augmentation will always be off for validation
    val_dataset = FloodDataset(
        fsv=opt["five_sets_value"],
        root=opt["dataset"],
        split="val",
        data_augmentation=False,
        split_name=opt["split_name"],
    )
else:
    sys.exit("This dataset is not implemented:", opt["dataset_type"])


# --- Load up training data

if opt["balance_dataset_option"] is True:
    if opt["balance_dataset_type"] == "WeightedRandomSampler":
        print("weightedrandomsampler")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=1.0 - dataset.weightings,
            num_samples=len(dataset.weightings),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = ImbalancedDatasetSampler(dataset)
        shuffle = False
else:
    sampler = None
    shuffle = True

print("sampler", sampler)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt["batchSize"],
    sampler=sampler,
    shuffle=shuffle,
    num_workers=int(opt["workers"]),
)

# --- Load up validation data
valdataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt["workers"]),
)

# --- Store the number of classes for classification
print(len(dataset), len(val_dataset))
num_classes = 2  # opt["classes"]
log_fout.write("classes\t" + str(num_classes) + "\n")
# print(dataset.cat)
# log_fout.write(str(dataset.cat) + "\n")

# --- If loading a pretrained model
if opt["model"] == "PretrainedUNet":

    import segmentation_models_pytorch as smp

    print(
        f"training {opt['model']} from {opt['PretrainedUNet_weights']}"
        + f" with {num_classes} classes"
    )

    classifier = smp.Unet(
        # https://github.com/qubvel/segmentation_models.pytorch/ \
        # blob/master/segmentation_models_pytorch/unet/model.py
        encoder_name=opt["PretrainedUNet_backbone"],  # resnet34
        # encoder_depth=5, # default
        encoder_weights=opt["PretrainedUNet_weights"],  # imagenet
        # decoder_use_batchnorm=True,  # default
        # decoder_channels=(256, 128, 64, 32, 16), # default
        # decoder_attention_type=None,
        in_channels=9,  # in_chnnl
        classes=2,
        # activation: Optional[Union[str, callable]] = None,
        # aux_params: Optional[dict] = None,
        # https://smp.readthedocs.io/en/latest/models.html#unet
    )
elif opt["model"] == "PretrainedUNet_dropout":

    import segmentation_models_pytorch as smp

    print(
        f"training {opt['model']} from {opt['PretrainedUNet_weights']}"
        + f" with {num_classes} classes"
    )

    classifier = smp.Unet(
        # https://github.com/qubvel/segmentation_models.pytorch/ \
        # blob/master/segmentation_models_pytorch/unet/model.py
        encoder_name=opt["PretrainedUNet_backbone"],  # resnet34
        # encoder_depth=5, # default
        encoder_weights=opt["PretrainedUNet_weights"],  # imagenet
        # decoder_use_batchnorm=True,  # default
        # decoder_channels=(256, 128, 64, 32, 16), # default
        # decoder_attention_type=None,
        in_channels=9,  # in_chnnl
        classes=2,
        # activation: Optional[Union[str, callable]] = None,
        # aux_params: Optional[dict] = None,
        # https://smp.readthedocs.io/en/latest/models.html#unet
        aux_params=dict(
            # pooling="avg",  # one of 'avg', 'max'; defauly is "avg"
            dropout=opt["dropout_val"],  # dropout ratio, default is None
            # activation=None,  # activation function, default is None
            classes=int(2),  # define number of output labels
        ),
    )
elif opt["model"] == "UNetPP":
    import segmentation_models_pytorch as smp

    print(
        f"training {opt['model']} from {opt['PretrainedUNet_weights']}"
        + f" with {num_classes} classes"
    )

    classifier = smp.UnetPlusPlus(
        encoder_name=opt["PretrainedUNet_backbone"],  # resnet34
        encoder_weights=opt["PretrainedUNet_weights"],  # imagenet
        in_channels=9,  # in_chnnl
        classes=2,
        aux_params=dict(
            # pooling="avg",  # one of 'avg', 'max'; defauly is "avg"
            dropout=opt["dropout_val"],  # dropout ratio, default is None
            # activation=None,  # activation function, default is None
            classes=int(2),  # define number of output labels
        ),
    )
elif opt["model"] == "FPN":
    import segmentation_models_pytorch as smp

    print(
        f"training {opt['model']} from {opt['PretrainedUNet_weights']}"
        + f" with {num_classes} classes"
    )

    classifier = smp.FPN(
        encoder_name=opt["PretrainedUNet_backbone"],  # resnet34
        encoder_weights=opt["PretrainedUNet_weights"],  # imagenet
        decoder_dropout=opt["dropout_val"],
        in_channels=9,  # in_chnnl
        classes=2,
    )
else:
    sys.exit("This model is not implemented:", opt["model"])


# --- Set optimizer. Right now only adam is set up to go
if opt["optimizer"] == "adam":
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=opt["lr"],
        betas=(opt["beta1"], opt["beta2"]),
    )
elif opt["optimizer"] == "sgd":
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=opt["lr"],
        momentum=opt["sgd_momentum"],
    )
else:
    sys.exit("This optimizer is not implemented:", opt["optimizer"])

# Used in original code.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=opt["gamma"],
    patience=opt["patience"],
    verbose=True,
)

# --- Load model onto appropriate device
classifier = classifier.to(device)

# --- Print header to logfile before beginning training
log_fout.write("****** TRAINING ******\n")
loss_out = open(os.path.join(opt["outf"], "losses.txt"), "a")
loss_out.write("Epoch, Mean_train_loss, Train IoU, Mean_val_loss, Val IoU\n")
loss_out.close()

best_validation_IoU = 0.0
best_validation_epoch = 0
stop_code = 0

# --- Training the model
for epoch in range(opt["nepoch"]):

    trn_intxn = 0
    trn_union = 0
    trn_IoU = 0
    #
    # --- Training of the model
    # - Initializing arrays to store batch loss/acc for each epoch, to
    # - average later.
    n_train_elem = len(dataloader)
    train_loss = [[] for x in range(n_train_elem)]
    train_acc = [[] for x in range(n_train_elem)]
    train_num = [[] for x in range(n_train_elem)]
    train_iou = [[] for x in range(n_train_elem)]
    #
    # - Looping over `dataloader` (Training)
    for ii, data_ii in enumerate(dataloader, 0):
        # Extracting the data points and target (classes)
        inputt = data_ii["chip"]
        target = data_ii["label"].long()

        if opt["water_perc_by_batch"] == "1":
            opt["water_perc"] = target[target == 1].sum() / (
                target.shape[0] * target.shape[1] * target.shape[2]
            )
            print(f"batch water perc {opt['water_perc']}")

        # Putting the data into a GPU, if applicable
        inputt, target = inputt.to(device), target.to(device)
        # Initializing the optimizer and making the gradient zero
        optimizer.zero_grad()
        # Sets the module in training mode.
        classifier = classifier.train()
        # Computing prediction
        pred = classifier(inputt)

        # !TODO understand why adding dictionary to smp.Unet needs this:
        if opt["model"] in {"PretrainedUNet_dropout", "UNetPP"}:
            pred = pred[0]

        # !TODO understand if this needs to be done
        pred_trn = torch.softmax(pred, dim=1)[:, 1]
        pred_trn = (pred_trn > 0.5) * 1

        intersection, union = intersection_and_union(pred_trn, target)
        trn_intxn += intersection
        trn_union += union
        train_iou[ii] = intersection / union
        # Computing loss

        if opt["weights"] == "division":
            weights = torch.tensor(
                [
                    1
                    / np.log(
                        opt["segmentation_weights_heuristic"]
                        + 1.0
                        - opt["water_perc"]
                    ),
                    1
                    / np.log(
                        opt["segmentation_weights_heuristic"]
                        + opt["water_perc"]
                    ),
                ]
            ).type(torch.FloatTensor)
            # print(weights)
            weights = weights.to(device)
            torch.use_deterministic_algorithms(False)
            criterion = XEDiceLoss(weights)
            loss = criterion(pred, target, opt["xentropy"])
            torch.use_deterministic_algorithms(opt["torch_deterministic"])
        elif opt["weights"] == "mask":
            # print('using mask')
            torch.use_deterministic_algorithms(False)
            weights = torch.tensor(
                make_the_weight_image(data_ii, opt), requires_grad=True
            ).to(device)

            def lossWithWeightmap(logit_output, target, weight_map, opt):
                # !TODO check if this is correct
                # https://discuss.pytorch.org/t/weighted-pixelwise-nllloss2d/7766
                # this gets the nn.CrossEntropyLoss
                logSoftmaxOutput = F.log_softmax(logit_output, dim=1)
                # print(logSoftmaxOutput.shape)
                logSoftmaxOutput = logSoftmaxOutput.gather(
                    1,
                    target.view(
                        int(opt["batchSize"]),
                        1,
                        int(opt["crop_size_square"]),
                        int(opt["crop_size_square"]),
                    ),
                )
                # print(logSoftmaxOutput.shape)

                weightedOutput = (logSoftmaxOutput * weight_map).view(
                    int(opt["batchSize"]), -1
                )
                weightedLoss = weightedOutput.sum(1) / weight_map.view(
                    int(opt["batchSize"]), -1
                ).sum(1)
                return -1.0 * weightedLoss.mean()

            # print(pred_trn.shape, target.shape, weights.shape)
            loss = lossWithWeightmap(pred, target, weights, opt)
        elif opt["weights"] == "mask_unet":
            # print('using mask')
            torch.use_deterministic_algorithms(False)

            wc = {
                0: 1
                / np.log(
                    opt["segmentation_weights_heuristic"]
                    + 1.0
                    - opt["water_perc"]
                ),  # background
                1: 1
                / np.log(
                    opt["segmentation_weights_heuristic"] + opt["water_perc"]
                ),  # objects
            }

            weights = torch.tensor(
                np.zeros_like(data_ii["label"].float()), requires_grad=True
            ).to(device)

            for id2 in range(0, weights.shape[0]):
                weights[id2, :, :] = torch.tensor(
                    unet_weight_map(
                        data_ii["label"].float()[id2, :, :],
                        wc,
                        w0=opt["w0"],
                        sigma=opt["weights_sigma"],
                    )
                )

            def lossWithWeightmap(logit_output, target, weight_map, opt):
                # !TODO check if this is correct
                # https://discuss.pytorch.org/t/weighted-pixelwise-nllloss2d/7766
                # this gets the nn.CrossEntropyLoss
                logSoftmaxOutput = F.log_softmax(logit_output, dim=1)
                # print(logSoftmaxOutput.shape)
                logSoftmaxOutput = logSoftmaxOutput.gather(
                    1,
                    target.view(
                        int(opt["batchSize"]),
                        1,
                        int(opt["crop_size_square"]),
                        int(opt["crop_size_square"]),
                    ),
                )
                # print(logSoftmaxOutput.shape)

                weightedOutput = (logSoftmaxOutput * weight_map).view(
                    int(opt["batchSize"]), -1
                )
                weightedLoss = weightedOutput.sum(1) / weight_map.view(
                    int(opt["batchSize"]), -1
                ).sum(1)
                return -1.0 * weightedLoss.mean()

            # print(pred_trn.shape, target.shape, weights.shape)
            loss = lossWithWeightmap(pred, target, weights, opt)
        else:
            torch.use_deterministic_algorithms(False)
            criterion = XEDiceLoss(None)
            loss = criterion(pred, target, opt["xentropy"])
            torch.use_deterministic_algorithms(opt["torch_deterministic"])

        # Regularization if feature transform is being used
        loss.backward()
        if opt["weights"] == "mask":
            torch.use_deterministic_algorithms(opt["torch_deterministic"])
        if opt["weights"] == "mask_unet":
            torch.use_deterministic_algorithms(opt["torch_deterministic"])
        optimizer.step()
        # pred_choice = pred.detach().max(1)[1]
        # correct = pred_choice.eq(target.detach()).sum()
        # Computing statistics
        train_loss_ii = loss.item()
        # train_acc_ii = correct.item()
        #
        # Message to be printed
        msg = "[{0:d}: {1:d}/{2:d}] Train loss: {3:.6f}"  # Accuracy: {4:.6f}"
        msg = msg.format(
            epoch,
            ii,
            n_train_elem,
            train_loss_ii,
        )
        # Writing to the screen
        print(msg)
        # Writing to log file
        log_fout.write(msg + "\n")
        # Appending data to arrays
        train_loss[ii] = train_loss_ii
        # train_acc[ii] = train_acc_ii
        # train_num[ii] = float(len(pred_choice.eq(target.detach()).cpu()))
    trn_IoU = trn_intxn / trn_union
    # --- Validation of the model
    # - Initializing arrays to store batch loss/acc for each epoch, to
    # - average later.
    #  Number of elements in the validation dataset
    n_validation_samples = len(valdataloader)
    val_loss = [[] for x in range(n_validation_samples)]
    validation_iou = [[] for x in range(n_validation_samples)]
    # val_num = [[] for x in range(n_validation_samples)]
    # Putting model into ``evaluation`` mode
    classifier = classifier.eval()

    val_intxn = 0
    val_union = 0
    val_iou = 0
    # Looping over the validation dataset
    for jj, val_jj in enumerate(valdataloader, 0):
        # Extracting points and targets (classes)
        inputt = val_jj["chip"]
        target = val_jj["label"].long()
        # Putting the data into a GPU, if applicable
        inputt, target = inputt.to(device), target.to(device)
        # computing prediction
        pred = classifier(inputt)

        # !TODO understand why adding dictionary to smp.Unet needs this:
        if opt["model"] in {"PretrainedUNet_dropout", "UNetPP"}:
            pred = pred[0]

        # !TODO understand if this needs to be done
        pred_val = torch.softmax(pred, dim=1)[:, 1]
        pred_val = (pred_val > 0.5) * 1
        # Computing loss value
        # https://github.com/pytorch/pytorch/issues/46024
        torch.use_deterministic_algorithms(False)
        criterion = XEDiceLoss(None)
        loss = criterion(pred, target, opt["xentropy"])
        torch.use_deterministic_algorithms(opt["torch_deterministic"])
        # Computing statistics
        validation_loss_jj = loss.item()
        intersection, union = intersection_and_union(pred_val, target)
        validation_iou[jj] = intersection / union
        val_intxn += intersection
        val_union += union
        # validation_acc_jj = correct.item()
        #
        # Message to be printed
        msg = "[{0:d}: {1:d}/{2:d}] Val loss: {3:.6f}"
        msg = msg.format(
            epoch,
            jj,
            n_validation_samples,
            validation_loss_jj,
        )
        # Writing to the screen
        print(msg)
        # Writing to log file
        log_fout.write(msg + "\n")
        #
        # Appending data to arrays
        val_loss[jj] = validation_loss_jj
        # val_acc[jj] = validation_acc_jj
        # val_num[jj] = float(len(pred_choice.eq(target.detach())))
        #
    val_iou = val_intxn / val_union
    # --- Save model after every "save_freq" number of epochs.  Method of saving depends on whether
    # --- using DataParallel

    # --- Write mean loss/acc for epoch to file
    # Defining what data to write to the "loss" file
    loss_msg = (
        "{0:d}, {1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}\n"  # {5:.6f}, {6:.6f}\n"
    )
    loss_msg = loss_msg.format(
        epoch,
        np.mean(train_loss),
        trn_IoU,
        # np.mean(train_iou),
        np.mean(val_loss),
        val_iou,
        # np.mean(validation_iou),
    )
    # Saving data to output file
    with open(os.path.join(opt["outf"], "losses.txt"), "a") as loss_out:
        loss_out.write(loss_msg)
    #

    if val_iou > best_validation_IoU:
        print(
            f"best val_iou has improved to {val_iou} from {best_validation_IoU}"
        )
        best_validation_IoU = val_iou
        torch.save(
            classifier.state_dict(),
            "%s/cls_model_%d.pth" % (opt["outf"], epoch),
        )
        best_validation_epoch = epoch
        print(
            f"current best val_iou is {best_validation_IoU} at Epoch {best_validation_epoch}"
        )

        log_fout.write("**** Model saving ****\n")
        log_fout.write(
            "Epoch, Mean_train_loss, Train IoU, Mean_val_loss, Val IoU\n"
        )
        log_fout.write("*>" + str(loss_msg))
        log_fout.write("**** Model saved ****\n")

    else:
        if (
            epoch
            == (opt["stop_patience_fraction"] * opt["patience"])
            + best_validation_epoch
        ):
            print(
                f"best val epoch was {best_validation_epoch} \n"
                + f'we are now on epoch {epoch}, with stop patience frac of {opt["stop_patience_fraction"]}. \n'
                + "Stopping training."
            )
            # End time

            # torch.save(
            #     classifier.state_dict(),
            #     "%s/cls_final_model_%d.pth" % (opt["outf"], epoch),
            # )

            # log_fout.write(f"We are now on epoch {epoch}, with stop patience of {opt['stop_patience']}.Stopping training.")
            # end_time = datetime.now()
            # time_diff = end_time - start_time
            # #
            # # Final accuracy
            # # print(">>> Final accuracy {}".format(total_correct / float(total_valset)))
            # log_fout.write(">>> It started at: {0} \n".format(start_time))
            # log_fout.write(">>> It ended at: {0} \n".format(end_time))
            # log_fout.write(">>> It took: {0} \n".format(time_diff))
            # log_fout.close()

            break

    # --- Update learning rate
    #
    # scheduler.step()
    # print(scheduler.get_last_lr())

    scheduler.step(val_iou)
    # print(scheduler.get_lr())
log_fout.close()


torch.save(
    classifier.state_dict(),
    "%s/cls_final_model_%d.pth" % (opt["outf"], epoch),
)


# --- Once training is done, run final calculation of loss/accuracy on validation data

print("loading in best model")

classifier.load_state_dict(
    torch.load(
        "%s/cls_model_%d.pth" % (opt["outf"], best_validation_epoch),
    )
)

classifier.eval()

val_dataset_2 = FloodDataset(
    fsv="1",
    root="~/Documents/drivendata/stac-overflow/data/processed/five_sets/",
    split="val",
    data_augmentation=False,
    split_name="~/Documents/drivendata/stac-overflow/data/processed/five_sets/",
)

val_dataloader2 = torch.utils.data.DataLoader(
    val_dataset_2,
    batch_size=1,
    shuffle=False,
    num_workers=int(1),
)

intxn_val = 0
unnnn_val = 0

intxn = 0
unnnn = 0

for jj2, val_jj2 in enumerate(val_dataloader2, 0):
    inputt2 = val_jj2["chip"]
    target2 = val_jj2["label"].long()
    inputt2, target2 = inputt2.to(device), target2.to(device)

    pred2 = classifier(inputt2)
    if opt["model"] in {"PretrainedUNet_dropout", "UNetPP"}:
        pred2 = pred2[0]

    pred_val2 = torch.softmax(pred2, dim=1)[:, 1]
    pred_val2 = (pred_val2 > 0.5) * 1
    intersection2, union2 = intersection_and_union(pred_val2, target2)

    intxn += intersection2
    unnnn += union2

    intxn_val += intersection2
    unnnn_val += union2

print(intxn_val / unnnn_val)

classifier.load_state_dict(
    torch.load(
        "%s/cls_model_%d.pth" % (opt["outf"], best_validation_epoch),
    )
)

classifier.eval()

trn_dataset2 = FloodDataset(
    fsv="1",
    root="~/Documents/drivendata/stac-overflow/data/processed/five_sets/",
    split="train",
    data_augmentation=False,
    split_name="~/Documents/drivendata/stac-overflow/data/processed/five_sets/",
)

trn_dataloader2 = torch.utils.data.DataLoader(
    trn_dataset2,
    batch_size=1,
    shuffle=False,
    num_workers=int(1),
)
intxn_trn = 0
unnnn_trn = 0

for ii2, trn_ii2 in enumerate(trn_dataloader2, 0):
    inputt3 = trn_ii2["chip"]
    target3 = trn_ii2["label"].long()
    inputt3, target3 = inputt3.to(device), target3.to(device)

    pred3 = classifier(inputt3)

    if opt["model"] in {"PretrainedUNet_dropout", "UNetPP"}:
        pred3 = pred3[0]

    pred_trn3 = torch.softmax(pred3, dim=1)[:, 1]
    pred_trn3 = (pred_trn3 > 0.5) * 1

    intersection3, union3 = intersection_and_union(pred_trn3, target3)

    intxn += intersection3
    unnnn += union3

    intxn_trn += intersection3
    unnnn_trn += union3

print(intxn / unnnn)
print(intxn_trn / unnnn_trn)

final_msg = "{0:.6f}, {1:.6f}, {2:.6f}\n"
final_msg = final_msg.format(
    intxn_trn / unnnn_trn,
    intxn_val / unnnn_val,
    intxn / unnnn,
)

with open(os.path.join(opt["outf"], opt["logfile"]), "a") as log_fout:
    log_fout.write("**** All data on best model ****\n")
    log_fout.write("Train IoU, Val IoU, Global IoU\n")
    log_fout.write("&> X, " + str(final_msg))
    log_fout.write("**** All data on best model ****\n")

end_time = datetime.now()
time_diff = end_time - start_time
#
# Final accuracy
# print(">>> Final accuracy {}".format(total_correct / float(total_valset)))
print("")
#
print(">>> It started at: {0}".format(start_time))
print(">>> It ended at: {0}".format(end_time))
print(">>> It took: {0}".format(time_diff))
#
with open(os.path.join(opt["outf"], opt["logfile"]), "a") as log_fout:
    # log_fout.write("****** FINAL ACCURACY ******\n")
    # log_fout.write(
    #     "final accuracy {} \n".format(total_correct / float(total_valset))
    # )
    log_fout.write(">>> It started at: {0} \n".format(start_time))
    log_fout.write(">>> It ended at: {0} \n".format(end_time))
    log_fout.write(">>> It took: {0} \n".format(time_diff))
