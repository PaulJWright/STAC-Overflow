from __future__ import print_function

import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

import albumentations
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import yaml
from dataset import FloodDataset
from torchsampler import ImbalancedDatasetSampler


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

    def forward(self, pred, true):
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

        # print(f'({opt["xentropy"]} * xe_loss) + ({opt["xdice"]} * dice_loss)')
        return (opt["xentropy"] * xe_loss) + (opt["xdice"] * dice_loss)


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

# --- Load parameters for training
with open("training_parameters.yml") as file:
    opt = yaml.load(file, Loader=yaml.FullLoader)

print(opt)

# --- Ensure torch is using deterministic algorithms for reproducibility
torch.set_deterministic(opt["torch_deterministic"])
# --- Set buffer size to force deterministic results from cuDNN
if opt["torch_deterministic"]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# --- Set random seed
if opt["manualSeed"] == 0:
    # Chosen randomly if not set in parameter file
    opt["manualSeed"] = random.randint(1, 10000)

print("Random Seed: ", opt["manualSeed"])
random.seed(opt["manualSeed"])
np.random.seed(opt["manualSeed"])
torch.manual_seed(opt["manualSeed"])

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
shutil.copyfile(
    "training_parameters.yml",
    os.path.join(opt["outf"], "training_parameters.yml"),
)

# --- Set dataset type depending on problem
if opt["dataset_type"] == "FloodDataset":

    # https://github.com/albumentations-team/albumentations#pixel-level-transforms

    transformations = [
        albumentations.RandomCrop(
            opt["crop_size_square"], opt["crop_size_square"]
        ),
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
        root=opt["dataset"],
        split="train",
        data_augmentation=opt["augmentation"],
        split_name=opt["split_name"],
        transforms=training_transformations,
    )

    # --- Load modelnet validation set
    # --- Data_augmentation will always be off for validation
    val_dataset = FloodDataset(
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

# https://pytorch.org/docs/stable/data.html#
# sampler = torch.utils.data.WeightedRandomSampler(
#     weights,
#     num_samples,
#     replacement=True,
#     generator=None)


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
        # aux_params=dict(
        #     # pooling="avg",  # one of 'avg', 'max'; defauly is "avg"
        #     dropout=0.99,  # dropout ratio, default is None
        #     # activation=None,  # activation function, default is None
        #     classes=int(2),  # define number of output labels
        # ),
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
            dropout=0.5,  # dropout ratio, default is None
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

    classifier = smp.Unet(
        encoder_name=opt["PretrainedUNet_backbone"],  # resnet34
        encoder_weights=opt["PretrainedUNet_weights"],  # imagenet
        in_channels=2,  # in_chnnl
        classes=2,
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
            dropout=0.2,  # dropout ratio, default is None
            # activation=None,  # activation function, default is None
            classes=int(2),  # define number of output labels
        ),
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

# # --- Learning rate scheduler
# scheduler = optim.lr_scheduler.StepLR(
#     optimizer, step_size=opt["step_size"], gamma=opt["gamma"]
# )

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
loss_out.write("Epoch, Mean_train_loss, Train IoU, Mean_val_loss, Val IoU \n")
loss_out.close()

trn_intxn = 0
trn_union = 0
trn_IoU = 0
# --- Training the model
for epoch in range(opt["nepoch"]):
    #
    # --- Training of the model
    # - Initializing arrays to store batch loss/acc for each epoch, to
    # - average later.
    n_train_elem = len(dataloader)
    train_loss = [[] for x in range(n_train_elem)]
    train_acc = [[] for x in range(n_train_elem)]
    train_num = [[] for x in range(n_train_elem)]
    #
    # - Looping over `dataloader` (Training)
    for ii, data_ii in enumerate(dataloader, 0):
        # Extracting the data points and target (classes)
        inputt = data_ii["chip"]
        target = data_ii["label"].long()
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
        # Computing loss

        if opt["weights"] is True:
            print("true")
            weights = torch.tensor(
                [
                    1
                    # / np.log(
                    / (
                        opt["segmentation_weights_heuristic"]
                        + 1.0
                        - opt["water_perc"]
                    ),
                    1
                    # / np.log(
                    / (
                        opt["segmentation_weights_heuristic"]
                        + opt["water_perc"]
                    ),
                ]
            ).type(torch.FloatTensor)
            # print(weights)
            weights = weights.to(device)
            criterion = XEDiceLoss(weights)
        else:
            criterion = XEDiceLoss(None)

        loss = criterion(pred, target)
        # Regularization if feature transform is being used
        loss.backward()
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
    val_IoU = [[] for x in range(n_validation_samples)]
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
        criterion = XEDiceLoss(None)
        loss = criterion(pred, target)
        # Computing statistics
        validation_loss_jj = loss.item()
        intersection, union = intersection_and_union(pred_val, target)
        val_intxn += intersection
        val_union += union
        # validation_acc_jj = correct.item()
        #
        # Message to be printed
        msg = "[{0:d}: {1:d}/{2:d}] {3} loss: {4:.6f}"
        msg = msg.format(
            epoch,
            jj,
            n_validation_samples,
            "\033[94m" + "val" + "\033[0m",
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
    if epoch % opt["save_freq"] == 0:
        # if torch.cuda.device_count() > 1:
        #    torch.save(
        #        classifier.module.state_dict(),
        #        "%s/cls_model_%d.pth" % (opt["outf"], epoch),
        #    )
        # else:
        # torch.save(
        #     classifier.state_dict(),
        #     "%s/cls_model_%d.pth" % (opt["outf"], epoch),
        # )
        pass

    # --- Write mean loss/acc for epoch to file
    # Defining what data to write to the "loss" file
    loss_msg = "{0:d}, {1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}\n"
    loss_msg = loss_msg.format(
        epoch,
        np.mean(train_loss),
        trn_IoU,
        np.mean(val_loss),
        val_iou,
    )
    # Saving data to output file
    with open(os.path.join(opt["outf"], "losses.txt"), "a") as loss_out:
        loss_out.write(loss_msg)
    #

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

# run through val data
for jj, val_jj in enumerate(valdataloader, 0):
    # Extracting points and targets (classes)
    inputt = val_jj["chip"]
    target = val_jj["label"].long()
    # Putting the data into a GPU, if applicable
    inputt, target = inputt.to(device), target.to(device)
    # computing prediction
    pred = classifier(inputt)

    if opt["model"] in {"PretrainedUNet_dropout", "UNetPP"}:
        pred = pred[0]

    pred_val = torch.softmax(pred, dim=1)[:, 1]
    pred_val = (pred_val > 0.5) * 1

    np.save(
        "%s/val_%d" % (opt["outf"], jj),
        [inputt.cpu(), target.cpu(), pred_val.cpu()],
    )

for ii, trn_ii in enumerate(dataloader, 0):
    inputt = trn_ii["chip"]
    target = trn_ii["label"].long()
    # Putting the data into a GPU, if applicable
    inputt, target = inputt.to(device), target.to(device)
    # computing prediction
    pred = classifier(inputt)

    if opt["model"] in {"PretrainedUNet_dropout", "UNetPP"}:
        pred = pred[0]

    pred_val = torch.softmax(pred, dim=1)[:, 1]
    pred_val = (pred_val > 0.5) * 1

    np.save(
        "%s/train_%d" % (opt["outf"], ii),
        [inputt.cpu(), target.cpu(), pred_val.cpu()],
    )

#
# End time
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
