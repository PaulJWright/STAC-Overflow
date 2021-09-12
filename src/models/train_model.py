from __future__ import print_function

import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import yaml
from tqdm import tqdm

from data.dataset import FloodDataset

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

    # --- Load modelnet training set
    dataset = FloodDataset(
        root=opt["dataset"],
        split="train",
        npoints=opt["num_points"],
        data_augmentation=opt["augmentation"],
        split_name=opt["split_name"],
    )

    # --- Load modelnet validation set
    # --- Data_augmentation will always be off for validation
    val_dataset = FloodDataset(
        root=opt["dataset"],
        split="val",
        npoints=opt["num_points"],
        data_augmentation=False,
        split_name=opt["split_name"],
    )


# --- Load up training data
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt["batchSize"],
    shuffle=True,
    num_workers=int(opt["workers"]),
)

# --- Load up validation data
valdataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt["batchSize"],
    shuffle=True,
    num_workers=int(opt["workers"]),
)

# --- Store the number of classes for classification
print(len(dataset), len(val_dataset))
num_classes = len(dataset.classes)
log_fout.write("classes\t" + str(num_classes) + "\n")
print(dataset.cat)
log_fout.write(str(dataset.cat) + "\n")

# --- If loading a pretrained model
if opt["model"] == "PretrainedUNet":

    import segmentation_models_pytorch as smp

    print(
        f"training {opt['model']} from {opt['PretrainedUNet_weights']}"
        + f"with {num_classes} classes"
    )

    classifier = smp.Unet(
        # https://github.com/qubvel/segmentation_models.pytorch/ \
        # blob/master/segmentation_models_pytorch/unet/model.py
        encoder_name=opt["PretrainedUNet_backbone"],  # resnet34
        # encoder_depth=5, # default
        encoder_weights=opt["PretrainedUNet_weights"],  # imagenet
        # decoder_use_batchnorm=True # default
        # decoder_channels=(256, 128, 64, 32, 16), # default
        # decoder_attention_type=None,
        in_channels=2,  # in_chnnl
        classes=opt["classes"],
        # activation: Optional[Union[str, callable]] = None,
        # aux_params: Optional[dict] = None,
        # aux_params=dict(
        #     pooling="avg",  # one of 'avg', 'max'
        #     dropout=0.5,  # dropout ratio, default is None
        #     activation="sigmoid",  # activation function, default is None
        #     classes=k,  # define number of output labels
        # ),
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
else:
    sys.exit("This optimizer is not implemented:", opt["optimizer"])

# --- Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=opt["step_size"], gamma=opt["gamma"]
)

# --- Load model onto appropriate device
classifier = classifier.to(device)

# --- If there are multiple GPUs to run on, use DataParallel
# if torch.cuda.device_count() > 1:
#     classifier = torch.nn.DataParallel(classifier)

# --- Print header to logfile before beginning training
log_fout.write("****** TRAINING ******\n")
loss_out = open(os.path.join(opt["outf"], "losses.txt"), "a")
loss_out.write(
    "Epoch, Mean_train_loss, Mean_train_acc, Mean_val_loss, Mean_val_acc \n"
)
loss_out.close()

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
    for ii, data in enumerate(dataloader, 0):
        # Extracting the data points and target (classes)
        points, target = data
        # Extracting targets
        target = target[:, 0]
        # Transposing points
        points = points.transpose(2, 1)
        # Putting the data into a GPU, if applicable
        points, target = points.to(device), target.to(device)
        # Initializing the optimizer and making the gradient zero
        optimizer.zero_grad()
        # Sets the module in training mode.
        classifier = classifier.train()
        #
        pred, trans, trans_feat = classifier(points)
        # Computing loss
        loss = F.nll_loss(pred, target)
        # Regularization if feature transform is being used
        loss.backward()
        optimizer.step()
        pred_choice = pred.detach().max(1)[1]
        correct = pred_choice.eq(target.detach()).sum()
        # Computing statistics
        train_loss_ii = loss.item()
        train_acc_ii = correct.item()
        #
        # Message to be printed
        msg = "[{0:d}: {1:d}/{2:d}] Train loss: {3:.6f} Accuracy: {4:.6f}"
        msg = msg.format(
            epoch,
            ii,
            n_train_elem,
            train_loss_ii,
            train_acc_ii / float(len(pred_choice.eq(target.detach()).cpu())),
        )
        # Writing to the screen
        print(msg)
        # Writing to log file
        log_fout.write(msg + "\n")
        # Appending data to arrays
        train_loss[ii] = train_loss_ii
        train_acc[ii] = train_acc_ii
        train_num[ii] = float(len(pred_choice.eq(target.detach()).cpu()))
    #
    # --- Validation of the model
    # - Initializing arrays to store batch loss/acc for each epoch, to
    # - average later.
    #  Number of elements in the validation dataset
    n_validation_samples = len(valdataloader)
    val_loss = [[] for x in range(n_validation_samples)]
    val_acc = [[] for x in range(n_validation_samples)]
    val_num = [[] for x in range(n_validation_samples)]
    # Putting model into ``evaluation`` mode
    classifier = classifier.eval()
    # Looping over the validation dataset
    for jj, val_jj in enumerate(valdataloader, 0):
        # Extracting points and targets (classes)
        points, target = val_jj
        target = target[:, 0]
        # Transposing the points of the dataset
        points = points.transpose(2, 1)
        # Sending objects to GPU, if applicable
        points, target = points.to(device), target.to(device)
        #
        pred, _, trans_feat_val = classifier(points)
        # Computing loss value
        loss = F.nll_loss(pred, target)
        # --- Regularization if feature transform is being used

        pred_choice = pred.detach().max(dim=1)[1]
        correct = pred_choice.eq(target.detach()).sum()
        # Computing statistics
        validation_loss_jj = loss.item()
        validation_acc_jj = correct.item()
        #
        # Message to be printed
        msg = "[{0:d}: {1:d}/{2:d}] {3} loss: {4:.6f} Accuracy: {5:.6f}"
        msg = msg.format(
            epoch,
            jj,
            n_validation_samples,
            "\033[94m" + "val" + "\033[0m",
            validation_loss_jj,
            validation_acc_jj / float(len(pred_choice.eq(target.detach()))),
        )
        # Writing to the screen
        print(msg)
        # Writing to log file
        log_fout.write(msg + "\n")
        #
        # Appending data to arrays
        val_loss[jj] = validation_loss_jj
        val_acc[jj] = validation_acc_jj
        val_num[jj] = float(len(pred_choice.eq(target.detach())))
    #
    # --- Save model after every "save_freq" number of epochs.  Method of saving depends on whether
    # --- using DataParallel
    if epoch % opt["save_freq"] == 0:
        if torch.cuda.device_count() > 1:
            torch.save(
                classifier.module.state_dict(),
                "%s/cls_model_%d.pth" % (opt["outf"], epoch),
            )
        else:
            torch.save(
                classifier.state_dict(),
                "%s/cls_model_%d.pth" % (opt["outf"], epoch),
            )
    # --- Write mean loss/acc for epoch to file
    # Defining what data to write to the "loss" file
    loss_msg = "{0:d}, {1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}\n"
    loss_msg = loss_msg.format(
        epoch,
        np.mean(train_loss),
        np.sum(train_acc) / np.sum(train_num),
        np.mean(val_loss),
        np.sum(val_acc) / np.sum(val_num),
    )
    # Saving data to output file
    with open(os.path.join(opt["outf"], "losses.txt"), "a") as loss_out:
        loss_out.write(loss_msg)
    #
    # --- Update learning rate
    scheduler.step()
log_fout.close()

# Write final model
if torch.cuda.device_count() > 1:
    torch.save(
        classifier.module.state_dict(),
        "%s/cls_model_final.pth" % (opt["outf"]),
    )
else:
    torch.save(
        classifier.state_dict(),
        "%s/cls_model_final.pth" % (opt["outf"]),
    )


# --- Once training is done, run final calculation of loss/accuracy on validation data
total_correct = 0
total_valset = 0
for i, data in enumerate(tqdm(valdataloader), 0):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.to(device), target.to(device)
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.detach().max(1)[1]
    correct = pred_choice.eq(target.detach()).cpu().sum()
    total_correct += correct.item()
    total_valset += points.size()[0]

#
# End time
end_time = datetime.now()
time_diff = end_time - start_time
#
# Final accuracy
print(">>> Final accuracy {}".format(total_correct / float(total_valset)))
print("")
#
print(">>> It started at: {0}".format(start_time))
print(">>> It ended at: {0}".format(end_time))
print(">>> It took: {0}".format(time_diff))
#
with open(os.path.join(opt["outf"], opt["logfile"]), "a") as log_fout:
    log_fout.write("****** FINAL ACCURACY ******\n")
    log_fout.write(
        "final accuracy {} \n".format(total_correct / float(total_valset))
    )
    log_fout.write(">>> It started at: {0} \n".format(start_time))
    log_fout.write(">>> It ended at: {0} \n".format(end_time))
    log_fout.write(">>> It took: {0} \n".format(time_diff))