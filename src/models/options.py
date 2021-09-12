import argparse

import torch

# from src import default_variables as dv


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # --- Project-specific
        # Verbose output
        self.parser.add_argument(
            "-v",
            "--verbose",
            dest="doVerbose",
            action="store_true",
            help="""
            Option for outputting a verbosed output. [Default: '%(default)s']
            """,
        )
        # --- Required variables
        # NOTE: This variables are the ones, for which the model will
        # produce an inference.

        #
        # --- Models-specific
        # Model to use
        self.parser.add_argument(
            "--model",
            type=str,
            default="PretrainedUNet",
            help="""
            Model to use.
            [Default: '%(default)s']
            """,
        )

        # --- Pretrained UNet Specific
        self.parser.add_argument(
            "--PretrainedUNet_backbone",
            type=str,
            default="resnet34",
            help="""
            Model to use.
            [Default: '%(default)s']
            """,
        )
        self.parser.add_argument(
            "--PretrainedUNet_weights",
            type=str,
            default="imagenet",
            help="""
            Model to use.
            [Default: '%(default)s']
            """,
        )
        # --- /Pretrained UNet Specific

        # Momentum - Step-size
        self.parser.add_argument(
            "--mom_step_size",
            type=int,
            default=20,
            help="""
            Step size for decreasing BN momentum.
            [Default: '%(default)s']
            """,
        )
        # Momentum - Gamm variable
        self.parser.add_argument(
            "--mom_gamma",
            type=float,
            default=0.5,
            help="""
            Factor for decreate BN momentum.
            [Default: '%(default)s']
            """,
        )
        # Momentum initiazation
        self.parser.add_argument(
            "--mom_init",
            type=float,
            default=0.1,
            help="""
            Initial momentum for batch normalization.
            [Default: '%(default)s']
            """,
        )
        # Beta-1 variable
        self.parser.add_argument(
            "--beta1",
            type=float,
            default=0.9,
            help="""
            Beta1 parameter for PointNet.
            [Default: '%(default)s']""",
        )
        # Beta-2 variable
        self.parser.add_argument(
            "--beta2",
            type=float,
            default=0.999,
            help="""
            Beta2 parameter for PointNet.
            [Default: '%(default)s']
            """,
        )
        # Gamma variable
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=0.5,
            help="""
            Rate for decreasing learning rate.
            [Default: '%(default)s']""",
        )
        # Learning rate
        self.parser.add_argument(
            "--lr",
            type=float,
            default=5e-5,
            help="""
            Learning rate.
            [Default: '%(default)s']
            """,
        )
        # Step size for decreasing learning rate
        self.parser.add_argument(
            "--step_size",
            type=int,
            default=20,
            help="""
            Step size for decreasing learning rate.
            [Default: '%(default)s']
            """,
        )
        # Type of optimizer to use
        self.parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            help="""
            Optimizer to use.
            [Default: '%(default)s']
            """,
        )
        # Name of the split to use for the model.
        self.parser.add_argument(
            "--split_name",
            type=str,
            default="/data/scratch/mackenzie/handoff/towernet_handoff_",
            help="""
            Split name.
            [Default: '%(default)s']
            """,
        )
        # Path to the dataset to use
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="/data/5x5-ai-rd-S3/internal-data/point-cloud/towers/cropped_boxes_pc",
            help="""
            Dataset path.
            [Default: '%(default)s']
            """,
        )
        # Augmentation
        self.parser.add_argument(
            "--augmentation",
            type=bool,
            default=True,
            help="""
            If true, will implement augmentation during training.
            [Default: '%(default)s']
            """,
        )
        # Type of dataset to use
        self.parser.add_argument(
            "--dataset_type",
            type=str,
            default="towernet",
            help="""
            Dataset type.
            [Default: '%(default)s']
            """,
        )
        # Option for whether to use "Torch-deterministic" option or not
        self.parser.add_argument(
            "--torch_deterministic",
            type=bool,
            default=False,
            help="""
            If true, will force torch to use deterministic algorithms.
            [Default: '%(default)s']
            """,
        )
        # Number of epoch to train the model for.
        self.parser.add_argument(
            "--nepoch",
            type=int,
            default=100,
            help="""
            Number of epochs [Default: '%(default)s']
            """,
        )
        # Epoch frequency with which to save a model.
        self.parser.add_argument(
            "--save_freq",
            type=int,
            default=10,
            help="""
            Epoch frequency with which to save model.
            [Default: '%(default)s']
            """,
        )
        # Manual random seed
        self.parser.add_argument(
            "--manualSeed",
            type=int,
            default=25,
            help="""
            Manually set random seed.  If 0, random number will be chosen
            to set seed.
            [Default: '%(default)s']
            """,
        )
        # Name of the log-file to use.
        self.parser.add_argument(
            "--logfile",
            type=str,
            default="logfile.txt",
            help="""
            Logfile name
            [Default: '%(default)s']
            """,
        )
        # Output directory
        self.parser.add_argument(
            "--outf",
            type=str,
            default="./outf",
            help="""
            Output folder
            [Default: '%(default)s']
            """,
        )
        # Number of data loading workers.
        self.parser.add_argument(
            "--workers",
            type=int,
            default=16,
            help="""
            Number of data loading workers [Default: '%(default)s']
            """,
        )
        # Batch size
        self.parser.add_argument(
            "--batchSize",
            type=int,
            default=16,
            help="""
            Batch size [Default: '%(default)s']
            """,
        )

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        args = vars(self.opt)

        return args
