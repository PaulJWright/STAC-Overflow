stac-overflow
==============================

Repository for training a Floodwater detection model. See the competition details (https://www.drivendata.org/competitions/81/detect-flood-water/) for more information.

The final model submitted consisted of training the same model (described below) twice with two random seeds and combining their predictions.

This resulted in a final:

* Public Score:  `0.8831`
* Private Score: `0.7601`
* Final Position: **13**/664

The model trained with `src/models/best_model.yml` received `0.8801` on the public scoreboard (`0.7569` on the private scoreboard).


---

# Contents
- [Project Requirements](#requirements)
- [1 - Instructions](#instructions)
   - [1.1 - Clone the repository](#clone-the-repository)
   - [1.2 - Create environment](#create-environment)
   - [1.3 - Installing all project dependencies](#installing-all-project-dependencies)
   - [1.4 - Making sure that we all follow the same coding standards](#making-sure-that-we-all-follow-the-same-coding-standards)
   - [1.5 - Before committing a piece of code](#before-committing-a-piece-of-code)
- [2 - Adding changes to the repository](#adding-changes-to-the-repository)

---
## Requirements

There are certain requirements in order to develop and run `stac-overflow`.

- Python >=3.8.10

## Instructions

In order to run this project, the developer must first follow the
next set of steps in order:

### Clone the repository

The first thing to do is to clone the repository from Gitlab.

To clone it via `SSH` keys, type the following in the terminal:

```bash
# Go into the directory, in which the repository will be stored.
cd /path/to/directory

# Cloning it via SSH keys
git clone git@github.com:PaulJWright/STAC-Overflow.git
```

or via `HTTP`:

```bash
# Go into the directory, in which the repository will be stored.
cd /path/to/directory

# Cloning it via HTTP
git clone https://github.com/PaulJWright/STAC-Overflow.git
```

### Create environment

The repository comes with a set of rules and definitions to make it easy
for the user to interact with the repository.

To create a brand new environment, one must do the following:

```bash
# Go into `stac-overflow` repository
cd /path/to/stac-overflow

# Execute the `requirements` function
make create_environment
```

This will check whether or not Anaconda is installed. If it is, the function
will create a **brand new** Anaconda environment. If not, it will create
a new virtual environment.

**NOTE**: After creating the environment, make sure to **activate** the
environment!

Activate the **Anaconda** environment via:

```bash
conda activate stac-overflow
```

### Installing all project dependencies

To make sure that the one is able to properly interact with the repository,
one must first install all the packages needed for this project.

To do this, the user must execute the following command:

```bash
make requirements
```

This will install the necessary packages via `pip` into the project's
virtual environment (either Anaconda's or virtual environment), as long
as you **have activated the environment prior to executing this command.

### Making sure that we all follow the same coding standards

This step is meant to make sure that every team member follows the same
coding guidelines and rules for the team.

To install these dependencies, the user must execute the next set of
commands:


```bash
# Install `pre-commit` hooks
make pre-commit-install
```

This will install certain hooks, as described in the `.pre-commit-config.yaml`
file.

For more information about this, we refer the user to
[pre-commit's website](https://pre-commit.com/) or this
[guide](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
about how to use `pre-commit` in a data science workflow.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
