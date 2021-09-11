stac-overflow
==============================

Repository for training a Floodwater detection model. See the competition details (https://www.drivendata.org/competitions/81/detect-flood-water/) for more information.

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

or the virtual environment:

```bash
source activate stac-overflow
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

### Before committing a piece of code

Before a user is able to commit and push any changes made to the repository,
these changes have to first be run through `pre-commit` and **pass all checks**.

To check that all of the changes are in agreement to the team's
coding rules and guidelines, one can run a **manual check** by executing
the following command:

```bash
# Running manual lint check
make lint
```

This will provide a screen that will inform the user of any issues with the
code, and it will also **reformat** the code according to the
project's rules.

Example of the output after having run the `make lint1 command:

```bash
pre-commit run -a --hook-stage manual
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
Check Yaml...............................................................Passed
Check JSON...........................................(no files to check)Skipped
Check for added large files..............................................Passed
Check for case conflicts.................................................Passed
Check for merge conflicts................................................Passed
Debug Statements (Python)................................................Passed
Fix requirements.txt.....................................................Passed
Flake8...................................................................Passed
black....................................................................Passed
isort....................................................................Passed
seed isort known_third_party.............................................Passed
```

**NOTE**: Only until all checks have passed, the user is able to commit and push
the code to a designated branch!

## Adding changes to the repository

The `main` branch will always be the **clean** and **working** version
of the repository. Any changes to the `main` branch have to go under
code-review via a *Pull Request* (PR).

When making changes to the repository, the user must:

1. Create a new branch
   ```bash
   git checkout -b <name-of-branch>
   ```

2. Make all the changes necessary to the code.
3. Test all of your changes.
4. Make sure that they are in agreement to the team's coding standards by
   running the following command:
   ```bash
   make lint
   ```
   This will reformat the code and point out any issues with the script.
5. Once all tests have passed, the user is able to commit the changes and
   push them to the current branch.
6. Then one can submit a Pull Request, and one of more teammembers will
   evaluate the changes. If all changes are approved, the changes will be
   incorporated into the `main` branch and the branch will be deleted.


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
