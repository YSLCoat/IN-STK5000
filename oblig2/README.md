# IN-STK5000, mandatory assignment 2, fall 2022
## Group members: Espen H. Kristensen (espenhk), Cornelius Bencsik (corneb), Amir Basic (amirbas), Torstein Forseth (torfor) 

## 1. Setting up your environment
We recommend you setup a separate environment using for instance `conda`, which you can install for
your system from the site below. Make sure that you are installing a version of Conda using Python 3.9
(currently the latest installers have this):

https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Now, make sure your current folder is the root of this project. We will use `in-stk-grupetto` as the
environment name as an example below, but you could set it to anything you prefer using the
`--name` argument throughout. Create it and install the relevant packages:

~~~
conda create -y --name in-stk-grupetto python==3.9
conda install -f -y -q --name in-stk-grupetto -c conda-forge --file requirements.txt
~~~

Now finally, activate the environment:

~~~
conda activate in-stk-grupetto
~~~

You should now have a fully separate environment with all necessary packages installed.

### Environment cleanup (after use):
Once you are done using the environment, simply de-activate it using
~~~
conda deactivate
~~~
or even remove it from your system with

~~~
conda env remove -n in-stk-grupetto
~~~

## 2. Random seeds
All random seeds are set to `42`. You can set a different one using the command-line argument `--random-state` (see below).

## 3. Running experiments

To run all experiments with all default values, simply run
~~~
python main.py
~~~

Some handy arguments (try `python main.py --help` to see usage):
* `-f` or `--file`: (full or relative) path to input data file, as CSV. Default: `data/diabetes.csv` (this file is included in the repo).
* `--train` and `--test`: paths to train and test set. Overrides `--file`, if set. See "Train-test split" below.k
* `--verbose`: add to get more output to the terminal, showing more details from each step. 
* `--random-state`: Random seed used for all operations involving randomness. Default: `42`.
* `--random-samples`: If specified, samples will be randomly drawn for local interpretability. If left un-specified, pre-selected samples will be used.

An example run using all these:
~~~
python main.py --file data/more_diabetes.csv --random-state 1337 --verbose --random-samples
~~~

### Train-test split
#### Backup split
The train and test data from our default experiment run are stored as `data_backup/train.csv` and `data_backup/test.csv`. These are stored separately and static to ensure they are always available for reproducibility purposes. Thus, the `data_backup/` folder should not be written to or changed in any way.

#### Each run split
Training and test sets from each run performed are stored as `data/train.csv` and `data/test.csv` during normal execution (with `--file`).

These can be inspected independently, or you can use them as the input when running. This can be done using the `--train` and `--test` arguments:
* `--train`: path to training dataset
* `--test`: path to testing dataset

These arguments take priority over `--file`, so in case all are used (not intended), `--train` and `--test` will be
used and `--file` will be ignored.

**NB: these have to be cleaned in the same way that the `read_cleanup_dataset` function does. If you're
using training and testing sets that have not been generated from running this script with `--file`, you
have to ensure this yourself. This is not intended behavior, but it is possible.**

### Random Samples
Specifying this argument will select three random samples from the data instead of the three randomly pre-selected samples.

### Variable names - Output
Our model features will be printed out during the local interpretability part of our experiment. Just for clarification they have the following meaning:

* Gender (1/0) - 1 if male, 0 if female.
* Urination (1/0) - 1 if urination level is high, 0 if urination level is low.
* Irritability (1/0) - 1 if the individual experiencied irritability lately, 0 if not.
