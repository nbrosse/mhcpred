# MHCpred

## Installation

To install the mhcpred package, do 

    poetry env use 3.9.*
    poetry activate
    poetry install

You have to set-up the correct paths for the data, outputs and models.
Please modify the `settings.toml` file accordingly.

The code is heavily inspired from [mhcflurry package](https://github.com/openvax/mhcflurry).

## notebooks

We have two notebooks: 

- `EDA.ipynb` which contains an EDA analysis.
- `metrics.ipynb` which analyses the predictions through different metrics.

## mhcflurry benchmark

Follow the documentation on https://openvax.github.io/mhcflurry/commandline_tutorial.html

    mhcflurry-downloads fetch models_class1_presentation
    python scripts/mhcflurry_benchmark.py

## mhcpred scripts

We take all the folds and do a random split for validation data.

    python scripts/launch_training_inference.py

## git branch - class1_binary_predictor

There is a git branch `class1_binary_predictor` which contains a `class1_binary_predictor.py` 
inspired from the predictor of mhcflurry. The code is better organised, unfortunately I had a 

    Process finished with exit code 138 (interrupted by signal 10: SIGBUS)

error, in the middle of the training. I was not able to solve the error.
To reproduce the error, inside the git branch

    python scripts/launch_training_predictor.py

## Improvements

Here we suggest some improvements if we had more time on the project.
Of course, the code here is a very simple version, and there are many possible ways to improve the current version.

- Use the folds to tune the hyperparameters or train an ensemble of networks 
- improve the architecture of the models: transformer, convolution.
- augment the dataset including BA data and EL MA data.


/Users/nicolasbrosse/Library/Application Support/mhcflurry/4/2.2.0
