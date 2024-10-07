# Allloy-Detection

## Introduction
This project aims to identify and classify different types of alloys using machine learning techniques based on the energy formation.

## Goals
The primary goal of this project is to accurately detect energy formation per atom given the DFT dataset.

## Contributors
Subham Adhikari

## Project Architecture


# Status
## Known Issue
- None at this time.

## High Level Next Steps
- Enhance the model accuracy by experimenting with different algorithms and feature selection methods.
- Improve data processing and feature engineering to better capture relevant alloy characteristics.


# Usage
## Installation
To begin this project, use the included `Makefile`

#### Creating Virtual Environment

This package is built using `python-3.8`. 
We recommend creating a virtual environment and using a matching version to ensure compatibility.

#### pre-commit

`pre-commit` will automatically format and lint your code. You can install using this by using
`make use-pre-commit`. It will take effect on your next `git commit`

#### pip-tools

The method of managing dependencies in this package is using `pip-tools`. To begin, run `make use-pip-tools` to install. 

Then when adding a new package requirement, update the `requirements.in` file with 
the package name. You can include a specific version if desired but it is not necessary. 

To install and use the new dependency you can run `make deps-install` or equivalently `make`

If you have other packages installed in the environment that are no longer needed, you can you `make deps-sync` to ensure that your current development environment matches the `requirements` files. 

## Usage Instructions
1. Clone the repository.
2. Set up your environment using the installation instructions above.
3. Use the notebook file `AlloyIdentification.ipynb` to run the data processing, visualization, and classification steps.


# Data Source
- Data includes structural and physical properties of various alloys.
- Link : `https://zenodo.org/records/10854500`

## Code Structure
- The primary notebook is `AlloyIdentification.ipynb`.
- Scripts for data processing, visualization, and model training are included within the notebook.

## Artifacts Location
- Results and model artifacts are saved in the specified directories within the local computer.

# Results
## Metrics Used
- RMSE, MAE, R2 to evaluate model performance.

## Evaluation Results
Results are saved and displayed within the notebook. Specific metrics can be further analyzed by running the evaluation cells in the notebook.