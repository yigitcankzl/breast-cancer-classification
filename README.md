# Breast Cancer Classification Project

This project aims to predict breast cancer outcomes using machine learning models. The dataset used is from the **Breast Cancer Wisconsin (Diagnostic) dataset**, which contains various features that can be used to predict whether a tumor is malignant or benign. We apply and evaluate multiple machine learning algorithms including Logistic Regression, Random Forest, and Support Vector Machines (SVM).

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting-started)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Running the Code](#running-the-code)
6. [Models](#models)
7. [Results](#results)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

## Description

This project uses supervised learning techniques to predict breast cancer malignancy based on features such as cell radius, texture, perimeter, and others. The primary goal is to understand the relationship between these features and tumor classification, utilizing models like Logistic Regression, Random Forest, and Support Vector Machines.

## Getting Started

To get started with this project, follow the instructions below. This will guide you through setting up the environment, installing dependencies, and running the code to evaluate the models.

## Prerequisites

Before you can run the project, you need to have the following installed:

- **Python 3.6+**: The code is written in Python and requires version 3.6 or higher.
- **pip**: Python package manager to install the required dependencies.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yigitcankzl/breast-cancer-classification.git
    ```

2. Navigate to the project directory:
    ```bash
    cd breast_cancer_classification
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

After setting up the environment and installing dependencies, you can run the main script to train and evaluate the models.

1. Navigate to the `src` directory:
    ```bash
    cd src
    ```

2. Run the main program:
    ```bash
    python main.py
    ```

The output will display the training and test accuracies of each model (Logistic Regression, Random Forest, SVM), along with cross-validation scores and predictions for a sample instance.

## Models

In this project, we use the following models:

### Logistic Regression

A linear model that predicts the probability of a tumor being malignant or benign.

- **Training Accuracy**: ~98%
- **Test Accuracy**: ~97%

### Random Forest

An ensemble learning model that builds multiple decision trees and combines their results.

- **Training Accuracy**: 100%
- **Test Accuracy**: ~95%

### Support Vector Machine (SVM)

A powerful classifier that works well in high-dimensional spaces.

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~96%

## Results

The models were evaluated on their ability to predict whether a tumor is malignant or benign. All models performed well with high accuracy rates, though Random Forest showed perfect training accuracy, while Logistic Regression and SVM had slightly better test accuracies.

### Example Predictions:
- **Logistic Regression Prediction**: Malignant
- **Random Forest Prediction**: Malignant
- **SVM Prediction**: Malignant

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The **Breast Cancer Wisconsin dataset** was provided by the UCI Machine Learning Repository.
- Thanks to the contributors of the `scikit-learn` library for their valuable machine learning algorithms.

