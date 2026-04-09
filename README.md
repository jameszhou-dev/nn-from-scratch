# Neural Network from Scratch in C++

A fully functional neural network implementation built from scratch in C++, without any ML libraries. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Overview

This project implements a scalar-valued autograd engine and a multi-layer perceptron (MLP) trained on the Iris dataset, achieving ~95% classification accuracy.

## Features

- Scalar autograd engine
- Backpropagation w/ arbitrary operations
- Multi-layer perceptron (MLP) w/ configurable layers
- MSE, cross-entropy loss, and one-hot encoding for multiclass classification
- Mini-batch gradient descent w/ epoch shuffling

## Results

| Metric | Value |
|--------|-------|
| Dataset | Iris (150 samples) |
| Train/Test Split | 80/20 |
| Epochs | 100 |
| Batch Size | 8 |
| Learning Rate | 0.05 |
| Test Accuracy | ~95% |

## Usage

```bash
# Compile
g++ nn-components.cpp bin-class-nn.cpp -o nn -std=c++17

# Download dataset
curl -o data/iris.csv https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv

# Run
./nn
```
