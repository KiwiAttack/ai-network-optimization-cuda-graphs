# CUDA Graphs for AI Network Optimization

This repository contains the code and profiling data for the experimental evaluation of the performance impact of CUDA Graphs on various AI network architectures. The experiments were conducted as part of a bachelor's thesis on optimizing AI networks using CUDA Graphs.

## Overview
The code in this repository covers the implementation and benchmarking of the following neural network architectures:
- Convolutional Neural Network (CNN)
- Vision Transformer (ViT)
- TimeSformer

For each architecture, experiments were performed with and without the use of CUDA Graphs on different GPU systems:
- NVIDIA GeForce RTX 3080
- NVIDIA Tesla V100
- NVIDIA Tesla T4


The profiling data includes runtime measurements, CPU and GPU utilization profiles, as well as traces for training and inference runs.

## Repository Structure
- **huggingface/:** Contains the Python source code for model implementation and inference on ViT and TimeSformer.
- **Cifar10/:** Contains the Python source code for model implementation, training and inference on a classic CNN.
- **./log/:** Includes the profiling data files generated during the experiments.
- **README.md:** This file, providing an overview of the repository.

## Datasets
The actual training and inference datasets used in the experiments are not included in this repository due to potential data privacy concerns. 
- CNN used the [Cifar-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- ViT used the [ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) from Kaggle
- TimeSformer used 50 videos from the [Kinetics dataset](https://github.com/cvdfoundation/kinetics-dataset)
  
The datasets employed are described in detail in the bachelor's thesis.

## Dependencies
The code was implemented using the following dependencies:
- PyTorch 2.2.1
- CUDA 12.1
- Hugging Face Transformers
