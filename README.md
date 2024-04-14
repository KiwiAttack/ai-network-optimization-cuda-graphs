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
huggingface/: Contains the Python source code for model implementation and inference on ViT and timeSformer.
Cifar10/: Contains the Python source code for model implementation, training and inference on a classic CNN.
./log/: Includes the profiling data files generated during the experiments.
README.md: This file, providing an overview of the repository.

## Dependencies
The code was implemented using the following dependencies:
- PyTorch 2.2.1
- CUDA 12.1
- Hugging Face Transformers
