# Fine-Tuning GPT-2 with Selective Layer Freezing

This repository contains code for fine-tuning the GPT-2 language model on the Wikitext-2 dataset using selective layer freezing. The model is fine-tuned by freezing most of the layers except for the last transformer block and the language modeling head.

## Table of Contents
- Introduction
- Setup
  - Requirements
  - Installation
- Usage
  - Training the Model
  - Evaluation
- Code Explanation
  - Loading and Preparing Data
  - Computing Metrics
  - Freezing Parameters
  - Training and Evaluation
- Results


## Introduction

This project demonstrates how to fine-tune the GPT-2 model on a subset of the Wikitext-2 dataset. The approach uses selective layer freezing to optimize training efficiency by updating only specific parts of the model.

## Setup

### Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Datasets library
- NumPy

### Installation

1. Clone the repository:

   git clone https://github.com/yourusername/gpt2-finetune-freeze.git
   cd gpt2-finetune-freeze

2. Install the required libraries:

pip install Requirements

3. Usage:

Training the Model
To train the model, run the following command:

python train.py

4. Evaluation:

The evaluation results, including accuracy, perplexity, and loss, will be printed after training.

5. Code Explanation: 

Loading and Preparing Data

The load_and_prepare_data function:

Loads the Wikitext-2 dataset.

Splits it into training and testing sets.

Tokenizes the text with a reduced maximum length for efficiency.

## Computing Metrics

The compute_metrics function calculates:

Accuracy: The proportion of correct predictions.

Perplexity: A measure of how well the model predicts a sample.

Loss: The cross-entropy loss between predicted and actual values.

## Freezing Parameters 

The freeze_parameters function:

Freezes all model parameters initially.
Unfreezes the last transformer block and the language modeling head for fine-tuning.

## Training and Evaluation

The train_and_evaluate_model function:

Sets up the model for training with specified arguments.

Uses the Hugging Face Trainer class for training and evaluation.

Prints the evaluation results after training.

## Main Function

The main function orchestrates the loading of data, training, and evaluation.

## Results

After running the training and evaluation, the script prints the evaluation results, including accuracy, perplexity, and loss.