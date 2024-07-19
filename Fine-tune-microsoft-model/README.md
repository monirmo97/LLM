## Fine-tune Microsoft Model

    This project aims to fine-tune a pre-trained language model to understand and generate context based on personal data. Initially, the goal was to fine-tune the Mistral model, but due to memory constraints, the task was performed using microsoft/Phi-3-mini-4k-instruct.

## Table of Contents

    Overview
    Setup
    Dataset Creation
    Installation

## Overview

    This project involves fine-tuning a pre-trained language model to understand the context of personal data. The fine-tuned model can generate coherent continuations or responses to given journal-like notes. Due to memory limitations, the microsoft/Phi-3-mini-4k-instruct model was used instead of the intended Mistral model.

## Setup

    Clone the repository to your local machine:
    git clone "git address"



## Dataset Creation

    Create a dataset of personal data for training and validation. Define your training and validation entries and save them in JSONL format.

## Installation

    Install the required libraries:

    pip install bitsandbytes transformers peft accelerate datasets scipy ipywidgets
    pip install -q wandb -U
    Log in to Weights & Biases:
