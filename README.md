# TimeCapsuleLLM on AMD ROCm

This repository contains inference code for the haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B model using PyTorch with AMD ROCm support.

It is specifically configured to resolve memory access violations and segmentation faults encountered while using rocm.

## Prerequisites

The setup script requires Python 3.11 and uv. The wheel files referenced are built specifically for Linux x86_64 and Python 3.11.

## Installation

Run the setup.sh script to create a virtual environment and install the required PyTorch wheels and dependencies.

## Usage

Activate the virtual environment with source .venv/bin/activate and execute main.py.
