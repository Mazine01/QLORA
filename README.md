# Project: Efficient LLM Fine-Tuning for Security Classification (QLoRA)

This repository showcases the conceptual implementation and technical design for a Parameter-Efficient Fine-Tuning (PEFT) pipeline. The goal is to adapt a massive Large Language Model (LLM) (e.g., Llama 2 7B) for the specialized task of Security Vulnerability Classification.

The core focus is on utilizing QLoRA (Quantized Low-Rank Adaptation) to drastically reduce GPU memory consumption (VRAM) and enable training on resource-constrained hardware, demonstrating crucial MLOps optimization skills.

# 1. Demonstrated Core Skills

Advanced LLM Adaptation: Proficiency with LoRA and QLoRA.

Resource Optimization: Implementing 4-bit Normal-Float (NF4) quantization.

Reproducibility: Using the Anaconda/Conda environment for stable dependency management.

Hugging Face Ecosystem: Expertise with transformers, peft, and bitsandbytes.

# 2. Technical Implementation Summary

The script finetuning_setup.py details the five critical stages of the QLoRA pipeline:

Quantization Setup: Configuration of BitsAndBytesConfig for 4-bit loading.

Model Loading: Loading the base LLM weights in a frozen, 4-bit state.

Model Preparation: Enabling gradient checkpointing and preparing layers for k-bit stability.

LoRA Configuration: Defining the small, trainable adapter matrices (r=16, lora_alpha=32).

Adapter Application: Attaching the PEFT adapters, resulting in less than 1% of the model parameters being trainable.

# 3. Environment Setup (MANDATORY: Anaconda/Conda)

Due to the heavy GPU/CUDA requirements of bitsandbytes, this project MUST be run within a dedicated Conda environment.

Step A: Create and Activate Environment

# Create a new environment named 'qlora-env' with Python 3.10
conda create -n qlora-env python=3.10
# Activate the new environment
conda activate qlora-env


Step B: Install Dependencies

Ensure your PyTorch installation command matches your local CUDA version.

# 1. Install PyTorch (Example for CUDA 11.8 - CHECK YOUR VERSION)
# This is a critical step for BITSANDBYTES compatibility.
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 2. Install Hugging Face and PEFT libraries
pip install transformers peft accelerate datasets bitsandbytes scikit-learn


# 4. Running the Conceptual Setup

The finetuning_setup.py file demonstrates the necessary imports and configuration to prepare the model for fine-tuning.

python finetuning_setup.py


Note: Running this script will attempt to download and load a multi-billion parameter model (e.g., Llama 2 7B) and may fail in environments without sufficient VRAM (typically 12GB+). The purpose of this script is to demonstrate the correct configuration code, not to execute a full training job.

# 5. Deployment Advantage

The resulting trained model is a lightweight LoRA Adapter file (only a few megabytes) that can be easily stored, versioned, and merged with the base LLM during inference, proving scalability and efficiency for production deployment.
