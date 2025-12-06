# FLUX Image-to-Image Alignment Pipeline

This repository contains a pipeline for training alignment models to detect and mitigate inappropriate sexualization in image-to-image transformations using FLUX models.

## Overview

The pipeline consists of four main stages:

Binary Classifier – Detects sexualization in image transformations

Reward Model – Ranks image outputs by safety score

Data Generation – Creates synthetic training data with FLUX Kontext

DPO Fine-tuning – Aligns the model using Direct Preference Optimization

Project Structure \
classifier.py              # Binary sexualization classifier \
reward_model.py            # Reward model for ranking outputs \
data_generation.py         # Synthetic data generation \
data_augmentation.py       # Prompt augmentation utilities \
sft_finetune.py            # Supervised fine-tuning script \
dpo_finetune.py            # DPO alignment training \
requirements.txt           # Python dependencies \
README.md                  # This file 

### Generate training data

First, create a CSV file with your input images and prompts:

from data_generation import create_example_csv, generate_training_data

#### Create example CSV
create_example_csv("inputs.csv")

#### Generate synthetic data
generate_training_data(
    csv_path="inputs.csv",
    output_dir="./generated_data",
    num_variations=3,
    num_inference_steps=50
)

Expected CSV format
input_image,input_prompt
path/to/image1.jpg,"make me a princess"
path/to/image2.png,"put me on the beach"


Once generated, annotate the dataset with sexualization labels.

### Train binary classifier

After labeling the dataset (0 = benign, 1 = violating), train the classifier:

from classifier import ClassifierConfig, SexualizationClassifier, train_classifier
from transformers import AutoProcessor

config = ClassifierConfig( \ 
    model_name="meta-llama/Llama-3.2-11B-Vision", \ 
    learning_rate=2e-6, \ 
    batch_size=8, \ 
    num_epochs=3 \ 
)

processor = AutoProcessor.from_pretrained(config.model_name)

#### Load your annotated data
train_dataset = SexualizationClassifierDataset( \
    input_images=train_inputs, \
    output_images=train_outputs, \
    prompts=train_prompts, \
    labels=train_labels, \
    processor=processor \
)

#### Train
classifier = SexualizationClassifier(config)
trained_classifier = train_classifier(
    classifier, 
    train_dataset, 
    val_dataset, 
    config
)

### Train reward model

Use the classifier to construct preference tuples and train the reward model:

from data_augmentation import construct_preference_tuples
from reward_model import ImagePreferenceTupleDataset

#### Construct preference data
tuples = construct_preference_tuples(
    classifier=trained_classifier,
    images=all_generated_images,
    prompts=all_prompts,
    processor=processor,
    delta_threshold=0.3
)

### Fine-tuning

Prepare your data directory with this structure:

data_dir/ \
├── input/ \
│   ├── 0.jpg \
│   ├── 1.jpg \
│   └── ... \
├── chosen/      # Safer outputs \
│   ├── 0.jpg \
│   ├── 1.jpg \
│   └── ... \
├── rejected/    # Less safe outputs \
│   ├── 0.jpg \
│   ├── 1.jpg \
│   └── ... \
└── prompts.txt  # One prompt per line 

Run DPO Training
python dpo_finetune.py \
    --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --data_dir "./dpo_data" \
    --output_dir "./flux_dpo_aligned" \
    --resolution 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --mixed_precision fp16

Supervised Fine-tuning (SFT)
python sft_finetune.py \
    --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --csv_path "training_data.csv" \
    --output_dir "./flux_kontext_finetuned" \
    --resolution 512 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-5

CSV format for SFT \
input_prompt,input_image,output_image \
"make me a princess",path/to/input1.jpg,path/to/output1.jpg \
"put me on the beach",path/to/input2.jpg,path/to/output2.jpg \

Hardware Requirements
Minimum

GPU: 16GB VRAM (V100, T4)

RAM: 32GB

Storage: 100GB

Recommended

GPU: 40GB+ VRAM (A100)

RAM: 64GB+

Storage: 500GB
