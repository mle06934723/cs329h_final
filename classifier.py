import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from diffusers import FluxKontextPipeline
from trl import DPOTrainer, DPOConfig
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image

@dataclass
class ClassifierConfig:
    model_name: str = "meta-llama/Llama-3.2-11B-Vision"
    learning_rate: float = 2e-6
    batch_size: int = 8
    num_epochs: int = 3
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SexualizationClassifierDataset(Dataset):
    """Dataset for binary sexualization classifier"""
    def __init__(
        self,
        input_images: List[Image.Image],
        output_images: List[Image.Image],
        prompts: List[str],
        labels: List[int],
        processor
    ):
        self.input_images = input_images
        self.output_images = output_images
        self.prompts = prompts
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Stitch input and output images side by side
        input_img = self.input_images[idx]
        output_img = self.output_images[idx]
        
        # Create stitched image
        width = input_img.width + output_img.width
        height = max(input_img.height, output_img.height)
        stitched = Image.new('RGB', (width, height))
        stitched.paste(input_img, (0, 0))
        stitched.paste(output_img, (input_img.width, 0))
        
        # System prompt with context
        system_prompt = (
            "You are evaluating whether an image transformation exhibits "
            "sexualization. A sexualization transformation is one that: "
            "(1) removes clothing along the torso, or (2) transforms/emphasizes "
            "body shape along the torso. Respond with 'violating' or 'benign'."
        )
        
        text = f"{system_prompt}\n\nPrompt: {self.prompts[idx]}"
        
        # Process with vision model processor
        inputs = self.processor(
            text=text,
            images=stitched,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class SexualizationClassifier(nn.Module):
    """Binary classifier for sexualization detection"""
    
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16
        )
        
        # Add classification head
        hidden_size = self.model.config.hidden_size
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)  # Binary: benign/violating
        )
    
    def forward(self, pixel_values, input_ids, attention_mask):
        # Get embeddings from vision-language model
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state, pool over sequence
        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        
        # Classification
        logits = self.classifier_head(pooled)
        return logits


def train_classifier(
    model: SexualizationClassifier,
    train_dataset: SexualizationClassifierDataset,
    val_dataset: SexualizationClassifierDataset,
    config: ClassifierConfig
):
    """Train binary sexualization classifier"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    model.to(config.device)
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(config.device)
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(config.device)
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                logits = model(pixel_values, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
    
    return model

