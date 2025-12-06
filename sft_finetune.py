import argparse
import os
import math
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import diffusers
from diffusers import FluxKontextPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FLUX Kontext model")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="training_data.csv",
        help="Path to CSV file with columns: input_prompt, input_image, output_image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./flux_kontext_finetuned",
        help="The output directory where the model will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images",
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=10
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=100, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision training",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    
    args = parser.parse_args()
    return args


class FluxKontextDataset(Dataset):
    def __init__(self, csv_path, resolution=512):
        self.df = pd.read_csv(csv_path)
        self.resolution = resolution
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load images
        input_img = Image.open(row['input_image']).convert('RGB')
        output_img = Image.open(row['output_image']).convert('RGB')
        
        # Resize
        input_img = input_img.resize((self.resolution, self.resolution), Image.LANCZOS)
        output_img = output_img.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Convert to tensors and normalize to [-1, 1]
        input_tensor = torch.from_numpy(np.array(input_img)).float().permute(2, 0, 1) / 127.5 - 1.0
        output_tensor = torch.from_numpy(np.array(output_img)).float().permute(2, 0, 1) / 127.5 - 1.0
        
        return {
            'prompt': row['input_prompt'],
            'input_image': input_tensor,
            'target_image': output_tensor,
        }


def collate_fn(examples):
    prompts = [example['prompt'] for example in examples]
    input_images = torch.stack([example['input_image'] for example in examples])
    target_images = torch.stack([example['target_image'] for example in examples])
    
    return {
        'prompts': prompts,
        'input_images': input_images,
        'target_images': target_images,
    }


def main():
    args = parse_args()
    
    # Setup logging and accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models and schedulers
    logger.info("Loading FLUX pipeline...")
    pipeline = FluxKontextPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    )
    
    # Extract components
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    transformer = pipeline.transformer
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    
    # Freeze VAE and text encoders
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        try:
            transformer.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    # Move models to device
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)
    text_encoder_2.to(accelerator.device, dtype=torch.float16)
    
    # Setup optimizer (following Diffusers examples pattern)
    params_to_optimize = transformer.parameters()
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Setup dataset and dataloader
    train_dataset = FluxKontextDataset(
        csv_path=args.csv_path,
        resolution=args.resolution
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    
    # Prepare with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is not None:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    # Training loop (following official Diffusers example pattern)
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Encode text prompts
                with torch.no_grad():
                    # Tokenize and encode
                    text_inputs = tokenizer(
                        batch['prompts'],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    text_inputs_2 = tokenizer_2(
                        batch['prompts'],
                        padding="max_length",
                        max_length=tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    prompt_embeds = text_encoder(
                        text_inputs.input_ids,
                        output_hidden_states=True,
                    ).hidden_states[-2]
                    
                    pooled_prompt_embeds = text_encoder_2(
                        text_inputs_2.input_ids,
                        output_hidden_states=True,
                    )[0]
                    
                    # Encode images to latent space
                    input_latents = vae.encode(
                        batch['input_images'].to(accelerator.device, dtype=torch.float16)
                    ).latent_dist.sample()
                    target_latents = vae.encode(
                        batch['target_images'].to(accelerator.device, dtype=torch.float16)
                    ).latent_dist.sample()
                    
                    input_latents = input_latents * vae.config.scaling_factor
                    target_latents = target_latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.rand(bsz, device=target_latents.device)
                
                # Add noise according to flow matching
                noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * target_latents + timesteps.view(-1, 1, 1, 1) * noise
                
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                
                # Flow matching loss (predict velocity)
                target = noise - target_latents
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Backpropagate
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
    
    # Save the final trained pipeline
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        pipeline.transformer = transformer
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"Saved model to {args.output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()