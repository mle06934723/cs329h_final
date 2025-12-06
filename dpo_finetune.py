import argparse
import os
import math
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm


check_min_version("0.31.0")
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training for FLUX")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing input/, prompts.txt, chosen/, rejected/ subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./flux_dpo_aligned",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for training images",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Learning rate (DPO typically uses lower LR than SFT)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature parameter (controls strength of preference)",
    )
    parser.add_argument(
        "--reference_model_path",
        type=str,
        default=None,
        help="Path to reference model (if None, uses pretrained model)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for LR scheduler",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory efficient attention",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Logging directory",
    )
    
    return parser.parse_args()


class DPODataset(Dataset):
    """Dataset for DPO alignment"""
    
    def __init__(
        self,
        input_images: List[Image.Image],
        prompts: List[str],
        chosen_images: List[Image.Image],
        rejected_images: List[Image.Image],
        resolution: int = 512
    ):
        self.input_images = input_images
        self.prompts = prompts
        self.chosen_images = chosen_images
        self.rejected_images = rejected_images
        self.resolution = resolution
    
    def __len__(self):
        return len(self.prompts)
    
    def _process_image(self, img):
        """Process image to tensor"""
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0
        return tensor
    
    def __getitem__(self, idx):
        return {
            'input_image': self._process_image(self.input_images[idx]),
            'prompt': self.prompts[idx],
            'chosen': self._process_image(self.chosen_images[idx]),
            'rejected': self._process_image(self.rejected_images[idx])
        }


def load_dataset_from_dir(data_dir: str, resolution: int = 512):
    """
    Load DPO dataset from directory structure:
    data_dir/
        input/
            0.jpg, 1.jpg, ...
        prompts.txt (one prompt per line)
        chosen/
            0.jpg, 1.jpg, ...
        rejected/
            0.jpg, 1.jpg, ...
    """
    data_dir = Path(data_dir)
    
    # Load prompts
    with open(data_dir / "prompts.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    
    # Load images
    input_images = []
    chosen_images = []
    rejected_images = []
    
    for i in range(len(prompts)):
        # Try common image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            input_path = data_dir / "input" / f"{i}{ext}"
            if input_path.exists():
                input_images.append(Image.open(input_path).convert('RGB'))
                break
        
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            chosen_path = data_dir / "chosen" / f"{i}{ext}"
            if chosen_path.exists():
                chosen_images.append(Image.open(chosen_path).convert('RGB'))
                break
        
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            rejected_path = data_dir / "rejected" / f"{i}{ext}"
            if rejected_path.exists():
                rejected_images.append(Image.open(rejected_path).convert('RGB'))
                break
    
    assert len(input_images) == len(prompts), "Mismatch in number of input images and prompts"
    assert len(chosen_images) == len(prompts), "Mismatch in number of chosen images and prompts"
    assert len(rejected_images) == len(prompts), "Mismatch in number of rejected images and prompts"
    
    return DPODataset(input_images, prompts, chosen_images, rejected_images, resolution)


def collate_fn(examples):
    """Collate function for dataloader"""
    prompts = [ex['prompt'] for ex in examples]
    input_images = torch.stack([ex['input_image'] for ex in examples])
    chosen_images = torch.stack([ex['chosen'] for ex in examples])
    rejected_images = torch.stack([ex['rejected'] for ex in examples])
    
    return {
        'prompts': prompts,
        'input_images': input_images,
        'chosen_images': chosen_images,
        'rejected_images': rejected_images,
    }


def compute_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    ref_chosen_logprobs,
    ref_rejected_logprobs,
    beta=0.1
):
    """
    Compute DPO loss
    
    DPO Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    where log_ratio = model_logprob - ref_logprob
    """
    # Compute log ratios
    chosen_log_ratios = model_chosen_logprobs - ref_chosen_logprobs
    rejected_log_ratios = model_rejected_logprobs - ref_rejected_logprobs
    
    # DPO loss
    logits = beta * (chosen_log_ratios - rejected_log_ratios)
    loss = -F.logsigmoid(logits).mean()
    
    # Compute implicit reward (for logging)
    with torch.no_grad():
        implicit_reward_chosen = beta * chosen_log_ratios
        implicit_reward_rejected = beta * rejected_log_ratios
    
    return loss, implicit_reward_chosen.mean(), implicit_reward_rejected.mean()


def compute_log_prob(noise_pred, target, timesteps):
    """
    Compute log probability of predictions
    This is a simplified version - in practice you may want to use the full likelihood
    """
    # MSE loss as negative log likelihood (assuming Gaussian)
    mse = F.mse_loss(noise_pred, target, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])  # Average over spatial dims and channels
    
    # Convert to log prob (negative of MSE for Gaussian assumption)
    log_prob = -mse
    return log_prob


def main():
    args = parse_args()
    
    # Setup logging
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
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load policy model (trainable)
    logger.info("Loading policy model...")
    policy_pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    )
    
    # Load reference model (frozen)
    logger.info("Loading reference model...")
    ref_model_path = args.reference_model_path or args.pretrained_model_name_or_path
    ref_pipeline = FluxPipeline.from_pretrained(
        ref_model_path,
        torch_dtype=torch.float16,
    )
    
    # Extract components
    vae = policy_pipeline.vae
    text_encoder = policy_pipeline.text_encoder
    text_encoder_2 = policy_pipeline.text_encoder_2
    tokenizer = policy_pipeline.tokenizer
    tokenizer_2 = policy_pipeline.tokenizer_2
    policy_transformer = policy_pipeline.transformer
    ref_transformer = ref_pipeline.transformer
    
    # Freeze everything except policy transformer
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    ref_transformer.requires_grad_(False)
    
    # Enable gradient checkpointing for policy model
    if args.gradient_checkpointing:
        policy_transformer.enable_gradient_checkpointing()
    
    # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        try:
            policy_transformer.enable_xformers_memory_efficient_attention()
            ref_transformer.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    # Move models to device
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)
    text_encoder_2.to(accelerator.device, dtype=torch.float16)
    ref_transformer.to(accelerator.device, dtype=torch.float16)
    
    # Setup optimizer (only for policy model)
    optimizer = torch.optim.AdamW(
        policy_transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    train_dataset = load_dataset_from_dir(args.data_dir, args.resolution)
    
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
    
    # Setup scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    
    # Prepare with accelerator
    policy_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy_transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training info
    logger.info("***** Running DPO training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Beta (DPO temperature) = {args.beta}")
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(args.num_train_epochs):
        policy_transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(policy_transformer):
                # Encode text prompts
                with torch.no_grad():
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
                    
                    # Encode all images to latent space
                    chosen_latents = vae.encode(
                        batch['chosen_images'].to(accelerator.device, dtype=torch.float16)
                    ).latent_dist.sample() * vae.config.scaling_factor
                    
                    rejected_latents = vae.encode(
                        batch['rejected_images'].to(accelerator.device, dtype=torch.float16)
                    ).latent_dist.sample() * vae.config.scaling_factor
                
                # Sample noise and timesteps (same for chosen and rejected)
                noise_chosen = torch.randn_like(chosen_latents)
                noise_rejected = torch.randn_like(rejected_latents)
                
                bsz = chosen_latents.shape[0]
                timesteps = torch.rand(bsz, device=chosen_latents.device)
                
                # Add noise
                noisy_chosen = (1 - timesteps.view(-1, 1, 1, 1)) * chosen_latents + timesteps.view(-1, 1, 1, 1) * noise_chosen
                noisy_rejected = (1 - timesteps.view(-1, 1, 1, 1)) * rejected_latents + timesteps.view(-1, 1, 1, 1) * noise_rejected
                
                # Get policy model predictions
                policy_pred_chosen = policy_transformer(
                    hidden_states=noisy_chosen,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                
                policy_pred_rejected = policy_transformer(
                    hidden_states=noisy_rejected,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                
                # Get reference model predictions (no grad)
                with torch.no_grad():
                    ref_pred_chosen = ref_transformer(
                        hidden_states=noisy_chosen,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    
                    ref_pred_rejected = ref_transformer(
                        hidden_states=noisy_rejected,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                
                # Compute targets (velocity for flow matching)
                target_chosen = noise_chosen - chosen_latents
                target_rejected = noise_rejected - rejected_latents
                
                # Compute log probabilities
                policy_logprob_chosen = compute_log_prob(policy_pred_chosen, target_chosen, timesteps)
                policy_logprob_rejected = compute_log_prob(policy_pred_rejected, target_rejected, timesteps)
                ref_logprob_chosen = compute_log_prob(ref_pred_chosen, target_chosen, timesteps)
                ref_logprob_rejected = compute_log_prob(ref_pred_rejected, target_rejected, timesteps)
                
                # Compute DPO loss
                loss, reward_chosen, reward_rejected = compute_dpo_loss(
                    policy_logprob_chosen,
                    policy_logprob_rejected,
                    ref_logprob_chosen,
                    ref_logprob_rejected,
                    beta=args.beta
                )
                
                # Backprop
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy_transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
            
            logs = {
                "loss": loss.detach().item(),
                "reward_chosen": reward_chosen.item(),
                "reward_rejected": reward_rejected.item(),
                "reward_margin": (reward_chosen - reward_rejected).item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        policy_transformer = accelerator.unwrap_model(policy_transformer)
        policy_pipeline.transformer = policy_transformer
        policy_pipeline.save_pretrained(args.output_dir)
        logger.info(f"DPO training complete! Model saved to {args.output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()