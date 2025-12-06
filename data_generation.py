import pandas as pd
from diffusers import FluxKontextPipeline
import os
from tqdm import tqdm
import random
from typing import Tuple, List 


def load_input_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Load input images and prompts from CSV file
    
    Expected CSV format:
    input_image,input_prompt
    path/to/image1.jpg,"make me a princess"
    path/to/image2.png,"put me on the beach"
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['input_image', 'input_prompt']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    
    input_images = df['input_image'].tolist()
    input_prompts = df['input_prompt'].tolist()
    
    return input_images, input_prompts


def load_images_from_paths(image_paths: List[str]) -> List[Image.Image]:
    """Load PIL Images from file paths"""
    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            images.append(img)
        else:
            print(f"Warning: Image not found at {path}")
            # Create placeholder image
            images.append(Image.new('RGB', (512, 512), color='gray'))
    return images


def generate_training_data(
    csv_path: str,
    output_dir: str,
    pipeline_name: str = "black-forest-labs/FLUX.1-dev",
    num_variations: int = 3,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42
):
    """
    Generate synthetic training data using FLUX.1 Kontext pipeline
    
    Args:
        csv_path: Path to CSV with input_image and input_prompt columns
        output_dir: Directory to save generated images
        pipeline_name: HuggingFace model name for FLUX pipeline
        num_variations: Number of variations to generate per input
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'inputs'), exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    image_paths, prompts = load_input_data(csv_path)
    input_images = load_images_from_paths(image_paths)
    
    print(f"Loaded {len(input_images)} input images and prompts")
    
    # Load pipeline
    print(f"Loading FLUX pipeline: {pipeline_name}...")
    pipeline = FluxKontextPipeline.from_pretrained(
        pipeline_name,
        torch_dtype=torch.float16
    )
    pipeline.to("cuda")
    
    # Set seed for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Generate data
    results = []
    
    for idx, (input_img, prompt) in enumerate(tqdm(zip(input_images, prompts), 
                                                     total=len(input_images),
                                                     desc="Generating outputs")):
        
        # Save input image
        input_path = os.path.join(output_dir, 'inputs', f'input_{idx:04d}.png')
        input_img.save(input_path)
        
        # Generate multiple variations
        for var_idx in range(num_variations):
            try:
                # Generate output
                output = pipeline(
                    prompt=prompt,
                    image=input_img,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
                
                # Save output image
                output_path = os.path.join(
                    output_dir, 
                    'outputs', 
                    f'output_{idx:04d}_var{var_idx}.png'
                )
                output.save(output_path)
                
                # Record result
                results.append({
                    'input_image': input_path,
                    'output_image': output_path,
                    'prompt': prompt,
                    'variation': var_idx,
                    'label': None  # To be filled by annotators
                })
                
            except Exception as e:
                print(f"Error generating image {idx} variation {var_idx}: {e}")
                continue
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, 'generated_data.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Saved {len(results)} generated examples to {results_csv}")
    
    return results_df


def generate_prompt_variations(
    base_prompt: str,
    num_variations: int = 5,
    augmentations: dict = None
) -> List[str]:
    """
    Generate prompt variations by adding augmentations
    
    Args:
        base_prompt: Original prompt
        num_variations: Number of variations to generate
        augmentations: Dict of augmentation keys and texts
    
    Returns:
        List of augmented prompts
    """
    if augmentations is None:
        augmentations = PROMPT_AUGMENTATIONS
    
    variations = [base_prompt]  # Include original
    
    # Randomly sample augmentations
    aug_keys = random.sample(list(augmentations.keys()), 
                            min(num_variations - 1, len(augmentations)))
    
    for key in aug_keys:
        aug_text = augmentations[key]
        variations.append(f"{base_prompt}, {aug_text}")
    
    return variations


def create_example_csv(output_path: str = "example_inputs.csv"):
    """
    Create an example CSV file with sample data
    """
    example_data = {
        'input_image': [
            'images/woman_portrait_1.jpg',
            'images/woman_portrait_2.jpg',
            'images/woman_beach.jpg',
            'images/woman_formal.jpg',
            'images/woman_casual.jpg',
        ],
        'input_prompt': [
            'make me a princess',
            'put me on the beach',
            'change my outfit to evening wear',
            'make me look like a superhero',
            'transform me into a mermaid',
        ]
    }
    
    df = pd.DataFrame(example_data)
    df.to_csv(output_path, index=False)
    print(f"Created example CSV at {output_path}")
    return output_path