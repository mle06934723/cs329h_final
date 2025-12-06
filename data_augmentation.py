from .reward_model import RewardModel 
from .classifier import SexualizationClassifier
from typing import List, Tuple 
from PIL import Image
from torch.nn.functional import nn.functional as F 
import torch 
from diffusers import FluxKontextPipeline

def construct_preference_tuples(
    classifier: SexualizationClassifier,
    images: List[Image.Image],
    prompts: List[str],
    processor,
    delta_threshold: float = 0.3
) -> List[Tuple[Image.Image, Image.Image, str, int, int]]:
    """
    Construct preference tuples from classifier scores.
    Returns: (image_a, image_b, prompt, label_a, label_b) tuples
    """
    classifier.eval()
    
    # Group by prompt
    prompt_to_images = {}
    for img, prompt in zip(images, prompts):
        if prompt not in prompt_to_images:
            prompt_to_images[prompt] = []
        prompt_to_images[prompt].append(img)
    
    tuples = []
    
    with torch.no_grad():
        for prompt, imgs in prompt_to_images.items():
            if len(imgs) < 2:
                continue
            
            # Score all images for this prompt
            scores = []
            for img in imgs:
                # Create dummy input (no input image for classifier)
                inputs = processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt"
                )
                
                logits = classifier(
                    inputs['pixel_values'].to(classifier.config.device),
                    inputs['input_ids'].to(classifier.config.device),
                    inputs['attention_mask'].to(classifier.config.device)
                )
                
                # Get probability of violating class
                probs = F.softmax(logits, dim=-1)
                score = probs[0, 1].item()  # Violating class probability
                scores.append(score)
            
            # Create tuples with sufficient delta
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    delta = abs(scores[i] - scores[j])
                    if delta >= delta_threshold:
                        # Determine which is safer (lower violation score)
                        if scores[i] < scores[j]:
                            safer_idx, less_safe_idx = i, j
                            label_a, label_b = 0, 1  # benign, violating
                        else:
                            safer_idx, less_safe_idx = j, i
                            label_a, label_b = 1, 0
                        
                        tuples.append((
                            imgs[safer_idx],
                            imgs[less_safe_idx],
                            prompt,
                            label_a,
                            label_b
                        ))
    
    return tuples



def generate_with_prompt_augmentation(
    pipeline: FluxKontextPipeline,
    input_image: Image.Image,
    base_prompt: str,
    num_inference_steps: int = 50
) -> List[Image.Image]:
    """Generate multiple outputs with prompt augmentations"""
    
    outputs = []
    augmentations = {
        "attn_subj": "; drawing attention to the input subject",
        "studio_lighting": "; hyperrealistic, sharp focus, detailed, lighting",
        "mesmerizing": "; mesmerizing, aesthetic, pleasing",
        "cinema": "; cinematic lighting, ultra high definition",
        "shadows": "; enhancing shadows, Photoshop curves adjustment, highlights to create depth and photorealism",
        "glow": "; making the input subject glow",
        "sharpen_details": "; sharpening details, clarity enhancement, noise reduction",
        "improve_exposure": "; adjusting exposure, correcting brightness, optimizing dynamic range",
        "enhance_colors": "; color grading, saturation boost, vibrancy enhancement",
        "reduce_noise": "; noise reduction, smoothing out grain, detail preservation",
        "increase_depth": "; depth enhancement, layering, dimensionality",
        "refine_edges": "; edge refinement, crispness, definition",
        "balance_white": "; white balance correction, neutral tone, accurate color representation",
        "boost_contrast": "; contrast adjustment, nuanced shading, visual interest",
        "smooth_skin": "; skin smoothing, blemish removal, portrait retouching",
        "add_luster": "; luster enhancement, subtle sheen, eye-catching",
        "detailed_texture": "; intricate details, textured, realistic",
        "atmospheric_perspective": "; atmospheric perspective, depth cueing, hazy",
        "high_contrast": "; high contrast, dramatic, chiaroscuro",
        "crispify": "; crispening edges, defining textures, precision detailing",
        "natural_tones": "; natural color palette, authentic tones, subtle nuance",
        "even_lighting": "; uniform lighting, balanced illumination, reduced harsh shadows",
        "refined_shading": "; refined shading, nuanced gradations, dimensional accuracy",
        "precision_focus": "; precise focus, tack-sharp details, razor-thin depth of field",
        "subtle_gradients": "; subtle gradient transitions, smooth blending, nuanced color shifts",
        "deepen_blacks": "; deepening blacks, rich shadows, increased contrast ratio",
        "lift_highlights": "; highlight recovery, lifted details, preserved texture",
        "soothe_colors": "; soothing color palette, muted tones, calming atmosphere",
        "clarity_boost": "; clarity enhancement, improved definition, enhanced visibility",
        "micro_detail": "; micro-detail enhancement, revealing hidden textures, minute particulars",
        "film_grain_reduction": "; film grain reduction, smoothed out noise, clean presentation",
        "local_contrast": "; local contrast adjustment, accentuated details, enhanced dimensionality",
        "inner_radiance": "; inner radiance, confidence, and poise, shining through",
        "radiant_complexion": "; radiant complexion, healthy glow, subtle warmth",
        "soft_skin": "; smooth, flawless skin, subtle softening",
        "defined_features": "; defined facial features, subtle sculpting, natural contours",
        "youthful_appearance": "; youthful appearance, refreshed, revitalized",
        "eye_catchers": "; eye-catching highlights on jewelry, accessories, or clothing, added sparkle",
        "overall_sheen": "; overall sheen, subtle glow, healthy appearance",
        "texture_and_detail": "; detailed textures, visible fibers, tactile quality",
        "highlighted_fabrics": "; highlighted fabrics, subtle sheen, luxurious appearance",
        "clothing_drape": "; realistic clothing drape, folds, and creases, natural fit",
        "polished_finish": "; polished finish, refined appearance, high-end quality",
        "dimensional_quality": "; dimensional quality, layered, and nuanced, visually interesting",
        "visual_flow": "; guiding the viewer's eye, visual flow, compositionally balanced",
        "tactile_quality": "; tactile quality, inviting touch, sensory experience",
        "atmospheric_mood": "; atmospheric mood, evocative, and emotive, setting the tone",
        "refined_atmosphere": "; refined atmosphere, sophisticated, and elegant, understated luxury",
        "immersive_experience": "; immersive experience, engaging, and interactive, drawing the viewer in",
        "nuanced_lighting": "; nuanced lighting, subtle variations, and interplay, creating depth",
        "rich_textures": "; rich textures, detailed, and varied, adding depth and interest"
    }
    for k, aug in augmentations:
        augmented_prompt = f"{base_prompt}; {aug}"
        output = pipeline(
            prompt=augmented_prompt,
            image=input_image,
            num_inference_steps=num_inference_steps
        ).images[0]
        outputs.append(output)
    
    return outputs


def rank_with_reward_model(
    reward_model: RewardModel,
    images: List[Image.Image],
    prompt: str,
    processor
) -> List[Tuple[Image.Image, float]]:
    """Rank images by reward model score"""
    
    reward_model.eval()
    scored_images = []
    
    with torch.no_grad():
        for img in images:
            inputs = processor(
                text=prompt,
                images=img,
                return_tensors="pt"
            )
            
            logits, _ = reward_model(
                inputs['pixel_values'].to(reward_model.config.device),
                inputs['input_ids'].to(reward_model.config.device),
                inputs['attention_mask'].to(reward_model.config.device)
            )
            
            scored_images.append((img, logits.item()))
    
    # Sort by score (lowest = most violating = rejected)
    scored_images.sort(key=lambda x: x[1])
    return scored_images