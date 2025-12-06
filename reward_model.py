from collections import abc
import torch 
from typing import Callable, Iterator, Optional, List, Tuple, Any, Dict 
import json 
from PIL import Image
from torchvision import transforms  
from transformers import AutoTokenizer 
import os 
import torch.nn.functional as F 

import torch
from accelerate import Accelerator

from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration

from trl import (
    SFTTrainer
)

def apply_multimodal_image_preference_loss(
    pred: torch.Tensor,
    eos_indices: torch.Tensor, 
    polarities: torch.Tensor,
    inverse_permutation: List[int],
) -> Dict[str, torch.Tensor]:
    batch_size = pred.shape[0]
    assert batch_size % 2 == 0 
    reward_weight = 0.9
    margin = 0.5 

    loss_info = {}
    scalar_pred = pred[..., 0].cuda() # logit 
    logits = torch.gather(pred, 1, eos_indices.unsqueeze(-1)).reshape(-1).cuda()
    embeddings = pred[:, :-1, :].cuda() 

    # exploit interweaving pattern
    first_inds = inverse_permutation[0::2]
    second_inds = inverse_permutation[1::2]


    first_embs = embeddings[first_inds]
    first_logits = logits[first_inds]
    second_embs = embeddings[first_inds]
    second_logits = logits[second_inds]

    polarity_chosen = polarities[first_inds]
    polarity_rejected = polarities[second_inds]
    assert torch.all(polarity_chosen == polarity_rejected)
    polarity = polarity_chosen 

    flip_reward_mask = torch.where(polarity == 0, torch.tensor(1), torch.tensor(-1))
    reward_weight_mask = torch.where(polarity == 0, reward_weight, 1 - reward_weight)
    contrastive_weight_mask = torch.where(polarity == 0, 1 - reward_weight, reward_weight)

    first_embs_mean = first_embs.mean(dim=1)
    second_embs_mean = second_embs.mean(dim=1)
    distances = 1 - F.cosine_similarity(first_embs_mean, second_embs_mean, dim=1, eps=1e-6)
    distances = distances.unsqueeze(1)
    margin = torch.full_like(distances, margin)
    cl_loss = 0.5 * (
        polarity.float().unsqueeze(1) * distances.pow(2)
        + (1 - polarity).float().unsqueeze(1) * F.relu(margin - distances).pow(2)
    )
    cl_weighted = cl_loss * contrastive_weight_mask.unsqueeze(1)

    second_logits = second_logits * flip_reward_mask 
    rm_loss = -F.logsigmoid(first_logits - second_logits)
    rm_weighted = rm_loss * reward_weight_mask 

    loss = torch.mean(cl_loss + rm_loss)
    return {
        "loss": loss 
    }

class ImagePreferenceDatum:
    chosen_img_ar: Tuple[torch.Tensor, torch.Tensor]
    rejected_img_ar: Tuple[torch.Tensor, torch.Tensor]
    conversation = List[Dict[str, Any]]
    row_id: int
    dataloader_worker_id: int 


class ImagePreferenceTupleDataset:
    def __init__(
        self, 
        json_file: str, 
        transforms: Callable[[Image.Image], torch.Tensor], 
        world_rank: int, 
        world_size: int, 
        batch_size: int, 
        tokenizer: AutoTokenizer
    ) -> None: 
        self.annotations = self._preprocess_data(json_file)
        self.img_root = "/tmp/"
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.CenterCrop(224),     
            transforms.ToTensor(),          
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
    
    def _preprocess_data(self, json_file) -> None: 
        processed_data = [] 
        with open(json_file, "r") as f:
            data = json.load(f)
            for sample in data: 
                new_sample = {
                    "chosen_img": sample["chosen"],
                    "rejected_img": sample["rejected"],
                    "conversation": sample["conversation"]
                }
                processed_data.append(new_sample)
        
    def _process_image(
        self, image_path: str
    ) -> torch.Tensor:
        image_path = os.path.join(self.image_root, image_path)
        with open(image_path, "rb") as fp:
            image = Image.open(fp)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self._transforms(image)
            return image 
    
    def __next__(self) -> ImagePreferenceDatum: 
        worker_info = torch.utils.data.get_worker_info()
        dataloader_worker_id = 0 if worker_info is None else worker_info.id
        offset_index = self.idxes[self._current_position]
        data_sample = self._annotations[offset_index]
        return ImagePreferenceDatum(
            chosen_img=self._process_image(data_sample["chosen"]),
            rejected_img=self._process_image(data_sample["rejected"]),
            conversation=data_sample["ConversationSample"],
            dataloader_worker_id=dataloader_worker_id,
            row_id=offset_index
        )


    def __iter__(self) -> Iterator[ImagePreferenceDatum]:
        self._current_position = 0
        return self 


class ImagePreferenceTupleCollater(abc.ABC):
    def __init__(
        self, 
        tokenizer, 
        seq_len= 1024, 
        max_num_chunks=4,
        max_images=1, 
        shuffle_images_in_batch=True,
    ):
        self._tokenizer = tokenizer 
        self.seq_len = seq_len 
        self.max_num_chunks = max_num_chunks 
        self.max_images = max_images 
        self.shuffle_images_in_batch = shuffle_images_in_batch 
    

    def get_eos_indices(self, captions: torch.Tensor) -> torch.Tensor: 
        caption_list = captions.tolist()
        eos_indices = []
        for caption in caption_list:
            T = len(caption)
            try: 
                i = T - caption[::-1].index(self._tokenizer.eos_id) - 1
            except ValueError:
                i = T - 1
            eos_indices.append(i - 1)
        eos_indices = captions.new_tensor(eos_indices)
        return eos_indices 

    def __call__(self, samples: List[ImagePreferenceDatum]) -> ImagePreferenceInterleavedBatch: 
        images = [] 
        all_num_chunks = [] 
        for datum in samples: 
            chosen_image = datum.chosen_img 
            rejected_image = datum.rejected_img 
            images.append(chosen_image)
            images.append(rejected_image)
            num_chunks = min(self.max_num_chunks, chosen_image.shape[0])
            all_num_chunks.append(num_chunks)
        images = torch.stack(images)

        processed_samples = [datum.conversation for datum in samples]
        captions, x_attn_masks, labels = self._tokenizer.tokenize_batch_train(
            samples=processed_samples, 
            num_chunks=all_num_chunks, 
            max_num_media=self.max_images,
            max_text_length=self.max_words,
            max_num_chunks=self.max_num_chunks
        )

        captions = captions.repeat(2, 1)
        x_attn_masks = x_attn_masks.repeat(2, 1)
        eos_indices = self.get_eos_indices(captions).repeat(2, 1)
        row_ids = torch.tensor([s.row_id for s in samples for _  in range (2)])
        dataloader_worker_ids = torch.tensor([s.dataloader_worker_id for s in samples for _ in range(2)])

        if self.shuffle_images_in_batch:
            permutation = torch.randperm(len(images))
        else:
            permutation = torch.arange(len(images))
        
        inverse_permutation = torch.argsort(permutation)
        permutation = permutation.tolist()
        inverse_permutation = inverse_permutation.tolist() 

        return ImagePreferenceInterleavedBatch(
            images = images[permutation].contiguous(),
            captions=captions[permutation].contiguous(),
            x_attn_masks=x_attn_masks[permutation].contiguous(),
            eos_indices=eos_indices[permutation].contiguous(),
            row_ids=row_ids[permutation].contiguous(),
            dataloader_worker_ids=dataloader_worker_ids[permutation].contiguous(),
            permutation=permutation,
            inverse_permutation=inverse_permutation
        )

class ImagePreferenceInterleavedBatch: 
    images: torch.Tensor
    captions: torch.Tensor
    x_attn_masks: torch.Tensor
    eos_indices: torch.Tensor
    row_ids: torch.Tensor
    aspect_ratio: torch.Tensor 
    permutation: Optional[List[int]] = None 
    inverse_permutation: Optional[List[int]] = None


def __main__():
    """Train reward model with hybrid contrastive and reward loss"""
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        num_train_epochs=1,
        logging_dir='./logs',
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        fp16=True,  
        bf16=False, 
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="epoch",
        save_total_limit=10,
        logging_steps=100,
        per_device_eval_batch_size=16,
        # load_best_model_at_end=True,
    )

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    ckpt_path = "/tmp/meta-llama/Llama-3.2-11B-Vision-Instruct/consolidated.ckpt"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    dataset_train = ImagePreferenceTupleDataset("tmp/train.json")
    dataset_valid = ImagePreferenceTupleDataset("tmp/valid.json")

    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        data_collator=ImagePreferenceTupleCollater(processor.tokenizer),
        train_dataset=dataset_train, 
        eval_dataset=dataset_valid,
        compute_loss=apply_multimodal_image_preference_loss,
        tokenizer=processor.tokenizer, 
    )
    trainer.train()