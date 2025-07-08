
import logging
import os
from contextlib import nullcontext
from torch.utils.data import Dataset
import argparse
from PIL import Image
import json
import torch
import pandas as pd


TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
import matplotlib.pyplot as plt
tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)



class CustomDataset(Dataset):
    def __init__(self, stage, dataset_desc_easy, dataset_desc_hard, dataset_pope, image_path, batch_size):
        with open(dataset_desc_easy, 'r') as f:
            self.data_desc_easy = json.load(f)
        
        with open(dataset_desc_hard, 'r') as f:
            self.data_desc_hard = json.load(f)
        
        with open(dataset_pope, 'r') as f:
            self.data_pope = json.load(f)
        
        self.image_path = image_path
        
        if stage == "1":
            total_data = self.data_desc_easy
        elif stage == "2":
            total_data = self.data_desc_hard
        elif stage == "3":
            total_data = self.data_pope
        else:
            raise ValueError("Invalid stage value. Stage must be 1, 2, or 3.")
        
        total_len = len(total_data)
        print(f"Length before:{total_len}")
        print(f"batch_size:{batch_size}")
        # batch_size=4
        adjusted_len = (total_len // 4) * 4
        self.data = total_data[:adjusted_len]
        print(f"Length after:{len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.data[idx]
        messages = entry['messages']
        image_path = os.path.join(self.image_path, entry['images'])  # Construct full image path
        scores = entry['scores']

        # Process the image
        image = self.convert_image_to_jpeg(image_path)

        sample = {
            'messages': messages,
            'images': image,
            'scores': scores
        }
        return sample
    
    def convert_image_to_jpeg(self, image_path):  # Add 'self' as the first parameter
        # Load the image
        image = Image.open(image_path)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to JPEG format in memory
        jpeg_image = Image.new("RGB", image.size)
        jpeg_image.paste(image)
        
        return jpeg_image
    
class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):

        num_messages = len(examples[0]["messages"])
        for example in examples:
            assert len(example["messages"]) == num_messages, "Inconsistent number of messages in a batch"

        texts = []
        images = []
        scores = []

        for example in examples:
            image = example["images"]  # This is the single image for the example
            for i, messages in enumerate(example["messages"]):
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
                images.append(image)  # Append the same image for each message
                scores.append(example["scores"][i])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        # Flatten the batch dimensions for input_ids, attention_mask, labels, and scores
        batch_size = len(examples)
        num_messages = len(examples[0]["messages"])

        batch["input_ids"] = batch["input_ids"].view(batch_size * num_messages, -1)
        batch["attention_mask"] = batch["attention_mask"].view(batch_size * num_messages, -1)
        batch["labels"] = batch["labels"].view(batch_size * num_messages, -1)
        batch["scores"] = torch.tensor(scores).view(1,batch_size * num_messages)
        batch["pixel_values"] = batch["pixel_values"].view(batch_size * num_messages, 3, 336, 336)
        batch["scores"] = batch["scores"].detach()
        return batch


class RRHFTrainer(SFTTrainer):

    def __init__(self, rrhf_weight, margin_weight, gamma, *args, **kwargs):
        self.rrhf_weight = float(rrhf_weight)
        self.margin_weight = float(margin_weight)
        self.gamma = float(gamma)

        super().__init__(*args, **kwargs)

        self.per_device_train_batch_size = self.args.per_device_train_batch_size
        

    def gather_logits_labels(self, logits, labels):
        mask = (labels != -100).float()
        new_logits = logits.clone()
        new_labels = labels.clone()
        new_labels[labels == -100] = 0  
        output = torch.gather(new_logits, dim=-1, index=new_labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask  
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)  
        scores = logit_label.sum(-1) / (length ** 1.0)
        return scores
    
    def sft_loss(self, logit_label,rw_scores):
        max_idx = torch.argmax(rw_scores)
        return -logit_label[max_idx].mean()

    def rrhf_loss(self, scores, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1)  
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1)
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
        return -diff[aval].sum() / len(scores)

    def distance_loss(self, scores, labels):
        labels = labels + 1
        labels = labels / labels.sum()
        loss = torch.abs(torch.exp(scores) - labels).sum()
        return loss

    def margin_loss(self, scores, labels, gamma):
        max_label = torch.max(labels)
        loss = torch.tensor(0.0, dtype=scores.dtype, device=scores.device)  
        labels = labels.squeeze() 
        for i in range(labels.size(0)):
            if labels[i] != max_label:
                diff = max_label - labels[i]
                loss += ((scores[labels == max_label] - scores[i] - diff * gamma) ** 2).sum()
        return loss / len(scores)


    def compute_loss(self, model, inputs, return_outputs=False):

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params / 1e9:.2f}B")
        print(f"Trainable parameters: {trainable_params / 1e9:.2f}B")
        print(f"Trainable parameters ratio: {trainable_params / total_params:.4%}")

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        labels = inputs.get("labels")
        original_scores = inputs.get("scores")
        
        input_ids_tensors = torch.chunk(input_ids, self.per_device_train_batch_size)
        attention_mask_tensors = torch.chunk(attention_mask, self.per_device_train_batch_size)
        pixel_values_tensors = torch.chunk(pixel_values, self.per_device_train_batch_size)
        labels_tensors = torch.chunk(labels, self.per_device_train_batch_size)
        original_scores_transposed = original_scores.transpose(0, 1)
        original_scores_chunks = torch.chunk(original_scores_transposed, self.per_device_train_batch_size)
        original_scores_tensors = [chunk.transpose(0, 1) for chunk in original_scores_chunks]

        sft_losses = []
        rrhf_losses = []
        margin_losses = []
        total_losses = []

        for i in range(len(input_ids_tensors)):
            inner_input_ids = input_ids_tensors[i]
            inner_attention_mask = attention_mask_tensors[i]
            inner_pixel_values = pixel_values_tensors[i]
            inner_labels = labels_tensors[i]
            inner_original_scores = original_scores_tensors[i]

            outputs = model(input_ids=inner_input_ids, attention_mask=inner_attention_mask, pixel_values=inner_pixel_values, labels=inner_labels)
            logits = outputs.get("logits")
            import torch.nn.functional as F
            probabilities = F.log_softmax(logits, dim=-1)
            logit_label = self.gather_logits_labels(probabilities, inner_labels)
            scores = self.get_score(logit_label, inner_labels)


            sft_loss = self.sft_loss(logit_label, inner_original_scores)
            rrhf_loss = self.rrhf_loss(scores, inner_original_scores) * self.rrhf_weight
            margin_loss = self.margin_loss(scores, inner_original_scores, self.gamma) * self.margin_weight
            total_loss = rrhf_loss + sft_loss + margin_loss

            sft_losses.append(sft_loss)
            rrhf_losses.append(rrhf_loss)
            margin_losses.append(margin_loss)
            total_losses.append(total_loss)

            print(f"scores: {scores}")

            # Delete intermediate variables and free memory
            del inner_input_ids, inner_attention_mask, inner_pixel_values, inner_labels, inner_original_scores, logits, probabilities, logit_label, scores, sft_loss, rrhf_loss, margin_loss, total_loss
            # torch.cuda.empty_cache()

        avg_sft_loss = torch.mean(torch.stack(sft_losses))
        avg_rrhf_loss = torch.mean(torch.stack(rrhf_losses))
        avg_margin_loss = torch.mean(torch.stack(margin_losses))
        avg_total_loss = torch.mean(torch.stack(total_losses))

        self.log({
            "RRHFTrainer_sft_loss": avg_sft_loss.item(),
            "RRHFTrainer_rrhf_loss": avg_rrhf_loss.item(),
            "RRHFTrainer_margin_loss": avg_margin_loss.item(),
        })
        
        if return_outputs:
            outputs["sft_loss"] = avg_sft_loss
            outputs["rrhf_loss"] = avg_rrhf_loss
            outputs["margin_loss"] = avg_margin_loss
            outputs["total_loss"] = avg_total_loss
            return (avg_total_loss, outputs)
        else:
            return avg_total_loss



if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    parser.add_argument('--lora_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--rrhf_weight', type=str)
    parser.add_argument('--stage', type=str)
    parser.add_argument('--dataset_desc_easy', type=str)
    parser.add_argument('--dataset_desc_hard', type=str)
    parser.add_argument('--dataset_pope', type=str)
    parser.add_argument('--plot_path', type=str)
    parser.add_argument('--margin_weight', type=str)
    parser.add_argument('--plot_loss', type=str)
    parser.add_argument('--gamma', type=str)
    sft_script_args, training_args, model_config, extra_config = parser.parse_args_and_config()

    # print(f"sft_script_args:{sft_script_args}")
    # print(f"training_args:{training_args}")
    # print(f"model_config:{model_config}")
    # print(f"extra_config:{extra_config}")


    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()



    ################
    # Model, Tokenizer & Processor
    ################
    LLAVA_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, padding_side="left")
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)
    processor.tokenizer = tokenizer

    print(f"model_kwargs:{model_kwargs}")
    model = LlavaForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    # print(f"model:\n{model}")

    ######################
    ## merge lora weight
    ######################
    from peft import PeftModel
    if extra_config.lora_path != "None":
        print("merge lora model!")
        lora_model = PeftModel.from_pretrained(model, 
                                                model_id=extra_config.lora_path,
                                                **model_kwargs)
        # merge
        model = lora_model.merge_and_unload()
        print(f"merge successful:{extra_config.lora_path}!")




    ################
    # Create a data collator to encode text and image pairs
    ################


    data_collator = LLavaDataCollator(processor)

    ################
    # Dataset
    ################
    train_dataset = CustomDataset(
        extra_config.stage,
        extra_config.dataset_desc_easy,
        extra_config.dataset_desc_hard, 
        extra_config.dataset_pope, 
        extra_config.image_path,
        training_args.per_device_train_batch_size
    )
    
    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )


    ###################
    ### model parms ###
    ###################
    total_params = sum(p.numel() for p in model.parameters())

    # Freeze the visual encoder and MLP layers
    # for name, param in model.named_parameters():
    #     if name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
    #         param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params / 1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params / 1e9:.2f}B")
    print(f"Trainable parameters ratio: {trainable_params / total_params:.4%}")


    ################
    # Training
    ################
    with init_context:
        trainer = RRHFTrainer(
            rrhf_weight=extra_config.rrhf_weight,
            margin_weight=extra_config.margin_weight,
            gamma=extra_config.gamma,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            dataset_text_field="text",  # need a dummy field
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=data_collator,
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        print(f"get_peft_config(model_config):{get_peft_config(model_config)}")
    import time
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"train time:{(end_time - start_time)/3600}h")


    #################
    ### plot loss ###
    #################
    if extra_config.plot_loss:
        from plot_cgq import plot_training_curves
        plot_training_curves(trainer.state.log_history,extra_config.plot_path)



    ##################
    ### save model ###
    ##################
    with save_context:
        trainer.save_model(training_args.output_dir)
        # trainer.push_to_hub()
        # if Accelerator().is_main_process:
        #     processor.push_to_hub(training_args.hub_model_id)
