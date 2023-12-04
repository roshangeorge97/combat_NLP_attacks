from transformers import (
    GPTQConfig, 
    AutoModelForCausalLM, 
    LlamaTokenizer, 
    TrainingArguments, 
    Trainer, 
    default_data_collator, 
    get_linear_schedule_with_warmup, 
    DataCollatorWithPadding,
)
from trl import SFTTrainer
from bad_dataloader import load_dataset
from torch.utils.data import DataLoader
import torch
from peft import (get_peft_config, 
    get_peft_model, 
    get_peft_model_state_dict,
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training,
)
import evaluate
import argparse
import wandb
import numpy as np

from datetime import date
from tqdm import tqdm
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path', type=str, default='/data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model-path', type=str, default='/data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--data-path', type=str, default='bad_data.json')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--track', action = 'store_true')
    parser.add_argument('--wandb-project', type=str, default='finetune_classifier')
    parser.add_argument('--exp-name', type=str, default='andy1107')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--sample-rate', type=float, default=0.1)
    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    args = parse_args()
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # load data
    train_data, val_data, test_data = load_dataset(tokenizer, sample_rate = args.sample_rate)

    # set up model for PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        # target_modules=["k_proj","o_proj","q_proj","v_proj","lm_head.weight"],
        bias='none',
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1)

    if 'quantized' not in args.model_path and args.quantize:
        gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=gptq_config, device_map="auto")
        short_model_name = args.model_path.split('/')[-1]
        model.save_pretrained(f"/data/tir/projects/tir3/users/andyliu/{short_model_name}_quantized")
    elif args.quantize:
        gptq_config = GPTQConfig(bits=4, disable_exllama=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=gptq_config, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    if args.track:
        wandb.init(project = args.wandb_project, config=vars(args), name=args.exp_name, save_code=True)

    # path to save model
    short_model_name = args.model_path.split('/')[-1]
    peft_model_id = f"/data/tir/projects/tir3/users/andyliu/{short_model_name}_finetuned_{args.exp_name}"

    # train and eval
    training_args = TrainingArguments(
        output_dir = peft_model_id,
        learning_rate = args.lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs = args.epochs,
        # weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        bf16=args.bf16,
        bf16_full_eval=args.bf16,
        load_best_model_at_end=True,
        push_to_hub = False,
        report_to = "wandb",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field='text',
        peft_config=peft_config,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )
    trainer.train()
    trainer.save_model(peft_model_id)
