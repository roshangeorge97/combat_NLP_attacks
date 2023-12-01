from transformers import (
    GPTQConfig, 
    LlamaForSequenceClassification, 
    LlamaTokenizer, 
    TrainingArguments, 
    Trainer, 
    default_data_collator, 
    get_linear_schedule_with_warmup, 
    DataCollatorWithPadding,
)
from datasets import load_dataset
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
    parser.add_argument('--train-test-split', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--track', action = 'store_true')
    parser.add_argument('--wandb-project', type=str, default='finetune_classifier')
    parser.add_argument('--exp-name', type=str, default='andy1107')

    args = parser.parse_args()
    return(args)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    return(tokenizer(examples['text'], padding='max_length', truncation=True, max_length = 4096))

if __name__ == "__main__":
    args = parse_args()
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load data
    data = load_dataset('json', data_files=args.data_path, field='train')
    f1 = evaluate.load('f1')

    tokenized_data = data['train'].map(preprocess_function, batched=True)
    tokenized_data = tokenized_data.train_test_split(test_size = args.train_test_split)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    id2label = {0:'inoffensive', 1:'offensive'}
    label2id = {'inoffensive':0, 'offensive':1}
    
    # set up model for PEFT
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        target_modules=["k_proj","o_proj","q_proj","v_proj"],
        bias='none',
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1)

    if 'quantized' not in args.model_path:
        gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
        model = LlamaForSequenceClassification.from_pretrained(args.model_path, num_labels=2, id2label=id2label, label2id=label2id, quantization_config=gptq_config, device_map="auto")
        short_model_name = args.model_path.split('/')[-1]
        model.save_pretrained(f"/data/tir/projects/tir3/users/andyliu/{short_model_name}_quantized")
    else:
        model = LlamaForSequenceClassification.from_pretrained(args.model_path, num_labels=2, id2label=id2label, label2id=label2id, device_map="auto")
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = model.config.eos_token_id

    if args.track:
        wandb.init(project = args.wandb_project, config=vars(args), name=args.exp_name, save_code=True)

    # path to save model
    short_model_name = args.model_path.split('/')[-1]
    peft_model_id = f"/data/tir/projects/tir3/users/andyliu/{short_model_name}_finetuned_{date.today()}"

    # train and eval
    training_args = TrainingArguments(
        output_dir = peft_model_id,
        learning_rate = args.lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs = args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub = False,
        report_to = "wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
