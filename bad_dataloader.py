import json
import torch
import random
from transformers import LlamaTokenizer
import pdb

class BadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, text):
        self.encodings = encodings
        self.text = text
        # self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['text'] = self.text[idx]
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # return len(self.text)
        return len(self.encodings.input_ids)

def fit_to_prompt_template(messages, label, labels_dict):
    message = messages[-1]
    prompt = f'Classify the following message as either {labels_dict["ok"]} or {labels_dict["notok"]}.\nMessage: {message}\nLabel: {labels_dict[label]}'
    return(prompt)

def load_dataset(tokenizer, data_path='/home/andyliu/Miniconda3/envs/anlp-hw/lib/python3.10/site-packages/data/bot_adversarial_dialogue/dialogue_datasets/bot_adversarial_dialogue_datasets_with_persona/train.txt', sample_rate = 0.1):
    with open(data_path, 'r') as f:
        bad_data = f.read().split('\n')

    texts = []
    prompt_labels = {'notok':'offensive', 'ok':'inoffensive'}
    # labels = {'notok':1, 'ok':0}
    random.seed(0) 

    for line in bad_data:
        if line != '' and random.random() < sample_rate:
            text = line.split('\t')[0].replace('text:', '').split('\\n')
            label = line.split('\t')[1].replace('labels:', '').replace('_', '')
            prompt = fit_to_prompt_template(text, label, prompt_labels)
            texts.append(prompt)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    shuffled_indices = list(range(len(texts)))
    train_indices = shuffled_indices[:int(0.7*len(texts))]
    val_indices = shuffled_indices[int(0.7*len(texts)):int(0.85*len(texts))]
    test_indices = shuffled_indices[int(0.85*len(texts)):]

    # train_dataset = BadDataset([texts[i] for i in train_indices])
    # val_dataset = BadDataset([texts[i] for i in val_indices])
    # test_dataset = BadDataset([texts[i] for i in test_indices])

    train_encodings = tokenizer([texts[i] for i in train_indices], truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer([texts[i] for i in val_indices], truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer([texts[i] for i in test_indices], truncation=True, padding=True, max_length=128)
    train_dataset = BadDataset(train_encodings, [texts[i] for i in train_indices])
    val_dataset = BadDataset(val_encodings, [texts[i] for i in val_indices])
    test_dataset = BadDataset(test_encodings, [texts[i] for i in test_indices])

    return(train_dataset, val_dataset, test_dataset)