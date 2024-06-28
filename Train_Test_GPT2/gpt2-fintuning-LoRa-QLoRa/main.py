from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import torch
import math
import bitsandbytes as bnb

def load_and_prepare_data():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset = dataset["train"].train_test_split(test_size=0.01)["train"]
    train_test_split = dataset.train_test_split(test_size=0.01)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    accuracy = np.mean(predictions == labels.numpy())

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss.item())

    return {
        "accuracy": accuracy,
        "perplexity": perplexity,
        "loss": loss.item()
    }

class LoRaLayer(torch.nn.Module):
    def __init__(self, weight, rank):
        super(LoRaLayer, self).__init__()
        self.rank = rank
        self.weight = weight
        if len(weight.size()) != 2:
            raise ValueError(f"Expected weight tensor of dimension 2, but got {len(weight.size())}")
        self.lora_a = torch.nn.Parameter(torch.zeros(weight.size(0), rank))
        self.lora_b = torch.nn.Parameter(torch.zeros(rank, weight.size(1)))
        torch.nn.init.kaiming_uniform_(self.lora_a)
        torch.nn.init.zeros_(self.lora_b)
    
    def forward(self, x):
        return torch.mm(x, self.weight + torch.mm(self.lora_a, self.lora_b))

class QLoRaLayer(torch.nn.Module):
    def __init__(self, weight, rank):
        super(QLoRaLayer, self).__init__()
        self.rank = rank
        self.weight = weight
        if len(weight.size()) != 2:
            raise ValueError(f"Expected weight tensor of dimension 2, but got {len(weight.size())}")
        self.lora_a = bnb.nn.Int8Params(torch.zeros(weight.size(0), rank), requires_grad=True)
        self.lora_b = bnb.nn.Int8Params(torch.zeros(rank, weight.size(1)), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.lora_a)
        torch.nn.init.zeros_(self.lora_b)
    
    def forward(self, x):
        return torch.mm(x, self.weight + torch.mm(self.lora_a.to(torch.float32), self.lora_b.to(torch.float32)))

def apply_lora(model, rank):
    to_modify = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            if len(param.size()) == 2:  # Ensure the weight is 2D
                to_modify.append((name, param))
            else:
                print(f"Skipping {name} as it does not have 2D weight.")

    for name, param in to_modify:
        lora_layer = LoRaLayer(param, rank)
        setattr(model, name.replace('.', '_'), lora_layer)
        param.requires_grad = False

def apply_q_lora(model, rank):
    to_modify = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            if len(param.size()) == 2:  # Ensure the weight is 2D
                to_modify.append((name, param))
            else:
                print(f"Skipping {name} as it does not have 2D weight.")

    for name, param in to_modify:
        q_lora_layer = QLoRaLayer(param, rank)
        setattr(model, name.replace('.', '_'), q_lora_layer)
        param.requires_grad = False

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.transformer.h[-1].parameters():
        param.requires_grad = True

    for param in model.lm_head.parameters():
        param.requires_grad = True

def train_and_evaluate_model(tokenized_train_dataset, tokenized_test_dataset, tokenizer, lora=False, q_lora=False):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if lora:
        apply_lora(model, rank=8)
    elif q_lora:
        apply_q_lora(model, rank=8)
    else:
        freeze_parameters(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=4,
        save_safetensors=False  # Disable strict checks for shared tensors
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    results = trainer.evaluate()
    print("Evaluation results:", results)

def main(lora=False, q_lora=False):
    tokenized_train_dataset, tokenized_test_dataset, tokenizer = load_and_prepare_data()
    train_and_evaluate_model(tokenized_train_dataset, tokenized_test_dataset, tokenizer, lora, q_lora)

# Set the desired fine-tuning method
lora = True  # Set to False if you do not want to use LoRa
q_lora = False  # Set to True if you want to use QLoRa

# Call the main function with the desired arguments
main(lora=lora, q_lora=q_lora)