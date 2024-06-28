from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import torch
import math

def load_and_prepare_data():
    # Load and split the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # Use a smaller subset of the dataset
    dataset = dataset["train"].train_test_split(test_size=0.01)["train"]
    # Further split for training and testing
    train_test_split = dataset.train_test_split(test_size=0.01)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Define a function to tokenize the text with reduced max_length
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)  # Reduce max_length
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Apply the tokenizer to the datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Convert logits and labels to PyTorch tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Calculate accuracy
    accuracy = np.mean(predictions == labels.numpy())

    # Calculate perplexity
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


def freeze_parameters(model):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final layer(s) or specific layers you want to fine-tune
    for param in model.transformer.h[-1].parameters():  # unfreezing the last layer
        param.requires_grad = True

    for param in model.lm_head.parameters():  # unfreezing the language modeling head
        param.requires_grad = True

def train_and_evaluate_model(tokenized_train_dataset, tokenized_test_dataset, tokenizer):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    freeze_parameters(model)  # Freeze specific parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,    # Reduce batch size
        per_device_eval_batch_size=1,     # Reduce batch size for evaluation
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=4  # Accumulate gradients over 4 steps
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

def main():
    tokenized_train_dataset, tokenized_test_dataset, tokenizer = load_and_prepare_data()
    train_and_evaluate_model(tokenized_train_dataset, tokenized_test_dataset, tokenizer)

if __name__ == "__main__":
    main()
