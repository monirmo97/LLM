from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import torch
import math

def load_and_prepare_data():
    # Load and split the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_test_split = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add padding token to the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define a function to tokenize the text
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Apply the tokenizer to the datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    return tokenized_train_dataset, tokenized_test_dataset, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    
    # Calculate perplexity
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss)
    
    return {
        "accuracy": accuracy,
        "perplexity": perplexity,
        "loss": loss.item()
    }

def train_and_evaluate_model(tokenized_train_dataset, tokenized_test_dataset, tokenizer):
    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",           # Directory to save the model
        eval_strategy="epoch",            # Evaluate the model after each epoch
        save_strategy="epoch",            # Save the model after each epoch
        learning_rate=5e-5,               # Learning rate for training
        per_device_train_batch_size=4,    # Batch size for training
        per_device_eval_batch_size=4,     # Batch size for evaluation
        num_train_epochs=1,               # Number of training epochs
        weight_decay=0.01,                # Weight decay for regularization
        logging_dir='./logs',             # Directory for storing logs
        logging_steps=10,                 # Log every 'logging_steps'
        save_total_limit=2,               # Limit the total amount of checkpoints
        load_best_model_at_end=True,      # Load the best model at the end of training
        metric_for_best_model="loss",     # The metric to use to compare models
        greater_is_better=False           # Whether a greater metric is better
    )
    
    # Use DataCollatorWithPadding to handle dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,                          # The model to train
        args=training_args,                   # Training arguments
        train_dataset=tokenized_train_dataset,  # Training dataset
        eval_dataset=tokenized_test_dataset,  # Evaluation dataset
        compute_metrics=compute_metrics,      # Custom metrics
        data_collator=data_collator           # Dynamic padding
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    results = trainer.evaluate()
    print("Evaluation results:", results)

def main():
    tokenized_train_dataset, tokenized_test_dataset, tokenizer = load_and_prepare_data()
    train_and_evaluate_model(tokenized_train_dataset, tokenized_test_dataset, tokenizer)

if __name__ == "__main__":
    main()
