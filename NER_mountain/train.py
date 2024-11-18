import torch
import numpy as np

from seqeval.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset, DatasetDict


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(dataset_name, use_subset=False, subset_size=5):
    dataset = load_dataset(dataset_name)
    
    # use subset of data for checking of training
    if use_subset:
        train_subset = dataset['train'].select(range(subset_size))
        test_subset = dataset['test'].select(range(subset_size))
        return DatasetDict({'train': train_subset, 'test': test_subset})
    
    return dataset


def prepare_model_and_tokenizer(model_name, num_labels, device):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def create_tokenize_function(tokenizer):
    def tokenize_and_align_labels(examples):
        
        """
        Tokenizes input data and aligns labels with tokenized outputs.
        """
        
        # tokenize the input tokens
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True, 
            is_split_into_words=True #keep the word structure
        )
        
        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            # align labels with tokenized tokens
            for word_idx in word_ids:
                if word_idx is None: 
                    # special tokens
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # new word
                    label_ids.append(label[word_idx])
                else: 
                    # subword tokens
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            # add aligned labels for current example
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    return tokenize_and_align_labels


def create_metrics_function(label_list):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=2)

        true_labels = [[label_list[l] for l in label if l != -100] 
                        for label in labels]
        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]

        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions)
        }
    return compute_metrics
    

def setup_training_args(output_dir="./mountain-ner"):
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        no_cuda=False,
        warmup_steps=500,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        run_name="bert fine-tuning"
    )
    
def train_model(model,
                tokenizer, 
                training_args, 
                dataset,  
                data_collator, 
                compute_metrics):
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    train_results = trainer.train()
    eval_results = trainer.evaluate()
    return train_results, eval_results

def save_model(model, tokenizer, output_dir="./mountain-ner-final"):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def push_to_hub():
    from huggingface_hub import login
    login()
    
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id="Gepe55o/mountain-ner-bert-base", exist_ok=True)
    api.upload_folder(
        folder_path="/model",
        repo_id="Gepe55o/mountain-ner-bert-base",
        repo_type="model"
    )

def main():
    
    device = setup_device()
    dataset = load_data("Gepe55o/mountain-ner-dataset", use_subset=True)
    label_list = ['O', 'B-LOC']
    
    # prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(
        "dslim/bert-base-NER", 
        len(label_list), 
        device
    )
    
    # data preprocessing
    tokenize_and_align_labels = create_tokenize_function(tokenizer)
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # start training
    training_args = setup_training_args()
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    compute_metrics = create_metrics_function(label_list)
    train_results, eval_results = train_model(
        model, 
        tokenizer,
        training_args,
        tokenized_dataset,
        data_collator,
        compute_metrics
    )
    
    print(f"Training metrics: {train_results.metrics}")
    print(f"Evaluation metrics: {eval_results}")
    
    save_model(model, tokenizer)

if __name__ == "__main__":
    main()
