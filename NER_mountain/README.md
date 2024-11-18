# Mountain name Entity Recognition
This project focuses on fine-tuning [bert-base-ner](https://huggingface.co/dslim/bert-base-NER) on custom dataset to recognize mountain names in text.

## Dataset
 - Source data collected from [NERetrive](https://arxiv.org/pdf/2310.14282) and [Few-NERD](https://arxiv.org/pdf/2105.07464v6) datasets
 - Filtered for mountain-related entities
 - Converted to binary classification (mountain/non-mountain)
 
The [dataset](https://huggingface.co/datasets/Gepe55o/mountain-ner-dataset) contains tokenized text with corresponding NER tags where:
- `tag 1`: mountain entity 
- `tag 0`: not a mountain name

### Dataset Structure
The dataset contains two main columns:
- `tokens`: list of tokenized words
- `tags`: corresponding NER tags (0 or 1)

### Example:
```python
{
    'tokens': ['The', 'Everest', 'is', 'the', 'highest', 'peak'],
    'tags': [0, 1, 0, 0, 0, 0]
}
```

### Usage:
```python
from datasets import load_dataset

dataset = load_dataset("Gepe55o/mountain-ner-dataset")

train_data = dataset["train"]
test_data = dataset["test"]
```

## Model 

[mountain-ner-bert-base](https://huggingface.co/Gepe55o/mountain-ner-bert-base) is a fine-tuned model based on the bert-base-ner architecture for mountain names Entity Recognition tasks. The model is trained on the merging of two datasets: [NERetrieve](https://arxiv.org/pdf/2310.14282), [Few-NERD](https://arxiv.org/pdf/2105.07464v6), [Mountain-ner-dataset](https://huggingface.co/datasets/Gepe55o/mountain-ner-dataset). The model is trained to recognize two types of entities: `LABEL_0` (other), `LABEL_1` (mountain names).

- Model Architecture: bert-base-ner
- Task: mountain names entity recognition
- Training Data: [mountain-ner-dataset](https://huggingface.co/datasets/Gepe55o/mountain-ner-dataset)

### Performance
Metrics: 
| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall   | F1       |
|-------|---------------|----------------|----------|-----------|----------|----------|
| 1     | 0.027400      | 0.030793       | 0.988144 | 0.815692  | 0.924621 | 0.866748 |
| 2     | 0.020600      | 0.024568       | 0.991119 | 0.872988  | 0.921036 | 0.896369 |
| 3     | 0.012900      | 0.024072       | 0.991923 | 0.889878  | 0.920171 | 0.904771 |


Best model performance achieved at epoch 3 with:
- F1 Score: 0.9048
- Accuracy: 0.9919
- Precision: 0.8899
- Recall: 0.9202

### How to use
```python
from transformers import AutoModel, AutoTokenizer, pipeline

model = AutoModel.from_pretrained("Gepe55o/mountain-ner-bert-base")
tokenizer = AutoTokenizer.from_pretrained("Gepe55o/mountain-ner-bert-base")

text = "Mount Everest is the highest mountain in the world."

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
result = nlp(text)
```

### Hyperparameters
- Learning rate: 5e-5
- Batch size: 32
- Epochs: 3
- Weight decay: 0.01

## Project Structure
- **datasets/** - Training and testing datasets 
- **create_dataset.ipynb** - Create and preprocess the dataset
- **demo.ipynb** - Interactive demo notebook
- **inference.py** - Inference and visualization - - - functions
- **train.py** - Training script for the NER model
