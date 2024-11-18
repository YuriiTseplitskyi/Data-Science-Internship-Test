# Quantum Test Task
This project contains two main components: a Mountain Names Entity Recognition (NER) system and an Image Matching Tool for satellite imagery. Below is an overview of each component and the project structure

## Mountain Names Entity Recognition
This module focuses on fine-tuning a BERT-based model to recognize mountain names in text. More info [here](NER_mountain/README.md). 

## Sentinel-2 Image Matching Tool
This module implements a tool for keypoint detection, matching, and visualization for Level-2A images from Sentinel-2. More info [here](Sentinel-2_image_matching/README.md).

## Project Structure
```bash
Quantum_Test_Task/
├── NER_mountain/ 
│   ├── datasets/              # Dataset versions    
│   ├── dataset_creation.iypnb # Notebook for creating dataset        
│   ├── demo.ipynb             # Interactive demo  
│   ├── inference.py           # Utility
│   └── train.py               # Training script
├── Sentinel-2_image_matching/ 
│   ├── data/                  # Sample images
│   ├── algo.py                # Matcher class
│   ├── demo.ipynb             # Interactive demo
│   └── inference.py           # Utility                  
└── requirements.txt            
```

## Requirements
To install dependencies for both modules, run:

```bash
pip install -r requirements.txt
```