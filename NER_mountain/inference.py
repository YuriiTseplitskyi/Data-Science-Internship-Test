import string
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict


def predict(model, tokenizer, text: str) -> List[Dict]:
    
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp(text)

def parse(predictions: List[Dict], text: str) -> str:
    """
    Parse the output of a mountain NER model to mountain names with <mount> tags.

    :param predictions: List of dictionaries containing model predictions.
    :param text: Original input text.
    :return: Modified text with mountain names wrapped in <mount> tags.
    """
    modified_text = text
    mountain_ranges = []

    # combine subword tokens into full words and track mountain entities
    current_mountain = ""
    start_index = -1
    for prediction in predictions:
        word = prediction['word']
        entity = prediction['entity']
        
        if word.startswith("##"):  # subword continuation
            current_mountain += word[2:]
        else:
            if current_mountain:
                if start_index != -1:
                    mountain_ranges.append((start_index, current_mountain))
                current_mountain = ""
            if entity == "LABEL_1":  # start of a new mountain name
                current_mountain = word
                start_index = prediction['start']
            else:
                start_index = -1  # reset start index 

    # last mountain 
    if current_mountain and start_index != -1:
        mountain_ranges.append((start_index, current_mountain))

    mountain_ranges = sorted(mountain_ranges, key=lambda x: x[0], reverse=True)

    for start_idx, mountain_name in mountain_ranges:
        end_idx = start_idx + len(mountain_name)
        modified_text = (
            modified_text[:start_idx]
            + f"<mount>{mountain_name}</mount>"
            + modified_text[end_idx:]
        )

    return modified_text

def visualize(text: str):
    """
    Visualizes the parsed output with words inside <mount></mount> tags in light blue boxes
    """
    words = text.split()
    fig, ax = plt.subplots(figsize=(len(words) * 1.5, 2))
    
    x = 0  
    y = 0.5  
    
    for word in words:
        if word.startswith("<mount>") and word.endswith("</mount>"):

            display_word = word[7:-8]
            box_color = "lightblue"
        else:
            display_word = word
            box_color = "white"
        
        # rectangle around the word
        text_width = len(display_word) * 0.2  
        rect = patches.Rectangle((x, y - 0.25), text_width, 0.5, linewidth=1, edgecolor="black", facecolor=box_color)
        ax.add_patch(rect)
        
        # word text inside the box
        ax.text(x + text_width / 2, y, display_word, color="black", ha="center", va="center", fontsize=12)
        
        x += text_width + 0.2

    ax.set_xlim(0, x)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    plt.show()