# Empathetic Sentiment Analysis with TinyLlama 1.1B

## Overview

This project focuses on performing **empathetic sentiment analysis** using the **TinyLlama-1.1B** language model. The goal is to predict emotions in text based on empathetic dialogues and enhance the model's understanding of different emotional tones.

The model has been fine-tuned using the **Kaggle Empathetic Dialogues dataset** and leverages performance-efficient fine-tuning (PEFT) techniques to handle training on a limited GPU resource. The final model can predict emotion labels and generate responses based on input prompts.


### Key Features:
- **Data Preprocessing**: Includes handling of missing values, text standardization, and feature engineering to improve the dataset's quality.
- **Model Training**: Fine-tuning the TinyLlama-1.1B language model with the **BitsAndBytes** configuration for memory efficiency and **PEFT** (Performance Efficient Fine-Tuning) for reducing computational overhead while training specific layers of the model.
- **Evaluation**: Includes model testing and evaluation using metrics like accuracy, precision, recall, and F1-score.
- **Inference**: Interactive chat-based inference mode allowing you to input prompts and get emotion-labeled responses.

## Dependencies

This project requires the following Python libraries:

- `torch`
- `transformers`
- `datasets`
- `logging`
- `re`
- `argparse`
  
To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt


