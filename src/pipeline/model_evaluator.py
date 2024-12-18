
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logging
from src.utils import generate_report
class LLMResponseEvaluator:
    def __init__(self, 
                 model_path: str, 
                 tokenizer_path: str, 
                 test_data_path: str):
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.test_data_path = test_data_path
        
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
       
        self.test_data = pd.read_csv(test_data_path)
        
    def preprocess_text(self, text: str) -> str:
        
        logging.info("Getting into preprocess data..........")
        text = text.lower().strip()
        return text
    
    def predict_emotion(self, text: str) -> str:
        
        
        cleaned_text = self.preprocess_text(text)
        logging.info("Getting into predict_emotion ..........")
        
        inputs = self.tokenizer(
            cleaned_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        )
        
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)
        
       
        emotion_labels = self.test_data['emotion_label'].unique()
        predicted_emotion = emotion_labels[predicted_class.item()]
        logging.info("Successfully predicted results..........")
        return predicted_emotion
    
    def evaluate_model(self) -> Dict[str, Any]:
        
        
        true_labels = []
        predicted_labels = []
        logging.info("Getting into evaluate  data..........")
       
        for _, row in self.test_data.iterrows():
           
            full_text = f"{row['situation_text']} {row['user_queries']} {row['agent_responses']}"
            
            
            predicted_emotion = self.predict_emotion(full_text)
            true_emotion = row['emotion_label']
            
            true_labels.append(true_emotion)
            predicted_labels.append(predicted_emotion)
        
       
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        
        cm = confusion_matrix(true_labels, predicted_labels, 
                               labels=list(np.unique(true_labels)))
        logging.info("Sucessfully evaluated the data..........")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def visualize_confusion_matrix(self, cm: np.ndarray, labels: List[str]):
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix of Emotion Prediction')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def generate_classification_report(self) -> str:
        

        metrics = self.evaluate_model()
        

        unique_labels = list(np.unique(self.test_data['emotion_label']))
        self.visualize_confusion_matrix(metrics['confusion_matrix'], unique_labels)
        

        report = f"""
        Model Evaluation Report
        ======================
        Accuracy: {metrics['accuracy']:.4f}
        Precision: {metrics['precision']:.4f}
        Recall: {metrics['recall']:.4f}
        F1 Score: {metrics['f1_score']:.4f}
        
        Confusion Matrix:
        - Visualized and saved as 'confusion_matrix.png'
        
        Interpretation Notes:
        1. Accuracy: Overall correct predictions
        2. Precision: Ratio of correct positive predictions
        3. Recall: Proportion of actual positives correctly identified
        4. F1 Score: Harmonic mean of precision and recall
        
        """
        generate_report(report)
        return report
    
    def error_analysis(self) -> Dict[str, List[str]]:
        
        misclassified = {
            'true_label': [],
            'predicted_label': [],
            'text': []
        }
        
        for _, row in self.test_data.iterrows():
            full_text = f"{row['situation_text']} {row['user_queries']} {row['agent_responses']}"
            predicted_emotion = self.predict_emotion(full_text)
            true_emotion = row['emotion_label']
            
            if predicted_emotion != true_emotion:
                misclassified['true_label'].append(true_emotion)
                misclassified['predicted_label'].append(predicted_emotion)
                misclassified['text'].append(full_text)
        
        
        return misclassified

def main():

    model_path = 'model_ft/emotion_prediction_model'
    tokenizer_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    test_data_path = 'Data/test.csv'
    
    
    evaluator = LLMResponseEvaluator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        test_data_path=test_data_path
    )
    
    try:

        report = evaluator.generate_classification_report()
        print(report)
        

        misclassified = evaluator.error_analysis()
        print("\nMisclassified Examples:")
        for true, pred, text in zip(
            misclassified['true_label'], 
            misclassified['predicted_label'], 
            misclassified['text']
        ):
            print(f"True: {true}, Predicted: {pred}")
            print(f"Text: {text[:200]}...\n")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__=='__main__':
    main()