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

class LLMResponseEvaluator:
    def __init__(self, 
                 model_path: str, 
                 tokenizer_path: str, 
                 test_data_path: str):
        """
        Initialize the LLM Response Evaluator
        
        :param model_path: Path to the trained model
        :param tokenizer_path: Path to the tokenizer
        :param test_data_path: Path to the test dataset
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.test_data_path = test_data_path
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load test data
        self.test_data = pd.read_csv(test_data_path)
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text
        
        :param text: Input text
        :return: Cleaned text
        """
        # Basic text cleaning
        
        text = text.lower().strip()
        return text
    
    def predict_emotion(self, text: str) -> str:
        """
        Predict emotion for given text
        
        :param text: Input text
        :return: Predicted emotion
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize input
        inputs = self.tokenizer(
            cleaned_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)
        
        # Map back to original label
        emotion_labels = self.test_data['emotion_label'].unique()
        predicted_emotion = emotion_labels[predicted_class.item()]
        
        return predicted_emotion
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        :return: Dictionary of evaluation metrics
        """
        # Prepare ground truth and predictions
        true_labels = []
        predicted_labels = []
        
        # Iterate through test data
        for _, row in self.test_data.iterrows():
            # Use full text for prediction
            full_text = f"{row['situation_text']} {row['user_queries']} {row['agent_responses']}"
            
            # Predict emotion
            predicted_emotion = self.predict_emotion(full_text)
            true_emotion = row['emotion_label']
            
            true_labels.append(true_emotion)
            predicted_labels.append(predicted_emotion)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, 
                               labels=list(np.unique(true_labels)))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def visualize_confusion_matrix(self, cm: np.ndarray, labels: List[str]):
        """
        Visualize confusion matrix
        
        :param cm: Confusion matrix
        :param labels: Emotion labels
        """
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
        """
        Generate detailed classification report
        
        :return: Formatted classification report
        """
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Visualize confusion matrix
        unique_labels = list(np.unique(self.test_data['emotion_label']))
        self.visualize_confusion_matrix(metrics['confusion_matrix'], unique_labels)
        
        # Compile report
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
        
        return report
    
    def error_analysis(self) -> Dict[str, List[str]]:
        """
        Perform error analysis
        
        :return: Dictionary of misclassified examples
        """
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
    # Paths - adjust these to match your project structure
    model_path = 'model_ft/emotion_prediction_model'
    tokenizer_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    test_data_path = 'Data/test.csv'
    
    # Initialize evaluator
    evaluator = LLMResponseEvaluator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        test_data_path=test_data_path
    )
    
    try:
        # Generate classification report
        report = evaluator.generate_classification_report()
        print(report)
        
        # Perform error analysis
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