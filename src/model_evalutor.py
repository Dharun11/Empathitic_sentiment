from src.logger import logging
from src.exception import CustomException
from evaluate import load
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.components.data_ingestion import Ingect_data

class ModelEvaluator:
    def __init__(self,eval_dataset,tokenizer,model):
        self.model=model
        self.tokenizer=tokenizer
        self.eval_dataset=eval_dataset
        
    def extract_label(self,text):
        try:
            for line in text.split('\n'):
                if "Emotion Label:" in line:
                    return line.split("Emotion Label:")[1].strip()
        except Exception as e:
            raise CustomException(e,sys)  # Handle cases where the label is not found
        return None
    
    
    def evaluate(self):
        accuracy_metric=load("accuracy")
        predictions = []
        references = []

        # Iterate through the evaluation dataset
        for sample in tqdm(self.eval_dataset):
            input_text = sample["text"]
            true_label = self.extract_labels(input_text)
            
            if true_label is None:
                continue  # Skip samples with no valid label
            
            # Tokenize input text
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Generate predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=-1)

            predictions.append(pred.cpu().numpy()[0])
            references.append(true_label)

        # Compute accuracy
        accuracy = accuracy_metric.compute(predictions=predictions, references=references)
        print(f"Accuracy: {accuracy['accuracy'] * 100:.2f}%")
        return accuracy
    
if __name__ == "__main__":
    # Load the tokenizer and fine-tuned model
    
    data_path="Data\Data.csv"
    data_ingection=Ingect_data(data_path)
    
    train_dataset,eval_dataset=data_ingection.load_custom_data()
    model_path = "model_ft/emotion_prediction_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Instantiate and evaluate the model
    evaluator = ModelEvaluator(model, tokenizer, eval_dataset)
    accuracy = evaluator.evaluate()