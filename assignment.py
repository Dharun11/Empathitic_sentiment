import torch
import pandas as pd
import json
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
import os
from typing import List, Union

class EmotionTrainer:
    def __init__(self, 
                 data_path: str, 
                 model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the Emotion Prediction Trainer
        
        :param data_path: Path to CSV or JSON file containing the dataset
        :param model_id: Pretrained model to use as base
        """
        self.data_path = data_path
        self.model_id = model_id
    
    def _load_custom_dataset(self) -> tuple:
        """
        Load custom dataset from CSV or JSON and split into train and eval sets
        
        :return: Tuple of (train_dataset, eval_dataset)
        """
        # Determine file type
        file_extension = os.path.splitext(self.data_path)[1].lower()
        
        # Load data based on file type
        if file_extension == '.csv':
            df = pd.read_csv(self.data_path)
            # Assuming CSV has columns: system_content, user_content, emotion
            df['text'] = df.apply(self._format_conversation, axis=1)
        elif file_extension in ['.json', '.jsonl']:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df['text'] = df.apply(self._format_conversation, axis=1)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Split dataset into train and eval
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        eval_df = df.iloc[train_size:]
        
        # Convert to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df[['text']])
        eval_dataset = Dataset.from_pandas(eval_df[['text']])
        
        return train_dataset, eval_dataset
    
    def _format_conversation(self, row):
        """
        Format conversation into a single text string for training
        
        :param row: DataFrame row
        :return: Formatted conversation string
        """
        # For JSON with list of dict structure
        if isinstance(row, dict) and len(row) > 0:
            system_msg = row[0]['content'] if 'system' in row[0]['role'] else ""
            user_msg = row[1]['content'] if 'user' in row[1]['role'] else ""
            emotion = row[2]['content'] if 'assistant' in row[2]['role'] else ""
            
            return f"System: {system_msg}\nUser: {user_msg}\nEmotion: {emotion}"
        
        # For CSV or other formats
        return f"System: {row.get('system_content', '')}\nUser: {row.get('user_content', '')}\nEmotion: {row.get('emotion', '')}"
    
    def train_emotion_model(self):
        """
        Train the emotion prediction model
        """
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires a CUDA-compatible GPU.")
        
        # Print GPU information
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Load custom dataset with train and eval splits
        train_dataset, eval_dataset = self._load_custom_dataset()
        
        # Print dataset sizes
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Configure quantization for GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with CUDA configuration
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            quantization_config=bnb_config, 
            device_map={'': 0},  # Explicitly map to first GPU
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration for emotion prediction fine-tuning
        peft_config = LoraConfig(
            r=16,  # Rank of the update matrices
            lora_alpha=32,  # Scaling factor for the weight matrix
            lora_dropout=0.05,  # Dropout to add to the LoRA layers
            bias='none',  # Bias type
            task_type="CAUSAL_LM"  # Causal Language Modeling task
        )
        model = get_peft_model(model, peft_config)
        
        # Training arguments optimized for local GPU
        training_args = TrainingArguments(
            output_dir="emotion-prediction-model",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            optim="adamw_torch",
            logging_steps=20,
            learning_rate=2e-4,
            fp16=True,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            num_train_epochs=3,  # Increased epochs for better learning
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"
        )
        
        # Trainer setup
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # Add evaluation dataset
            dataset_text_field="text",
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args,
            packing=True,
            peft_config=peft_config,
        )
        
        # Start training
        trainer.train()
        
        # Save the model
        trainer.save_model('model_ft/emotion_prediction_model')
        
        return model, tokenizer

# Usage example
if __name__ == "__main__":
    # Replace with your actual file path
    data_path = "Notebook\Data.csv"  # or .json
    
    trainer = EmotionTrainer(data_path)
    model, tokenizer = trainer.train_emotion_model()
    
    # Optional: Test the model
    def predict_emotion(text):
        # Implement inference logic here
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=50)
        predicted_emotion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return predicted_emotion