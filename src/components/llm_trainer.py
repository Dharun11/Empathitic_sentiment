import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
import os
from typing import List, Union
from src.logger import logging
from src.exception import CustomException
import torch
import sys



class ModelTrainer:
    def __init__(self,train_dataset,eval_dataset,model_id:str="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_id=model_id
        self.train_dataset=train_dataset
        self.eval_dataset=eval_dataset
        
    def train_model(self):
        
        logging.info("Entering train model")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires a CUDA-compatible GPU.")
        
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        try:
            print(f"Training dataset size: {len(self.train_dataset)}")
            print(f"Evaluation dataset size: {len(self.eval_dataset)}")
            
            logging.info("Configuring bits and bytes config")
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
            
            tokenizer=AutoTokenizer.from_pretrained(self.model_id,trust_remote_code=True)
            tokenizer.pad_token=tokenizer.eos_token
            
            logging.info("Started loading the model using gpu")
             # Load model with CUDA configuration
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                quantization_config=bnb_config, 
                device_map={'': 0},  
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            model.resize_token_embeddings(len(tokenizer))
            model = prepare_model_for_kbit_training(model)
            logging.info("finished loading the model using gpu")
            
            
            logging.info("Configuring peft......")
            peft_config = LoraConfig(
                r=16,  # Rank of the update matrices
                lora_alpha=32,  # Scaling factor for the weight matrix
                lora_dropout=0.05,  # Dropout to add to the LoRA layers
                bias='none',  # Bias type
                task_type="CAUSAL_LM"  # Causal Language Modeling task
            )
            model = get_peft_model(model, peft_config)
            
           
            training_args = TrainingArguments(
                output_dir="emotion-prediction-model",
                per_device_train_batch_size=4   ,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=4,
                optim="adamw_torch",
                logging_steps=20,
                learning_rate=2e-4,
                fp16=True,
                warmup_ratio=0.1,
                lr_scheduler_type="linear",
                num_train_epochs=2,  
                save_strategy="epoch",
                evaluation_strategy="epoch",
                load_best_model_at_end=True,
                report_to="none"
            )
            
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,  
                dataset_text_field="text",
                max_seq_length=1024,
                tokenizer=tokenizer,
                args=training_args,
                packing=True,
                peft_config=peft_config,
            )
            
            logging.info("Training started ....................")
            trainer.train()
            trainer.save_model('model_ft/emotion_prediction_model')
            logging.info("Training finished sucessfully ....................")
            return model, tokenizer
            
        except Exception as e:
            raise CustomException(e,sys)