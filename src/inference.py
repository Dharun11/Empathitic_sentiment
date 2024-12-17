import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging
from src.logger import logging

class ModelInference:
    def __init__(self, 
                 model_path: str = 'model_ft/emotion_prediction_model',
                 base_model_path: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'):
       
        
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
           
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='auto', 
                torch_dtype=torch.bfloat16  # Memory efficient dtype
            )
            
            self.logger.info("Model and tokenizer loaded successfully.")
        
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, 
                           prompt: str, 
                           max_length: int = 200,
                           temperature: float = 0.7,
                           top_k: int = 50,
                           top_p: float = 0.95) -> str:
        
        try:
            # Prepare input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response."
    
    def interactive_mode(self):
        # Creating a chat type inference ...
        self.logger.info("Starting interactive inference mode...")
        self.logger.info("Type 'exit' to quit the interaction.")
        
        while True:
            try:
                # Get user input
                user_input = input("\nEnter your prompt (or 'exit' to quit): ")
                
                # Checking  for exit condition
                if user_input.lower() == 'exit':
                    self.logger.info("Exiting interactive mode.")
                    break
                
               
                response = self.generate_response(user_input)
                print("\nModel Response:")
                print(response)
            
            except KeyboardInterrupt:
                self.logger.info("\nOperation cancelled by user.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")

def main():
    
    inference = ModelInference()
    
    try:
        
        inference.interactive_mode()
    except Exception as e:
        print(f"An error occurred: {e}")
       

if __name__ == "__main__":
    main()

