import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class EmotionPredictor:
    def __init__(self, 
                 base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 model_path="model_ft/emotion_prediction_model"):
        """
        Initialize the Emotion Predictor
        
        :param base_model_id: Base model identifier
        :param model_path: Path to fine-tuned model weights
        """
        self.base_model_id = base_model_id
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """
        Load the fine-tuned model and tokenizer
        """
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, 
            device_map={'': 0},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load fine-tuned PEFT model
        model = PeftModel.from_pretrained(base_model, self.model_path)
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        
    def predict(self, conversation):
        """
        Predict emotion for a given conversation
        
        :param conversation: Conversation text to predict emotion for
        :return: Predicted emotion
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Prepare input
        inputs = self.tokenizer(
            conversation, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        ).to("cuda")
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=False
            )
        
        # Decode and return prediction
        predicted_emotion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return predicted_emotion

def main():
    # Initialize predictor
    predictor = EmotionPredictor()
    
    # Example conversations
    conversations = [
        "System: You are a helpful assistant.\nUser: I just lost my job and I'm feeling devastated.",
        "System: You are a friendly chatbot.\nUser: I got a promotion today and I'm so excited!",
        "System: Provide emotional support.\nUser: I'm worried about my upcoming exam.",
        "System: Engage in friendly conversation.\nUser: The weather is nice today!"
    ]
    
    # Predict emotions for conversations
    for conversation in conversations:
        try:
            predicted_emotion = predictor.predict(conversation)
            print(f"Conversation: {conversation}")
            print(f"Predicted Emotion: {predicted_emotion}\n")
        except Exception as e:
            print(f"Error predicting emotion: {e}")

if __name__ == "__main__":
    main()