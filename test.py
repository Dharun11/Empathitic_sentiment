import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, 
                 model_path: str = 'model_ft/emotion_prediction_model',
                 base_model_path: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='auto', 
                torch_dtype=torch.bfloat16  # Memory efficient dtype
            )
            
            # Define a system instruction set
            self.system_prompt_final = """
            You are an empathetic AI assistant designed to:

Understand and analyze the emotion conveyed in the user's message.
Provide supportive and compassionate responses.
Use a warm, non-judgmental, and caring tone to validate the user's feelings.
Offer constructive insights or suggestions to help the user feel understood.
When responding:
you shouldn't provide any other information. Do not halucinate and always reply as a emotional partner.
Identify the user's emotion.
Respond empathetically and constructively.
Keep the response format strictly as follows:
Output Format:
Emotion Label: [DETECTED_EMOTION]
Agent Response: [YOUR_EMPATHETIC_RESPONSE]

Example Input and Output:
Input:
"I feel so overwhelmed with work. I don't know how I'll ever catch up."

Output:
Emotion Label: Overwhelmed
Agent Response: It's tough to feel buried under so much work. Remember, it's okay to take things one step at a time. Maybe start by prioritizing smaller tasks—it might help you feel more in control.

Expected Behavior:

Always analyze the user's emotional tone.
Respond in the specified format using concise and empathetic language. 
            """
            
            self.valid_emotions = ['sentimental', 'afraid', 'proud', 'faithful', 'terrified', 'joyful',
                  'angry', 'sad', 'jealous', 'grateful', 'prepared', 'embarrassed',
                  'excited', 'annoyed', 'lonely', 'ashamed', 'guilty', 'surprised',
                  'nostalgic', 'confident', 'furious', 'disappointed', 'caring', 
                  'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful', 
                  'content', 'impressed', 'apprehensive', 'devastated']
            self.emotion_label_prompt=f"""
            Understand and analyze the emotion conveyed in the user's message.
            Identify the user emotion and match the emotion from this list {self.valid_emotions}
            Finally give the emotion label only.
            do not give any other answers and do not reply with your own knowledge.
            This is the strict order you have to find only the emotion of the user input.
            Example Input and Output:
                Input:
                "I feel so overwhelmed with work. I don't know how I'll ever catch up."

                Output:
                Emotion Label: Overwhelmed
            
            """
            
            logger.info("Model and tokenizer loaded successfully.")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response_emotion(self, 
                           prompt: str, 
                           max_length: int = 350,
                           temperature: float = 0.7,
                           top_k: int = 50,
                           top_p: float = 0.95) -> str:
        try:
            # Combine system prompt with user input
            full_prompt = f"{self.valid_emotions}\n\nUser: {prompt}\n\nAgent Emotion Label:"
            
            inputs = self.tokenizer(
                full_prompt, 
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
            
            emotion_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return emotion_response
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response."
    
    def generate_response(self, 
                           prompt: str,
                           emotion: str, 
                           max_length: int = 512,
                           temperature: float = 0.7,
                           top_k: int = 50,
                           top_p: float = 0.95) -> str:
        try:
            # Combine system prompt with user input
            full_prompt = f"{self.system_prompt_final}\n\nUser: {prompt}\n\nEmotion Label: {emotion}\n\nAgent Response"
            
            inputs = self.tokenizer(
                full_prompt, 
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
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response."

# Streamlit app setup
def main():
    st.title("Empathetic Sentiment Analysis Chatbot")
    st.write("Chat with an AI assistant designed to understand and support your emotions.")

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    try:
        inference = ModelInference()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Display conversation history
    for user_msg, bot_msg in st.session_state.conversation:
        st.markdown(f"**User**: {user_msg}")
        st.markdown(f"**Bot**: {bot_msg}")

    # Create text input for user prompt
    user_input = st.text_input("Share your thoughts or feelings:", key="user_input")

    # Add button to send the message
    if st.button("Send"):
        if user_input:
            # Generate response from the model
            emotion_label=inference.generate_response_emotion(user_input)
            bot_response = inference.generate_response(user_input,emotion_label)
            st.write(bot_response,emotion_label)
            # Add to conversation history
            #st.session_state.conversation.append((user_input, bot_response))

            # Clear the input box
            #st.experimental_set_query_params()
        else:
            st.error("Please enter a message.")

if __name__ == "__main__":
    main()