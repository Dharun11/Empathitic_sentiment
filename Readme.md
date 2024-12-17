## Welcome to Empathitic sentiment analysis assignment
Empathetic Sentiment Analysis with TinyLlama 1.1B
Overview
This project focuses on performing empathetic sentiment analysis using the TinyLlama-1.1B language model. The goal is to predict emotions in text based on empathetic dialogues and enhance the model's understanding of different emotional tones.

The model has been fine-tuned using the Kaggle Empathetic Dialogues dataset and leverages performance-efficient fine-tuning (PEFT) techniques to handle training on a limited GPU resource. The final model can predict emotion labels and generate responses based on input prompts.

Project Structure
bash
Copy code
src/
│
├── components/
│   ├── dataingestion.py        # Data preprocessing and conversion to Hugging Face dataset
│   ├── llm_trainer.py          # Model training, fine-tuning, and optimization
│
└── pipeline/
    ├── training_main.py         # Training orchestration
    ├── model_evaluator.py       # Evaluation of the fine-tuned model
    ├── inference.py             # Generate chat-like responses from the fine-tuned model
Key Features:
Data Preprocessing: Includes handling of missing values, text standardization, and feature engineering to improve the dataset's quality.
Model Training: Fine-tuning the TinyLlama-1.1B language model with the BitsAndBytes configuration for memory efficiency and PEFT (Performance Efficient Fine-Tuning) for reducing computational overhead while training specific layers of the model.
Evaluation: Includes model testing and evaluation using metrics like accuracy, precision, recall, and F1-score.
Inference: Interactive chat-based inference mode allowing you to input prompts and get emotion-labeled responses.
Dependencies
This project requires the following Python libraries:

torch
transformers
datasets
logging
re
argparse
To install the required dependencies, use the following command:

bash
Copy code
pip install -r requirements.txt
Data
The model was fine-tuned using the Empathetic Dialogues dataset available on Kaggle by atharvjairath. The dataset includes 64,636 rows with the following features:

situation: The context or scenario of the conversation.
emotion: The emotional tone of the response.
empathetic_dialogues: The empathetic responses in the conversation.
labels: The emotion labels assigned to each response.
Additional features like unnamed columns were dropped for cleaning purposes.

Model Configuration and Fine-Tuning
TinyLlama-1.1B Model
We used TinyLlama-1.1B-Chat-v1.0 for the fine-tuning process due to its lightweight architecture, which is suitable for limited GPU resources.

Performance Efficient Fine-Tuning (PEFT)
PEFT was applied to update the weights of specific layers relevant to Causal Language Modeling (CLM) tasks. This allows for a more scalable and efficient fine-tuning process while keeping most of the model frozen.

BitsAndBytes Configuration
We used BitsAndBytes with 4-bit precision for memory efficiency and improved training speed. The configuration was set to:

load_in_4bit=True: Load the model with 4-bit precision.
bnb_4bit_use_double_quant=True: Use double quantization to further reduce memory.
bnb_4bit_quant_type="nf4": Use the NF4 quantization type.
bnb_4bit_compute_dtype=torch.bfloat16: Use bfloat16 precision for computation.
Usage
Model Inference: Once the model is fine-tuned, you can perform inference using the ModelInference class. It loads the fine-tuned model and tokenizer, and you can input a prompt to generate an emotional response.

Training: The llm_trainer.py script trains the model using the preprocessed dataset and stores the fine-tuned model in the designated folder.

Evaluation: Use the model_evaluator.py script to evaluate the model's performance on the test data, generate classification metrics, and perform error analysis.

Reflection and Future Work
While the model demonstrates strong performance in predicting emotions and generating responses, there is always room for improvement:

Cultural Sensitivity: Further work could involve training the model on more diverse datasets to ensure it understands a wide range of emotional tones across cultures and contexts. This would improve its ability to handle emotions in multi-cultural or cross-linguistic scenarios.

Enhanced Fine-Tuning: Additional hyperparameter optimization and experimenting with different techniques like knowledge distillation could lead to more efficient models.

Multimodal Capabilities: Future iterations of this project could extend the model to handle multimodal inputs, including visual or audio data, to improve its empathetic responses in real-world applications.

License
This project is licensed under the MIT License - see the LICENSE file for details.