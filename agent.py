import os

import streamlit as st
from dotenv import load_dotenv
from autogen import  AssistantAgent
from groq import Groq


load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

config_list = {
    "model": "llama-3.1-70b-versatile",

    "api_type": "groq",
    "api_key": api_key,
}
valid_emotions = ['sentimental', 'afraid', 'proud', 'faithful', 'terrified', 'joyful',
                  'angry', 'sad', 'jealous', 'grateful', 'prepared', 'embarrassed',
                  'excited', 'annoyed', 'lonely', 'ashamed', 'guilty', 'surprised',
                  'nostalgic', 'confident', 'furious', 'disappointed', 'caring', 
                  'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful', 
                  'content', 'impressed', 'apprehensive', 'devastated']


    
def initialize_agents():
    emotion_agent=AssistantAgent(
            name="Emotion_agent",
            llm_config=config_list,
            system_message=f""" You have to predict the emotion of the user input from these list {valid_emotions}  of emotions and you have to say only the
            emotion of the user input in one word. 
            example:
            User input :"I feel so overwhelmed with work. I don't know how I'll ever catch up."

            Output:
            Emotion Label: Overwhelmed """
        )
        
    response=AssistantAgent(
            name="response",
            llm_config=config_list,
            system_message="""
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

Agent Response: [YOUR_EMPATHETIC_RESPONSE]

Example Input and Output:
Input:
"I feel so overwhelmed with work. I don't know how I'll ever catch up."

Output:

Agent Response: It's tough to feel buried under so much work. Remember, it's okay to take things one step at a time. Maybe start by prioritizing smaller tasks—it might help you feel more in control.
"""
        )
        
    user_proxy = AssistantAgent(
        name="user_proxy",
        system_message="Given a user question and get emotion lable and response from Emotion_agent and response respectively",
        llm_config=config_list
    )
    return user_proxy,emotion_agent,response

    
def final_output(query: str, user_proxy, emotion_agent, response):
    """Process the query and return the appropriate response"""
    res_emotion = user_proxy.initiate_chat(emotion_agent, message=query, max_turns=1)
    
    final_result=user_proxy.initiate_chat(response,message=query,max_turns=1)
     
    return res_emotion,final_result


def main():
    st.title("welcome to multiagent emotional chat bot")
    
    user_proxy,emotion_agent,response=initialize_agents()
    
    query = st.text_input("Enter your query:", "")
    
    if st.button("Send"):
        if query:
            try:
                    emotion,final_result= final_output(query, user_proxy, emotion_agent,response)
                    #st.success(f"Search Result:{res_method}")
                    st.write(f"{emotion.summary}")
                    st.write(f"{final_result.summary}")
            except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()