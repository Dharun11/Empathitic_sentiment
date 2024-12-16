import pandas as pd
from datasets import Dataset,load_dataset


class Ingect_data:
    def __init__(self, data_path:str, model_id:str="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.data_path=data_path
        self.model_id=model_id
        
    def load_custom_data(self)-> tuple:
        
        df=pd.read_csv(self.data_path)
        df['text']=df.apply(self.format_data,axis=1)
        
        train_size=int(0.8*len(df))
        train_df=df.iloc[:train_size]
        eval_df=df.iloc[train_size:]
        
        train_dataset=Dataset.from_pandas(train_df[['text']])
        eval_dataset=Dataset.from_pandas(eval_df[['text']])
        
        return train_dataset,eval_dataset
    
    def format_data(self,row):
        return f"Situation: {row.get('situation_text','')}\n User Query: {row.get('user_queries','')}\n Emotion Label: {row.get('emotion_label','')}\n agent response: {row.get('agent_responses','')}"
    
    
    
if __name__=="__main__":
    data_path="Data\Data.csv"
    data_ingection=Ingect_data(data_path)
    
    train_dataset,eval_dataset=data_ingection.load_custom_data()