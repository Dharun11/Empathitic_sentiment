from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import Ingect_data
from src.components.llm_trainer import ModelTrainer
from src.model_evalutor import LLMResponseEvaluator
import sys


logging.info("Training started ....................")
if __name__=="__main__":
    try:
        logging.info("Passing the data path .....")
        data_path="Data\Data.csv"
        data_ingestion_obj=Ingect_data(data_path=data_path)
        train_dataset,eval_dataset=data_ingestion_obj.load_custom_data()
        logging.info("Sucessfully received the train and evaluation datset from data ingestion....... .....")
        
        
        logging.info("Entering into model trainer..............")
        model_trainer_obj=ModelTrainer(train_dataset,eval_dataset)
        model,tokenizer=model_trainer_obj.train_model()
        
        
        
        logging.info("Sucessfully trained the model..............")
        
        
        
    except Exception as e:
        raise CustomException(e,sys)
        
        