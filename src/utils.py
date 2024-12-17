import sys
from src.exception import CustomException

def generate_report(content,filename="report_ouput.txt"):
    try:
        with open(filename,"w") as f:
            f.write(content)
    except Exception as e:
        raise CustomException(e,sys)
        
    
        