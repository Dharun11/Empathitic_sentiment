import sys
from logger import logging

def error_message_detail(error,error_detail:sys):
    #through this we are getting error detail from the system It passes 3 outputs, we need only 3rd one
    _,_,exec_tb=error_detail.exc_info()
    filename=exec_tb.tb_frame.f_code.co_filename
    error_msg=f"Error occured in python script name is {filename} line number {exec_tb.tb_lineno} with the error message is {str(error)} "
    return error_msg
    
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        #since we inherit the class from exception class we are using Super.init()
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
    
