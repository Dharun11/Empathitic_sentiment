from setuptools import find_packages,setup
from typing import List

hypen_e_dot="-e ."    
def get_requirements(file_pathh:str)-> List[str]:
    
    '''
    This function will return the list of requirements
    '''
    requirements=[]
     
    with open(file_pathh) as f:
        requirements = f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    
    if hypen_e_dot in requirements:
        requirements.remove(hypen_e_dot)
    return requirements
        
         
         
setup(
    name="assignment_ml",
    version="0.0.1",
    author="Dharun balaji",
    author_email="dharunashok113@gmail.com",
    packages=find_packages(),  # this will automatically find the packages that we are using in the project
    install_requires=get_requirements('requirements.txt')
    
)