# Setting the environment with openAI API key
# User would require to use their own API key
import os
os.environ["OPENAI_API_KEY"]="********"

# A sample pdf has been loaded and converted to Langchain document
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("11. Electricity.pdf")
docs = loader.load()
full_text = " ".join([page.page_content for page in docs]) 

# A schema is created for the MCQ type question
from pydantic import BaseModel
class MCQ(BaseModel):
    question: str
    option_A: str
    option_B: str
    option_C: str
    option_D: str
    correct_answer: str
    explanation: str

# Another schema for a collection of multiple MCQ question
# using the previous schema
from typing import List
class MCQSet(BaseModel):
    questions: List[MCQ]

# Initializing model with the help of the schema and OpenAI
from langchain_openai import OpenAI
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai",temperature=0)    
structured_model=llm.with_structured_output(MCQSet)

# Getting result from the model, by passing the loaded sample document as 
# context in the prompt
result=structured_model.invoke(f'''Frame 10 MCQ questions with their options 
and solution out of the text provided below. Questions are Multiple choice 
correct type with exactly four options. Note that the question should not refer 
to the text as students does not have access to the text while answering 
the questions : \n {full_text}''')

# Constructing dataframe from the result and saving it as a csv file.
import pandas as pd
df = pd.DataFrame([mcq.model_dump() for mcq in result.questions])
df.to_csv('Electricity_MCQ.csv', index=False)
