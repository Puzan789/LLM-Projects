import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

## load gemini model and get response

model=genai.GenerativeModel("gemini-pro")
def get_response(question):
    response=model.generate_content(question)
    return response.text


st.set_page_config(page_title="Q and A demo")
input=st.text_input("Input :",key=input)
submit=st.button("Ask the question")

if submit:
    response=get_response(input)
    st.write(response)