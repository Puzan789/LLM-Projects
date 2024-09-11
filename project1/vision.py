import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os
from PIL import Image

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

model=genai.GenerativeModel("models/gemini-1.5-flash-latest")
def get_response(input,image=None):
    if input!="":
        response=model.generate_content([input,image])
    else:
        response=model.generate_content(image)
    return response.text

st.set_page_config(page_title="Q and A demo")
st.header("Gemini Aplication")
input=st.text_input("Input Prompt:",key="input")
uploaded_file=st.file_uploader("Choose an image...",type=["jpg", "png", "jpeg"])
image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_column_width=True)

submit=st.button("Submit")
if submit:
    response=get_response(input,image=image)
    st.subheader("Response")
    st.write(response)