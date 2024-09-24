import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Function to load gemini model

model = genai.GenerativeModel("gemini-pro")

def get_gemini_response(question):
    response = model.generate_content(question)
    return response.text

#Initialize Streamlit
st.set_page_config(page_title="Q&A Demo")

st.header("Google Gemini")

input = st.text_input("Enter Your Prompt: ", key='input')
submit = st.button("Ask the Question: ")

if submit:
    response = get_gemini_response(input)
    st.subheader("The Response is: ")
    st.write(response)
