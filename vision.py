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
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(input, image):
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

# Tab/App name with emojis
st.set_page_config(page_title="✨ Vision App 🌟")

# Header with emojis
st.header("🧠 Gemini App By RIZZZ !!! 🤖")
st.write("Welcome to the Gemini-powered Vision App! 🎨 Upload an image and get AI insights 🌟")

# Input prompt with emojis
input = st.text_input("📝 Input Prompt:", key='input', placeholder="Type your question here...")

# File uploader with emojis
uploaded_file = st.file_uploader("📷 Choose an Image...", type=['jpg', 'jpeg', 'png'])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📸 Uploaded Image', use_column_width=True)

# Submit button with emojis
submit = st.button("🔍 Tell me about image: ")

if submit:
    response = get_gemini_response(input, image)
    st.subheader("💡 Response Is:")
    st.write(response)
