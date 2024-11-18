import streamlit as st
import google.generativeai as ai


key = "AIzaSyDrt9sLDzhY7qm_noqxckAaQPFB3DYs-5U"

ai.configure(api_key=key)
st.title("Code Reviewer")


code_snippet = st.text_area("Enter your Python code here...")

prompt = f"The following Python code has an error. Debug it and give me a correct code"

model = ai.GenerativeModel(model_name="models/gemini-1.5-flash")


button = st.button("Generate")

if button:
    responce = model.generate_content([code_snippet,prompt])
    st.title("Fixed code")
    st.write(responce.text)
    