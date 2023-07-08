import streamlit as st

def get_apikey():
    OPENAI_API_KEY = st.text_input(":blue[Enter Your OPENAI API-KEY :]", 
                placeholder="Paste your OpenAI API key here (sk-...)",
                type="password",
                )
    return OPENAI_API_KEY