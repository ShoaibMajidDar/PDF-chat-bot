import streamlit as st
import os
from apikey import get_apikey
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import  LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory



def make_context(docs):
  context = ""
  for doc in docs:
    doc = doc.page_content +  "\n\nSource: " + doc.metadata
    context = context + doc + "\n\n"
  return context




OPENAI_API_KEY = get_apikey()


os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

if OPENAI_API_KEY:
    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature = 0, openai_api_key=OPENAI_API_KEY, max_tokens=800)
    gptturbo = ChatOpenAI(model='gpt-3.5-turbo',temperature = 0, openai_api_key=OPENAI_API_KEY, max_tokens=800)


    if "generated" not in st.session_state:
           st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")



    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")


    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        data = ""
        for page in pdf_reader.pages:
            data += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 0
        )


        texts = text_splitter.split_text(data)       
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        docsearch = FAISS.from_texts(texts, embedding = embeddings)

        question = st.text_input('Ask any question', key="input")

        if question:
            docs = docsearch.similarity_search(query=question)
            for doc in docs:
                 doc.metadata = uploaded_file.name
            template = """
your job is to answer the questions asked by the users. Create a final answer with references ("SOURCES").
If the answer is not in the context, then try to answer it using your own knowledge.
Source of the context is written at the end of the context.
At the end of your answer write the source of the context in the following way: \n\nSource: (source)
Chat history is also provided to you.
Context: {context}
---
Chat History: {chat_history}
Question: {question}
Answer: Let's think step by step and give best answer possible. Use points when needed.
"""

            context = make_context(docs)
            prompt = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"]).partial(context=context)

            llm_chain = LLMChain(prompt=prompt, llm=gptturbo, verbose=False, memory=st.session_state.memory)


            response = llm_chain.run(question)
            st.session_state.past.append(question)
            st.session_state.generated.append(response)

        with st.expander("Conversation", expanded=True):
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                st.info(st.session_state["past"][i],icon="🧐")
                st.success(st.session_state["generated"][i], icon="🤖")

            




