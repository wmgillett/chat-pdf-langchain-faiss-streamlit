import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
# get callback to get stats on query cost
from langchain.callbacks import get_openai_callback
import os
load_dotenv() 
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made by [Prompt Engineer](https://youtube.com/@engineerprompt)')
 
def main():
    st.write("Hello")
    st.header("Chat with PDF 💬")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    # check for pdf and read present
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # extract the text from the PDF
        page_text = ""
        for page in pdf_reader.pages:
            page_text += page.extract_text()
        # split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=page_text)
        # count the current number of chunks
        #st.write(f"Number of chunks: {len(chunks)}")
        store_name = pdf.name[:-4]
        st.write(f"{store_name}")
        # check if vector store already exists
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
            st.write("Embeddings loaded from disk")
        # else create embeddings and save to disk
        else:
            # create embeddings object
            embeddings = OpenAIEmbeddings()
            # create vector store to hold the embeddings
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            st.write("Embeddings saved to disk")
        # ask the user for a question
        question = st.text_input("Ask a question")
        if question:
            docs = vector_store.similarity_search(question, k=3)
            st.write(docs)
            # create the LLM object
            llm = OpenAI(temperature=0.0, max_tokens=1000, model_name="gpt-3.5-turbo")
            # load the question answering chain
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            # generate the answer
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=question)
                print(cb)
                st.write(response)

if __name__ == '__main__':
    main()