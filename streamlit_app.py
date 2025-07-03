#Importing necessary modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from google.colab import userdata
import pypdf
import streamlit as st
import random
import time

embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

client = Groq(
    api_key=userdata.get('Groq_Api_Key')
)

def document_loader(document):
    pdf_filename = list(document.keys())[0]
    loader = PyPDFLoader(pdf_filename)
    pages = loader.load_and_split()
    return pages

def split_text():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter

def create_chunks(pages,text_splitter):
    all_page_text = ""
    for page in pages:
        all_page_text += page.page_content
    texts = text_splitter.create_documents([all_page_text])
    return texts

def create_embeddings(chunks):
    text_contents = [doc.page_content for doc in chunks]
    embeddings = embedding_model.encode(text_contents)
    return embeddings, text_contents

def create_vector_store(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def retrieval(index, user_prompt, text_contents):
    query_embedding = embedding_model.encode([user_prompt])
    k = 6
    distances, indices = index.search(query_embedding, k)
    retrieved_info = [text_contents[idx] for idx in indices[0]]
    context = "\\n".join(retrieved_info)
    return context

def chat_completion_SRS_Analysis(context, user_input):
    prompt = f"""
    You are acting as a senior software analyst specializing in software requirements specification (SRS) documents.
    ...
    **Document Context:**
    {context}
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        model="llama-3.3-70b-versatile",
    )
    st.write(chat_completion.choices[0].message.content)

# (Repeat the same structure for other functions...)

# 1. Title
st.title("DeepDocx")

# 2. File Upload
uploaded_doc = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])

# 3. Session Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Chat Input
if prompt := st.chat_input("Hey there! Upload your document and I am here to analyze."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Document processing
    if uploaded_doc:
        pages=document_loader(uploaded_doc)
        text_splitter=split_text()
        chunks=create_chunks(pages,text_splitter)
        embeddings,text_contents=create_embeddings(chunks)
        vector_store=create_vector_store(embeddings)
        context=retrieval(vector_store, prompt, text_contents)

        mode = st.selectbox("Choose analysis mode:", (
            "SRS Analysis", "Research Paper", "GRC Document", "Whitepaper", "Project Report"))

        # Generate and display assistant response
        if mode == "SRS Analysis":
            response = chat_completion_SRS_Analysis(context, prompt)
        elif mode == "Research Paper":
            response = chat_completion_Research_Analysis(context, prompt)
        elif mode == "GRC Document":
            chat_completion_GRC_Analysis(context, prompt)
        elif mode == "Whitepaper":
            chat_completion_Whitepaper_Analysis(context, prompt)
        elif mode == "Project Report":
            chat_completion_project_report_Analysis(context, prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
