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


# Get API key from Streamlit secrets
Groq_API = st.secrets["Grok_Api_Key"]
client = Groq(api_key=Groq_API)


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

Your task is to analyze the uploaded SRS document and help users by:
1. Listing **functional** and **non-functional** requirements.
2. Extracting **modules**, **user roles**, **data flows**.
3. Explaining **technical terms** or acronyms.
4. Pointing out **incomplete or ambiguous** parts.
5. Summarizing the **scope and objective**.
6. Offering implementation or testing advice.

Only refer to the document context. Do not assume or invent.

Only answer the relevant questions like for example if a user asks to list functional requirements then list only functional requirements.

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

def chat_completion_Research_Analysis(context, user_input):
    prompt = f"""
You are a research assistant specializing in technical literature.

You must be able to:
- Extract objectives, key findings, and methods
- Explain terms or abbreviations
- Point out assumptions or gaps
- Summarize conclusions and future work

Only answer the relevant questions like for example if a user asks to explain a term or an abbrevation just do that only.

Use only this context:
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

def chat_completion_GRC_Analysis(context, user_input):
    prompt = f"""
You are a compliance specialist analyzing legal or regulatory documents.

You shpuld be able to extract and explain:
- Responsibilities, restrictions
- Key terms, penalties
- Risks or ambiguities
- Practical advice

Only answer the relevant questions like for example if a user asks about risks then only state that.

Only use this context:
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

def chat_completion_Whitepaper_Analysis(context, user_input):
    prompt = f"""
You are a strategist analyzing whitepapers.

You should be able to summarize:
- Problem and solution
- Architecture or technology
- Financial models or tokens
- Adoption or roadmap

Only answer the relevant questions like for example if a user wants a roadmap then provide only that.

Stick to context:
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

def chat_completion_project_report_Analysis(context, user_input):
    prompt = f"""
You are a senior project analyst.

You should be able to summarize:
- Milestones, KPIs
- Risks and delays
- Timeline and assignments
- Improvement recommendations

Only answer the relevant questions like for example if a user asks to state the timeline then only do that.

Context:
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


# 1. Title
st.title("📄 DeepDocx - Your Intelligent Document Analyzer")

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
