# Importing necessary modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import streamlit as st
import os

# Load embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Get API key from Streamlit secrets
Groq_API = st.secrets["Grok_Api_Key"]
client = Groq(api_key=Groq_API)

# --- Helper Functions ---

def document_loader(uploaded_doc):
    with open(uploaded_doc.name, "wb") as f:
        f.write(uploaded_doc.read())
    loader = PyPDFLoader(uploaded_doc.name)
    pages = loader.load_and_split()
    return pages

def split_text():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

def create_chunks(pages, text_splitter):
    all_page_text = "".join(page.page_content for page in pages)
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
    return "\n".join(retrieved_info)

# --- Chat Completion Functions ---

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
    return chat_completion.choices[0].message.content

def chat_completion_Research_Analysis(context, user_input):
    prompt = f"""
You are a research assistant specializing in technical literature.

You must:
- Extract objectives, key findings, and methods
- Explain terms or abbreviations
- Point out assumptions or gaps
- Summarize conclusions and future work

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
    return chat_completion.choices[0].message.content

def chat_completion_GRC_Analysis(context, user_input):
    prompt = f"""
You are a compliance specialist analyzing legal or regulatory documents.

Extract and explain:
- Responsibilities, restrictions
- Key terms, penalties
- Risks or ambiguities
- Practical advice

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
    return chat_completion.choices[0].message.content

def chat_completion_Whitepaper_Analysis(context, user_input):
    prompt = f"""
You are a strategist analyzing whitepapers.

Summarize:
- Problem and solution
- Architecture or technology
- Financial models or tokens
- Adoption or roadmap

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
    return chat_completion.choices[0].message.content

def chat_completion_project_report_Analysis(context, user_input):
    prompt = f"""
You are a senior project analyst.

Summarize:
- Milestones, KPIs
- Risks and delays
- Timeline and assignments
- Improvement recommendations

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
    return chat_completion.choices[0].message.content

# --- Streamlit App ---

st.title("📄 DeepDocx – Document Intelligence App")

uploaded_doc = st.file_uploader("📤 Upload your document", type=["pdf", "docx", "txt"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Hey there! Upload your document and ask me anything about it."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_doc:
        pages = document_loader(uploaded_doc)
        text_splitter = split_text()
        chunks = create_chunks(pages, text_splitter)
        embeddings, text_contents = create_embeddings(chunks)
        vector_store = create_vector_store(embeddings)
        context = retrieval(vector_store, prompt, text_contents)

        mode = st.selectbox("Choose analysis mode:", (
            "SRS Analysis", "Research Paper", "GRC Document", "Whitepaper", "Project Report"))

        if mode == "SRS Analysis":
            response = chat_completion_SRS_Analysis(context, prompt)
        elif mode == "Research Paper":
            response = chat_completion_Research_Analysis(context, prompt)
        elif mode == "GRC Document":
            response = chat_completion_GRC_Analysis(context, prompt)
        elif mode == "Whitepaper":
            response = chat_completion_Whitepaper_Analysis(context, prompt)
        elif mode == "Project Report":
            response = chat_completion_project_report_Analysis(context, prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
