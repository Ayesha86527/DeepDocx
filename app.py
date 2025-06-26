#Importing necessary modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import pypdf
import streamlit as st
import os


embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

Groq_API=st.secrets["Grok_Api_Key"]

client = Groq(
    api_key=Groq_API
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
    context = "\n".join(retrieved_info)
    return context



def chat_completion_SRS_Analysis(context, user_input):
    prompt = f"""
    You are acting as a senior software analyst specializing in software requirements specification (SRS) documents.

    Your task is to help developers, project managers, freelancers, and stakeholders analyze and understand the contents of an uploaded SRS document.

    You must be able to perform the following tasks ONLY using the provided document context below:

    1. Identify and list key **functional requirements** (what the system should do).
    2. Identify and list **non-functional requirements** (performance, usability, security, etc.).
    3. Extract any clearly defined **modules, features, user roles**, and **data flows**.
    4. Explain or define **technical terms** or acronyms present in the context.
    5. Point out any **incomplete, ambiguous, or contradictory** parts.
    6. Summarize the **overall scope and objective** of the system if evident.
    7. Provide brief **implementation advice**, test case ideas, or questions to clarify missing parts.

    If the required detail is not present in the provided context, do not make assumptions. Instead, clearly state:
    "The context does not provide enough information about [missing detail]."

    Only refer to the document context provided. Do not use external knowledge or invent features not mentioned.

    You are in **SRS Analysis Mode**.

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
    You are a senior research assistant with expertise in technical and academic literature.

    Your task is to help developers, researchers, and analysts understand and extract insights from research papers.

    When a user uploads a paper or asks a question, you must be able to:

    1. Extract the paper’s objective, hypothesis, and key findings.
    2. Identify and summarize the methodology used.
    3. Explain any technical terms or abbreviations.
    4. Point out limitations or assumptions made by the authors.
    5. Highlight any cited datasets, frameworks, or models.
    6. Summarize the conclusions and potential future work mentioned.
    7. Provide implementation advice if applicable.

    Only refer to the **provided context** below. Do not use external knowledge.

    You are in **Research Paper Analysis Mode**.

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


def chat_completion_GRC_Analysis(context, user_input):
    prompt = f"""
    You are a cybersecurity and compliance specialist.

    Your task is to assist professionals in interpreting legal, regulatory, or compliance documents (e.g., GDPR, HIPAA, SOC 2, etc.).

    When analyzing the document or answering a question, you should be able to:

    1. Identify the core rules, responsibilities, and restrictions.
    2. Explain the implications for organizations and individuals.
    3. Summarize the key sections in clear, accessible language.
    4. Highlight penalties, deadlines, or mandatory actions.
    5. Clarify any legal or policy terms.
    6. Point out any ambiguous or high-risk sections.
    7. Offer practical advice for aligning with the regulation (where relevant).

    Only refer to the **provided document context**.

    You are in **Compliance & Regulation Mode**.

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


def chat_completion_Whitepaper_Analysis(context, user_input):
    prompt = f"""
    You are a product strategist and technical writer with deep experience analyzing whitepapers.

    Your job is to help users understand key elements of whitepapers, such as new technologies, protocols, or systems.

    When reviewing a whitepaper, you should be able to:

    1. Summarize the problem the whitepaper is solving.
    2. Identify the proposed solution or system architecture.
    3. Highlight any unique technologies, mechanisms, or tokens.
    4. Clarify any financial or economic models if present.
    5. Note adoption strategies, competitive advantages, or roadmap plans.
    6. Flag vague, overly complex, or potentially misleading claims.

    Stick to the **document context only** and don't assume outside knowledge.

    You are in **Whitepaper Analysis Mode**.

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


def chat_completion_project_report_Analysis(context, user_input):
    prompt = f"""
    You are a senior project analyst and reporting specialist.

    Your task is to analyze and summarize software or technical project reports and roadmaps.

    When reviewing the content, you should be able to:

    1. Summarize milestones, goals, and completion status.
    2. Extract progress indicators (metrics, KPIs, charts).
    3. Identify risks, blockers, or delays.
    4. Provide a timeline overview if available.
    5. Extract task allocations, responsibilities, or stakeholder mentions.
    6. Recommend improvements or attention areas (if possible).

    Only refer to the **provided document context**.

    You are in **Project Report Analysis Mode**.

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



import random
import time

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
          chat_completion_GRC_Analysis(context,prompt)
        elif mode == "Whitepaper":
          chat_completion_Whitepaper_Analysis(context,prompt)
        elif mode == "Project Report":
          chat_completion_project_report_Analysis(context,prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
