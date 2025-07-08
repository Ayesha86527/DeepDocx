# Importing necessary modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import pypdf
import streamlit as st
import tempfile

embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Get API key from Streamlit secrets
Grok_Api_Key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=Grok_Api_Key)


def document_loader(document):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(document.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()
    return pages

def split_text():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

def create_chunks(pages, text_splitter):
    all_text = "".join([page.page_content for page in pages])
    return text_splitter.create_documents([all_text])

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

# --- Chat Completion Modes ---

def chat_completion_SRS_Analysis(context, user_input):
    prompt = f"""
You are acting as a senior software analyst specializing in software requirements specification (SRS) documents.

Your task is to analyze the uploaded SRS document and help users by:
1. Listing functional and non-functional requirements.
2. Extracting modules, user roles, data flows.
3. Explaining technical terms or acronyms.
4. Pointing out incomplete or ambiguous parts.
5. Summarizing the scope and objective.
6. Offering implementation or testing advice.

Only answer relevant questions. Do not assume or invent.

Document Context:
{context}
"""
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
        model="llama-3.3-70b-versatile",
    )
    st.write(response.choices[0].message.content)

def chat_completion_Research_Analysis(context, user_input):
    prompt = f"""
You are a research assistant specializing in technical literature.

You must be able to:
- Extract objectives, key findings, and methods
- Explain terms or abbreviations
- Point out assumptions or gaps
- Summarize conclusions and future work

Use only this context:
{context}
"""
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
        model="llama-3.3-70b-versatile",
    )
    st.write(response.choices[0].message.content)

def chat_completion_GRC_Analysis(context, user_input):
    prompt = f"""
You are a compliance specialist analyzing legal or regulatory documents.

You should be able to extract and explain:
- Responsibilities, restrictions
- Key terms, penalties
- Risks or ambiguities
- Practical advice

Use only this context:
{context}
"""
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
        model="llama-3.3-70b-versatile",
    )
    st.write(response.choices[0].message.content)

def chat_completion_Whitepaper_Analysis(context, user_input):
    prompt = f"""
You are a strategist analyzing whitepapers.

You should be able to summarize:
- Problem and solution
- Architecture or technology
- Financial models or tokens
- Adoption or roadmap

Use only this context:
{context}
"""
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
        model="llama-3.3-70b-versatile",
    )
    st.write(response.choices[0].message.content)

def chat_completion_project_report_Analysis(context, user_input):
    prompt = f"""
You are a senior project analyst.

You should be able to summarize:
- Milestones, KPIs
- Risks and delays
- Timeline and assignments
- Improvement recommendations

Use only this context:
{context}
"""
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
        model="llama-3.3-70b-versatile",
    )
    st.write(response.choices[0].message.content)


# Streamlit App Layout
st.title("📄 DeepDocx - Your Intelligent Document Analyzer")

# Upload
uploaded_doc = st.file_uploader("📤 Upload your document", type=["pdf", "docx", "txt"])

# Session state
if "mode" not in st.session_state:
    st.session_state.mode = ""
if "mode_selected" not in st.session_state:
    st.session_state.mode_selected = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "text_contents" not in st.session_state:
    st.session_state.text_contents = None

# Step 1: Mode Selection
if uploaded_doc:
    if not st.session_state.mode_selected:
        selected_mode = st.selectbox("🎯 Choose analysis mode", (
            "SRS Analysis", "Research Paper", "GRC Document", "Whitepaper", "Project Report"))

        if st.button("✅ Confirm Mode"):
            st.session_state.mode = selected_mode
            st.session_state.mode_selected = True
            st.rerun()

# Step 2: Process Document Once After Mode Selection
if uploaded_doc and st.session_state.mode_selected and st.session_state.vector_store is None:
    st.success(f"Mode selected: {st.session_state.mode}")
    pages = document_loader(uploaded_doc)
    text_splitter = split_text()
    chunks = create_chunks(pages, text_splitter)
    embeddings, text_contents = create_embeddings(chunks)
    vector_store = create_vector_store(embeddings)

    st.session_state.vector_store = vector_store
    st.session_state.text_contents = text_contents

# Step 3: Ask Questions
if st.session_state.mode_selected and st.session_state.vector_store:
    user_input = st.chat_input("Ask me anything about the document 💬")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        context = retrieval(st.session_state.vector_store, user_input, st.session_state.text_contents)

        mode = st.session_state.mode
        if mode == "SRS Analysis":
            chat_completion_SRS_Analysis(context, user_input)
        elif mode == "Research Paper":
            chat_completion_Research_Analysis(context, user_input)
        elif mode == "GRC Document":
            chat_completion_GRC_Analysis(context, user_input)
        elif mode == "Whitepaper":
            chat_completion_Whitepaper_Analysis(context, user_input)
        elif mode == "Project Report":
            chat_completion_project_report_Analysis(context, user_input)





