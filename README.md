# 📄 DeepDocx - Your Intelligent Document Analyzer 🚀

**Public App:** https://deepdocx-e3lrtqh8nossgchrrd9v5o.streamlit.app/

DeepDocx is an AI-powered document understanding tool tailored for **software engineers**, **researchers**, **product/project managers**, **compliance analysts**, and **technical strategists**.

With support for PDFs and a suite of specialized **analysis modes**, DeepDocx provides instant, contextual insights from technical documents using cutting-edge LLMs (via Groq API + LLaMA 3.3).


## 💡 What Can DeepDocx Do?

Upload a document and ask questions like:
- _"What are the functional requirements?"_
- _"Summarize the key objectives of this paper."_
- _"List the user roles and modules."_
- _"What risks are outlined in this policy?"_

All powered by blazing-fast vector search + LLaMA 3.3 (via Groq).


## 🎯 Supported Analysis Modes

Choose from 5 powerful modes based on your document type:

### ✅ SRS Analysis
> Ideal for: Software Engineers, System Analysts  
- Extract **functional** & **non-functional** requirements  
- Identify **user roles**, **modules**, **data flows**  
- Highlight **incomplete/ambiguous** areas  
- Summarize **scope**, **objectives**, **tech terms**


### ✅ Research Paper Analysis
> Ideal for: Researchers, Students  
- Extract **objectives**, **methods**, **key findings**  
- Explain **abbreviations**, **concepts**  
- Point out **assumptions** and **future work**


### ✅ GRC Document Analysis
> Ideal for: Legal, Compliance, Security Professionals  
- Extract **responsibilities**, **restrictions**, **penalties**  
- Clarify **key terms** or **risks**  
- Offer **practical compliance insights**


### ✅ Whitepaper Analysis
> Ideal for: Tech Strategists, Investors  
- Summarize **problem, solution**, and **architecture**  
- Understand **tokenomics** or **financial models**  
- Assess **adoption plans** and **roadmaps**


### ✅ Project Report Analysis
> Ideal for: Project Managers, Team Leads  
- Extract **milestones**, **timelines**, **KPIs**  
- Identify **delays**, **risks**, and **assignments**  
- Provide **recommendations for improvement**


## 🧠 Tech Stack

- 🧩 **LangChain** + **Sentence Transformers** for chunking & embedding  
- 🧠 **LLaMA 3.3 70B** via **Groq API** for blazing-fast inference  
- 📄 **FAISS** for vector search  
- 💻 **Streamlit** for UI  
- 🧠 **MiniLM-L6-v2** for lightweight embeddings


## 🚀 Try It Live

📎 [Launch DeepDocx Now](https://your-public-url.com)  
_(Replace with your deployed Streamlit URL)_


## 🛠️ Local Installation

Clone the repo and run locally:

```bash
git clone https://github.com/yourusername/deepdocx.git
cd deepdocx
pip install -r requirements.txt

Create a .streamlit/secrets.toml file and add your Groq API key:

```bash
GROQ_API_KEY = "your_groq_api_key_here"

Then launch the app:

```bash
streamlit run app.py

## 📁 Example Use Cases

🤖 Auto-analyze complex Software Requirement Specifications

📚 Get quick summaries of academic papers

📜 Deconstruct legal, compliance, or governance documents

📈 Review whitepapers for projects or investments

🧩 Audit and break down project reports across teams


## 🧑‍💻 Ideal For

Software Engineers & Architects

Research Scholars & Students

Product & Project Managers

Legal, Compliance & Risk Officers

Startup Founders & Investors


## 📜 License
MIT License ©Ayesha Noman

