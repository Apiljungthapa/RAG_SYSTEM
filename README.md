# RAG System Project

A Retrieval-Augmented Generation (RAG) system that combines multiple text and tables for enhanced information retrieval and generation.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- Virtualenv (recommended)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Apiljungthapa/RAG_SYSTEM.git
cd RAG_SYSTEM
```
### 2. Create and Activate Virtual Environment
```bash
python -m venv rag_env
source rag_env/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
## Create a .env file in the project root directory and add your API keys (you'll need to obtain these from the respective services):
```bash
GEMINI_API_KEY="YOUR GEMINI_API_KEY"
LANGCHAIN_API_KEY="YOUR LANGCHAIN_API_KEY"
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY="YOUR LANGSMITH_API_KEY"
LANGSMITH_PROJECT="MULTI RAG"
PINECONE_API="YOUR PINECONE_API_KEY"
```

### 5. Explore the Project Files
#### multimodel_rag.ipynb: Jupyter notebook with the multi-modal RAG implementation

#### ApiLThapa_Project_Explanation.pdf: Project documentation and explanation

#### requirements.txt: Python dependencies

#### .env: Environment variables (template provided, you need to add your keys)

### 6.  Run the Streamlit UI Application
```bash
streamlit run app.py```
