from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api = os.getenv("PINECONE_API")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    
)
pc = Pinecone(api_key=pinecone_api, environment="us-east-1-aws")
index_name = "finance-rag"
dense_index = pc.Index(index_name)
model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

def query_rag(user_query: str) -> str:
    query_embedding = model.encode([user_query])[0]

    results = dense_index.query(
        vector=query_embedding.tolist(),
        top_k=1,
        include_metadata=True,
        namespace=""
    )

    prompt_text = """
    You are a highly accurate and concise AI assistant.

    You are provided with:
    1. A user’s factual question.
    2. The full original retrieved text relevant to the question.
    3. A concise summary of that text.

    Your tasks:
    - Understand the user’s question carefully.
    - Use both the original text and its summary to extract exact factual information (e.g., dates, figures, names).
    - Resolve ambiguity if present using reasoning and cross-referencing.
    - Respond with a complete, precise sentence that directly answers the question.
    - Your answer must be naturally phrased — as a full, clear, and informative sentence — not just a fragment, number, or date.

    Output format:
    - A single complete and accurate sentence answering the question directly.
    - No preamble, explanation, repetition, or extra commentary.

    ---

    **Question:** {query}

    **Original Retrieved Text:** {text}

    **Summary of the Text:** {summary}

    ---

    Respond with one complete, factual sentence only.
    """


    prompt = PromptTemplate(
        input_variables=["query", "text", "summary"],
        template=prompt_text.strip()
    )

    output_parser = StrOutputParser()

    summarize_chain = (
        RunnableMap({
            "query": lambda x: x["query"],
            "text": lambda x: x["text"],
            "summary": lambda x: x["summary"]
        })
        | prompt
        | llm
        | output_parser
    )

    response = summarize_chain.invoke({
        "query": user_query,
        "text": results["matches"][0]["metadata"]["text"],
        "summary": results["matches"][0]["metadata"]["summary"]
    })

    return response