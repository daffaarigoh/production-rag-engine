"""
Prompt templates with strict guardrails for the Generation module.
"""
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Strict guardrails to prevent hallucinations
SYSTEM_PROMPT_STR = """You are a highly capable AI Assistant for the Production RAG Engine.
Your task is to answer the user's question based strictly on the provided Context.

GUARDRAILS:
1. ONLY use information from the provided Context. Do NOT use outside knowledge.
2. If the answer is not contained in the Context, say exactly: "I cannot answer this based on the provided documents."
3. When you make a claim, cite the source by using the `chunk_id` provided in the metadata. Format citations as: [chunk_id].
4. Keep your answer clear, concise, and structured. Use Markdown for formatting.
5. Provide your response in the same language as the user's question (preferably Indonesian if asked in Indonesian).

Context:
{context}
"""

USER_PROMPT_STR = """Question: {question}"""

qa_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_STR),
    HumanMessagePromptTemplate.from_template(USER_PROMPT_STR)
])

def format_docs_for_context(docs) -> str:
    """Helper to format retrieved documents into the context string."""
    context_str = ""
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx}")
        content = doc.page_content.strip()
        context_str += f"--- Document [{chunk_id}] (Source: {source}) ---\n{content}\n\n"
    return context_str
