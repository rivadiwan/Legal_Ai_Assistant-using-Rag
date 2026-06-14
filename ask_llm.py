import os
from groq import Groq
from bm25_retriever import *
from hybrid_retrieve import *
from reranker import rerank_results
from dotenv import load_dotenv
load_dotenv()

from retrieve import (
    load_model,
    load_faiss_index,
    load_chunks
)

# =========================
# GROQ API KEY

# =========================
client = Groq(
    api_key=os.getenv("API_KEY")
)

# =========================
# BUILD CONTEXT
# =========================
def build_context(results):

    context = ""
    max_chars = 12000

    for result in results:

        block = f"""

CASE NAME:
{result.get('case_name', '')}

YEAR:
{result.get('year', '')}

LEGAL TEXT:
{result.get('chunk_text', '')}

--------------------------------------------------

"""

        if len(context) + len(block) > max_chars:
            break

        context += block

    return context

# =========================
# ASK GROQ
# =========================

def ask_groq(query, context):

    prompt = f"""
You are an Indian Legal AI Assistant.

Use ONLY the legal context provided below.

Rules:

1. Do not invent facts.
2. If information is not present in the context, say:
   "The retrieved legal documents do not contain enough information to answer this question."
3. Mention case names whenever relevant.
4. Mention years whenever available.
5. Summarize judgments in simple language.
6. Explain legal sections clearly.
7. Explain verdicts clearly.
8. Answer in a professional and readable format.
9. Do not hallucinate.
10. Do not use outside knowledge.
11.If a fact is not explicitly stated in the context,do not infer it.
Do not assume outcomes.

Use only information directly available in the context.
LEGAL CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# =========================
# MAIN
# =========================

def main():

    print("\nLoading Legal RAG System...\n")

    print("Loading Embedding Model...")
    model = load_model()

    print("Loading FAISS Index...")
    index = load_faiss_index()

    print("Loading Chunks...")
    chunks = load_chunks()

    print("Building BM25...")
    bm25 = build_bm25(chunks)

    print("\nLegal RAG System Ready!")

    while True:

        query = input(
            "\nEnter Legal Query (or type exit): "
        )

        if query.lower() == "exit":

            print("\nExiting System...")
            break

        # =========================
        # HYBRID RETRIEVAL
        # =========================

        results = hybrid_search(
            query,
            model,
            index,
            chunks,
            bm25,
            top_k=10
        )

        # =========================
        # RERANKING
        # =========================

        if not (
            len(results) > 0
            and results[0].get("score") == "CASE_MATCH"
        ):
            results = rerank_results(
                query,
                results,
                top_k=5
            )


        # =========================
        # BUILD CONTEXT
        # =========================

        context = build_context(results)
        print("\n===== CONTEXT SENT TO LLM =====\n")
        print(context)

        # =========================
        # ASK GROQ
        # =========================

        answer = ask_groq(
            query,
            context
        )

        print("\n" + "=" * 60)
        print("LEGAL AI RESPONSE")
        print("=" * 60)
        print("\n")

        print(answer)


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    main()
    