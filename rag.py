import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from models import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def parse_args():
    parser = argparse.ArgumentParser(description="AIG RAG Query Pipeline")

    parser.add_argument("--query", type=str, required=True,
                        help="Query to run against the RAG pipeline")
    
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid search balance: 0=pure BM25, 1=pure embedding")    

    parser.add_argument("--dense-emb-path", type=str, default="vector-database/config_text-embedding-3-small_2000_100/aig.parquet",
                        help="The path for dense embedding vectors")   
    
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top chunks to retrieve")

    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small",
                        help="OpenAI embedding model to use")
    
    parser.add_argument("--chat-model", type=str, default="gpt-4o-mini",
                        help="OpenAI chat model to use")
    
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature (0.0 - 2.0)")
    
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Max tokens for LLM response")
                
    return parser.parse_args()


def log_step(msg, t0):
    print(f"[{time.time() - t0:.2f}s] {msg}")

def cosine_similarity_batch(query_embedding, embeddings):
    query = np.array(query_embedding).reshape(1, -1)
    emb = np.array(embeddings)

    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    return np.dot(emb_norm, query_norm.T).flatten()

def bm25_search(query, df, top_k=10):
    tokenized_chunks = [chunk.lower().split() for chunk in df["chunks"].tolist()]
    bm25 = BM25Okapi(tokenized_chunks)
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    scores = pd.Series(scores, index=df.index).nlargest(top_k)
    return scores

def hybrid_search(query, query_embedding, df, embeddings, top_k=10, alpha=0.5):
    emb_scores = pd.Series(
        cosine_similarity_batch(query_embedding, embeddings), index=df.index
    )
    bm25_scores = bm25_search(query, df, top_k=len(df))
    
    emb_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min())
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    
    hybrid_scores = alpha * emb_norm + (1 - alpha) * bm25_norm
    
    return hybrid_scores.nlargest(top_k)

def rag(args):
    t0 = time.time()
    log_step("Start", t0)

    model = ChatOpenAI(
        model_name=args.chat_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        streaming=True
    )

    structured_model = model.with_structured_output(YearExtraction)
    result = structured_model.invoke([
        SystemMessage("Extract four digit years from the user's query."),
        HumanMessage(args.query),
    ])
    
    years = result.years 

    log_step(f"Years extracted: {years}", t0)
    
    def create_query_embedding(args, client):
        response = client.embeddings.create(
            model=args.embedding_model,
            input=args.query
        )
        return response.data[0].embedding

    def load_embeddings(args, years):
        df = pd.read_parquet(args.dense_emb_path)
        filtered_df = df if not years else df[df["year"].isin(years)]
        embeddings = np.stack(filtered_df["embedding"].values)
        return filtered_df, embeddings

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_query = executor.submit(create_query_embedding, args, client)
        future_data = executor.submit(load_embeddings, args, years)

        query_embedding = future_query.result()
        df_aig_final, embeddings = future_data.result()

    log_step("Query embedding and data loaded", t0)
    
    top_k_indices = hybrid_search(
        args.query, 
        query_embedding, 
        df_aig_final, 
        embeddings, 
        top_k=args.top_k,
        alpha=args.alpha
    )
    log_step("Hybrid search finished", t0)

    df_top_chunks = df_aig_final.loc[top_k_indices.index]

    top_k_years = df_top_chunks["year"].to_list()
    top_k_sections = df_top_chunks["section"].to_list()
    top_k_chunks = df_top_chunks["chunks"].to_list()

    context = "\n".join(df_top_chunks["chunks"])
    log_step("Context built", t0)

    response = model.invoke([
        SystemMessage(f"Please answer the user query based on the given context {context}"),
        HumanMessage(args.query)
    ])

    log_step("Final LLM response", t0)

    pred_answer = response.content

    return {
        "pred_answer": pred_answer,
        "top_k_chunks": top_k_chunks,
        "top_k_sections": top_k_sections,
        "top_k_years": top_k_years
    }

if __name__ == "__main__":
    args = parse_args()
    result = rag(args)
    print(result["pred_answer"])

# if __name__ == "__main__":
#     args = parse_args()

#     print("AIG RAG Chat\n")
#     while True:
#         try:
#             query = input("You: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print("\nBye!")
#             break

#         if not query:
#             continue
#         if query.lower() in {"exit", "quit", "q"}:
#             print("Bye!")
#             break

#         args.query = query 
#         try:
#             result = rag(args)
#             print(f"\nAnswer: {result['pred_answer']}\n")
#         except Exception as e:
#             print(f"[error] {e}\n")