import json
import argparse
import os
from pathlib import Path
from rag import rag
import ast
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="AIG RAG Evaluation Pipeline")

    parser.add_argument("--gt-path", type=str, default="./ground-truth/ground_truth.json",
                        help="The JSON file that contains ground truth data")
    
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid search balance: 0=pure BM25, 1=pure embedding")

    parser.add_argument("--dense-emb-path", type=str, default="./vector-database/config_text-embedding-3-small_2000_100/aig.parquet",
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

class Eval:
    def __init__(self, args):
    
        with open(args.gt_path, "r") as f:
            ground_truth = json.load(f)

        self.top_k = args.top_k
        self.comparison = {}

        for data in ground_truth:
            prompt_id = data["prompt_id"]
            query = data["prompt"]
            gt_answer = data["gt_answer"]
            gt_full_sentence = data["full_sentence"]

            query_args = argparse.Namespace(
                alpha=args.alpha,
                dense_emb_path=args.dense_emb_path,
                top_k=args.top_k,
                embedding_model=args.embedding_model,
                chat_model=args.chat_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,                
                query=query,
            )

            self.df = pd.read_parquet(args.dense_emb_path)
            
            result = rag(query_args)
            top_k_chunks = result["top_k_chunks"]
            pred_answer = result["pred_answer"]
            top_k_sections = result["top_k_sections"]

            self.comparison[prompt_id] = {
                "top_k_chunks": top_k_chunks,
                "top_k_sections": top_k_sections,
                "gt_answer": gt_answer,
                "gt_full_sentence": gt_full_sentence,
                "pred_answer": pred_answer
            }
            print()
            print("pred_answer: ", pred_answer)
            print("gt_answer: ", gt_answer)
            print()
            print("---------------------------------------")

    def recall_k(self):
        scores = []

        for prompt_id in self.comparison:
            comparison = self.comparison[prompt_id]
            gt_full_sentence = comparison["gt_full_sentence"]

            relevant_df = self.df[
                self.df["chunks"].apply(lambda x: gt_full_sentence in x)
            ]

            relevant_chunks = relevant_df["chunks"].tolist()
            top_k_chunks = comparison["top_k_chunks"]

            relevant_set = set(relevant_chunks)
            retrieved_set = set(top_k_chunks)

            if len(relevant_set) == 0:
                continue
            
            recall = len(relevant_set & retrieved_set) / len(relevant_set)
            scores.append(recall)

        return np.round(np.mean(scores), 3)

    def mrr(self):
        scores = []

        for prompt_id in self.comparison:
            
            gt_full_sentence = self.comparison[prompt_id]["gt_full_sentence"]
            top_k_chunks = self.comparison[prompt_id]["top_k_chunks"]

            rr = 0

            for i, chunk in enumerate(top_k_chunks):
                if gt_full_sentence in chunk:
                    rr = 1 / (i + 1)
                    break

            scores.append(rr)

        return np.round(np.mean(scores),3)

    def exact_match(self):
        correct = 0
        total = 0
        for prompt_id in self.comparison:

            gt_answer = self.comparison[prompt_id]["gt_answer"]
            pred_answer = self.comparison[prompt_id]["pred_answer"]

            if isinstance(gt_answer, list):
                try:
                    pred_parsed = ast.literal_eval(pred_answer.strip())
                    correct += set(map(tuple, gt_answer)) == set(map(tuple, pred_parsed))
                except:
                    correct += 0
            else:
                correct += gt_answer.lower().strip() == pred_answer.lower().strip()

            total += 1

        return np.round(correct / total, 3)

    def compute_scores(self):
        recall_k = self.recall_k()
        mrr = self.mrr()
        exact_match_score = self.exact_match()

        scores = {
            "retriever": {
                "recall_k": recall_k,
                "mrr": mrr
            },
            "generator": {
                "exact_match_score": exact_match_score
            },
        }
        return scores

if __name__ == "__main__":
    args = parse_args()
    evaluator = Eval(args)
    scores = evaluator.compute_scores()

    config_path = Path(f"eval/config_alpha-{args.alpha}_topk-{args.top_k}_temperature-{args.temperature}")
    
    os.makedirs(config_path, exist_ok=True)

    with open(config_path / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)
