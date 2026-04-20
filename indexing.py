import datasets
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, rtrim, pandas_udf, explode, lit
from functools import reduce
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.types import ArrayType, StringType
from dotenv import load_dotenv
import os
from openai import OpenAI
import argparse
import json
from pathlib import Path

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser(description="AIG RAG Indexing Pipeline")

    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small",
                        help="OpenAI embedding model to use")

    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Chunk size for text splitting")

    parser.add_argument("--chunk-overlap", type=int, default=100,
                        help="Chunk overlap for text splitting")

    parser.add_argument("--parquet-dir", type=str, default="./parquet",
                        help="Directory to store parquet files")

    parser.add_argument("--output-dir", type=str, default="./vector-database",
                        help="Directory to store output CSV")

    return parser.parse_args()

    
@pandas_udf(ArrayType(StringType()))
def split_chunks(text: pd.Series, chunk_size: pd.Series, chunk_overlap: pd.Series) -> pd.Series:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size.iloc[0],
        chunk_overlap=chunk_overlap.iloc[0]
    )
    return text.apply(lambda t: text_splitter.split_text(t) if t else [])
    
def build_embed_partition(embedding_model, batch_size=8):
    def embed_partition(rows):
        client = OpenAI(api_key=API_KEY)
        batch = []

        for row in rows:
            batch.append((row.year, row.section, row.chunks))

            if len(batch) == batch_size:
                texts = [x[2] for x in batch]
                response = client.embeddings.create(model=embedding_model, input=texts)
                for (year, section, chunks), emb in zip(batch, response.data):
                    yield (year, section, chunks, emb.embedding)
                batch = []

        if batch:
            texts = [x[2] for x in batch]
            response = client.embeddings.create(model=embedding_model, input=texts)
            for (year, section, chunks), emb in zip(batch, response.data):
                yield (year, section, chunks, emb.embedding)

    return embed_partition

def index(args):
    embed_partition = build_embed_partition(args.embedding_model)

    spark = SparkSession.builder\
        .master('local[*]')\
        .appName('AIG_RAG')\
        .config('spark.driver.memory', '8g')\
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')\
        .config('spark.sql.shuffle.partitions', '8')\
        .getOrCreate()

    train_path = Path(f"{args.parquet_dir}/edgar-train.parquet")
    validation_path = Path(f"{args.parquet_dir}/edgar-validation.parquet")
    test_path = Path(f"{args.parquet_dir}/edgar-test.parquet")

    if not (train_path.is_file() and validation_path.is_file() and test_path.is_file()):
        raw_dataset = datasets.load_dataset("eloukas/edgar-corpus", "full")
        os.makedirs(args.parquet_dir, exist_ok=True)

        raw_dataset["train"].to_parquet(str(train_path))
        raw_dataset["validation"].to_parquet(str(validation_path))
        raw_dataset["test"].to_parquet(str(test_path))

    sdf_train = spark.read.parquet(str(train_path)).withColumn("split", lit("train"))
    sdf_validation = spark.read.parquet(str(validation_path)).withColumn("split", lit("validation"))
    sdf_test = spark.read.parquet(str(test_path)).withColumn("split", lit("test"))

    sdf = reduce(DataFrame.unionByName, [sdf_train, sdf_validation, sdf_test])

    output_path = Path(f"{args.output_dir}/config_{args.embedding_model}_{args.chunk_size}_{args.chunk_overlap}")
    os.makedirs(output_path, exist_ok=True)

    sdf.filter(col("cik") == 5272)\
        .drop("split", "filename")\
        .melt(
            ids=["cik", "year"],
            values=['section_1', 'section_1A', 'section_1B',
                    'section_2', 'section_3', 'section_4',
                    'section_5', 'section_6', 'section_7',
                    'section_7A', 'section_8', 'section_9',
                    'section_9A', 'section_9B', 'section_10',
                    'section_11', 'section_12', 'section_13',
                    'section_14', 'section_15'],
            variableColumnName="section",
            valueColumnName="description"
        )\
        .withColumn("description", rtrim(col("description"))) \
        .na.drop(subset=['description']).filter(col('description') != "") \
        .withColumn("chunks", split_chunks(
            col("description"), lit(args.chunk_size), lit(args.chunk_overlap))
        )\
        .withColumn("chunks", explode(col("chunks"))) \
        .drop("cik", "description")\
        .repartition(2)\
        .rdd.mapPartitions(embed_partition) \
        .toDF(["year", "section", "chunks", "embedding"])\
        .write.mode("overwrite").parquet(str(output_path / "aig.parquet"))

    info = {
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }

    with open(output_path / "info.json", "w") as f:
        json.dump(info, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    index(args)