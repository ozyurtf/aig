# Running Instructions 

```bash
python3 -m venv aig
```

```bash
source aig/bin/activate
```

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt 
``` 

```bash
python indexing.py \
    --embedding-model text-embedding-3-small \
    --chunk-size 2000 \
    --chunk-overlap 100 \
    --parquet-dir ./parquet \
    --output-dir ./vector-database
```

```bash
python rag.py \
    --query "How much collateral did AIG hold at December 31, 2015?" \
    --alpha 0.0 \
    --dense-emb-path ./vector-database/config_text-embedding-3-small_2000_100/aig.parquet \
    --top-k 10 \
    --embedding-model text-embedding-3-small \
    --chat-model gpt-4o-mini \
    --temperature 0.0 \
    --max-tokens 500
```

```bash
python evaluate.py \
    --gt-path ./ground-truth/ground_truth.json \
    --alpha 0.0 \
    --dense-emb-path ./vector-database/config_text-embedding-3-small_2000_100/aig.parquet \
    --top-k 10 \
    --embedding-model text-embedding-3-small \
    --chat-model gpt-4o-mini \
    --temperature 0.0 \
    --max-tokens 500
```