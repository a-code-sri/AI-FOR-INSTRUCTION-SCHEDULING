"""
train_model.py

Reads `data.csv` from the current working directory, extracts CFG-based
training samples for decision points (if/else-if/else/for), tokenizes and
pads inputs, trains a simple Embedding->LSTM->Dense binary classifier,
and saves artifacts into the current working directory:

  - cfg_lstm_model.h5        (TensorFlow model)
  - tokenizer.pkl            (Keras Tokenizer via pickle)
  - training_meta.pkl            (dict with maxlen, vocab_size, params)

Usage:
  python train_model.py --csv data.csv --n_samples 100 --epochs 8

All file I/O is relative to the current working directory.
"""
import re
import ast
import argparse
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
def build_cfg(code_str: str) -> Tuple[Dict[int, str], Dict[int, List[int]]]:
    """
    Build a simple control-flow-graph (CFG) from a C++-like code string.
    Returns (node_dict, edges) where node_dict[i] = node_text and edges is
    adjacency list mapping from node index -> list of indices.

    Node types are inferred from text contents (if(...), else if(...), else,
    for(...), and plain assignments).
    """
    lines = [ln.strip() for ln in code_str.splitlines() if ln.strip() != ""]

    items = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # --- if/else-if/else chain
        if ln.startswith("if(") or ln.startswith("else if(") or ln.startswith("else"):
            conds, bodies, has_else = [], [], False
            while i < len(lines) and (
                lines[i].startswith("if(")
                or lines[i].startswith("else if(")
                or lines[i].startswith("else")
            ):
                cur = lines[i]
                if cur.startswith("else") and not cur.startswith("else if("):
                    j = i + 1
                    if j < len(lines) and lines[j].startswith("{"):
                        j += 1
                    body = []
                    while j < len(lines) and "}" not in lines[j]:
                        body.append(lines[j])
                        j += 1
                    if j < len(lines) and "}" in lines[j]:
                        j += 1
                    bodies.append("\n".join(body) if body else "<EMPTY>")
                    has_else = True
                    i = j
                    break
                else:
                    conds.append(cur)
                    j = i + 1
                    if j < len(lines) and lines[j].startswith("{"):
                        j += 1
                    body = []
                    while j < len(lines) and "}" not in lines[j]:
                        body.append(lines[j])
                        j += 1
                    if j < len(lines) and "}" in lines[j]:
                        j += 1
                    bodies.append("\n".join(body) if body else "<EMPTY>")
                    i = j
            items.append(
                {
                    "type": "if_chain",
                    "conds": conds,
                    "bodies": bodies,
                    "has_else": has_else,
                }
            )
            continue

        # --- for loop
        if ln.startswith("for("):
            header = ln
            j = i + 1
            if j < len(lines) and lines[j].startswith("{"):
                j += 1
            body = []
            while j < len(lines) and "}" not in lines[j]:
                body.append(lines[j])
                j += 1
            if j < len(lines) and "}" in lines[j]:
                j += 1
            items.append(
                {
                    "type": "for",
                    "header": header,
                    "body": "\n".join(body) if body else "<EMPTY>",
                }
            )
            i = j
            continue

        # --- plain assignment
        items.append({"type": "assign", "text": ln})
        i += 1

    # convert items into nodes + edges
    nodes = []
    start_indices = []
    for it in items:
        start_indices.append(len(nodes))
        if it["type"] == "assign":
            nodes.append(it["text"])
        elif it["type"] == "for":
            nodes += [it["header"], it["body"]]
        elif it["type"] == "if_chain":
            for cond, body in zip(it["conds"], it["bodies"]):
                nodes += [cond, body]
            if it.get("has_else") and len(it["bodies"]) > len(it["conds"]):
                nodes.append(it["bodies"][-1])

    edges = {i: [] for i in range(len(nodes))}

    for idx, it in enumerate(items):
        start = start_indices[idx]
        exit_idx = (
            start_indices[idx + 1] if idx + 1 < len(start_indices) else len(nodes)
        )
        if it["type"] == "assign":
            if exit_idx < len(nodes):
                edges[start].append(exit_idx)
        elif it["type"] == "for":
            hdr, body = start, start + 1
            edges[hdr] = [body]
            if exit_idx < len(nodes):
                edges[hdr].append(exit_idx)
            edges[body] = [hdr]
        elif it["type"] == "if_chain":
            pairs = len(it["conds"])
            for k in range(pairs):
                cond, body = start + 2 * k, start + 2 * k + 1
                if k + 1 < pairs:
                    nxt = start + 2 * (k + 1)
                else:
                    nxt = start + 2 * pairs if it.get("has_else") else exit_idx
                edges[cond] = [body]
                if nxt < len(nodes):
                    edges[cond].append(nxt)
                if exit_idx < len(nodes):
                    edges[body].append(exit_idx)
            if it.get("has_else"):
                else_idx = start + 2 * pairs
                if exit_idx < len(nodes):
                    edges[else_idx].append(exit_idx)

    node_dict = {i: nodes[i] for i in range(len(nodes))}
    for k in edges:
        edges[k] = [x for x in edges[k] if x < len(nodes)]
    return node_dict, edges
def _safe_eval_condition(cond_text: str, ctx: Dict[str, int]) -> Optional[int]:
    """
    Evaluate simple boolean conditions like 'if(a < 3)' or 'a < 3'.
    We remove 'if' and parentheses if present. Replace variable names with
    integers from ctx and evaluate the comparison using ast parsing to avoid
    arbitrary code execution.
    Returns 1 (true), 0 (false), or None if we cannot evaluate.
    """
    try:
        t = cond_text.strip()
        if t.startswith("if"):
            t = t[2:].strip()
        if t.startswith("(") and t.endswith(")"):
            t = t[1:-1].strip()
        for var, val in ctx.items():
            t = re.sub(rf"\b{re.escape(var)}\b", str(int(val)), t)
        node = ast.parse(t, mode="eval")
        allowed = (
            ast.Expression,
            ast.Compare,
            ast.BinOp,
            ast.Num,
            ast.UnaryOp,
            ast.NameConstant,
            ast.Load,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.Lt,
            ast.Gt,
            ast.Eq,
            ast.LtE,
            ast.GtE,
            ast.NotEq,
            ast.BitAnd,
            ast.BitOr,
            ast.And,
            ast.Or,
            ast.BoolOp,
            ast.Constant,
        )
        for n in ast.walk(node):
            if not isinstance(n, allowed):
                return None
        val = eval(compile(node, "<string>", "eval"))
        return 1 if bool(val) else 0
    except Exception:
        return None
def _parse_for_iterations(header: str, ctx: Dict[str, int]) -> Optional[int]:
    """
    Parse very small for headers like 'for(int i=0; i<N; i++)' and return
    N (iterations) as integer when possible; otherwise return None.
    """
    try:
        inner = header[header.find("(") + 1 : header.rfind(")")]
        parts = [p.strip() for p in inner.split(";")]
        if len(parts) >= 2:
            cond = parts[1] 
            for var, val in ctx.items():
                cond = re.sub(rf"\b{re.escape(var)}\b", str(int(val)), cond)
            m = re.search(r"<\s*(-?\d+)", cond)
            if m:
                return int(m.group(1))
        return None
    except Exception:
        return None
def extract_training_df_from_dataset(
    csv_path: str, max_context_nodes: int = 20
) -> pd.DataFrame:
    """
    Read csv_path (expects column 'code_str'), build CFG per code sample,
    and produce a DataFrame with columns: ['input','label'] where 'input' is
    the concatenated code text of nodes visited up to the decision point and
    'label' is 1/0 depending on whether the model should enter the block.

    Strategy:
      - For each code snippet build node_dict, edges
      - Walk nodes linearly from index 0
      - Maintain a 'visited_nodes' list and assignment context parsed from them
      - When encountering a condition node (text starting with 'if(' or 'else if(')
        create one training sample per condition with 'input' = concat(visited)
        and 'label' = evaluation result (1/0) using current context
      - For else: label = 1 if none of previous if/elseif were true (we compute
        using their evaluations stored earlier)
      - For for header node: compute iterations using context; label = 1 if
        iterations>0 else 0

    This function re-computes labels from code rather than trusting CSV 'labels'.
    """
    df = pd.read_csv(csv_path)
    if "code_str" not in df.columns:
        raise ValueError("data.csv must contain 'code_str' column")

    records = []
    for idx, row in df.iterrows():
        code = str(row["code_str"])
        node_dict, edges = build_cfg(code)
        nodes = [node_dict[i] for i in range(len(node_dict))]

        visited = []
        # evaluations of if-chain conditions that precede an else
        last_if_chain_results = []

        i = 0
        steps = 0
        while i < len(nodes) and steps < max(200, len(nodes) * 4):
            steps += 1
            node = nodes[i]
            # if node is an 'if' or 'else if' condition
            if node.startswith("if(") or node.startswith("else if("):
                # input: concatenation of visited nodes (limit length)
                ctx = _extract_assignments_from_nodes(visited)
                inp_nodes = visited[-max_context_nodes:]
                inp = " || ".join(inp_nodes) if inp_nodes else "<EMPTY_CONTEXT>"
                # evaluate this condition
                val = _safe_eval_condition(node, ctx)
                if val is None:
                    # fallback: if we can't evaluate, create both variants by
                    # random labeling to avoid bias - but deterministic: 0
                    val = 0
                records.append({"input": inp, "label": int(val)})
                # store for else-chains: append this result
                last_if_chain_results.append(int(val))
                # move to body node (first successor) if true else to next
                # but for extraction we still traverse linear nodes; to avoid
                # complexity we'll continue linear scan and include bodies
                visited.append(node)
                i += 1
                continue

            # else node (body of else) - appears when previous if_chain had has_else
            if node.strip().startswith("else") or node.startswith("else"):
                ctx = _extract_assignments_from_nodes(visited)
                inp_nodes = visited[-max_context_nodes:]
                inp = " || ".join(inp_nodes) if inp_nodes else "<EMPTY_CONTEXT>"
                # else label: 1 if none of previous if/elseif evaluated true
                val = (
                    1
                    if (
                        len(last_if_chain_results) > 0
                        and sum(last_if_chain_results) == 0
                    )
                    else 0
                )
                records.append({"input": inp, "label": int(val)})
                visited.append(node)
                last_if_chain_results = []
                i += 1
                continue

            # for-loop header node
            if node.startswith("for("):
                ctx = _extract_assignments_from_nodes(visited)
                inp_nodes = visited[-max_context_nodes:]
                inp = " || ".join(inp_nodes) if inp_nodes else "<EMPTY_CONTEXT>"
                iters = _parse_for_iterations(node, ctx)
                val = 1 if (iters is not None and iters > 0) else 0
                records.append({"input": inp, "label": int(val)})
                visited.append(node)
                i += 1
                continue

            # plain body or assignment
            visited.append(node)
            i += 1

        # end sample
    out = pd.DataFrame.from_records(records)
    if out.empty:
        print("Warning: no training records were extracted from dataset.")
    return out
def tokenize_and_pad(
    texts: List[str],
    tokenizer: Optional[Tokenizer] = None,
    maxlen: Optional[int] = None,
) -> Tuple[np.ndarray, Tokenizer, int]:
    """
    Fit a Keras Tokenizer (if tokenizer is None) on the texts, convert to
    integer sequences and pad them using front (pre) zero-padding. Returns
    (padded_sequences, tokenizer, maxlen_used).
    """
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    if maxlen is None:
        maxlen = max(1, max(len(s) for s in seqs))
    padded = pad_sequences(seqs, maxlen=maxlen, padding="pre")
    return padded, tokenizer, maxlen
def build_and_train_model(
    X: np.ndarray,
    y: np.ndarray,
    vocab_size: int,
    embed_dim: int = 64,
    lstm_units: int = 64,
    epochs: int = 8,
    batch_size: int = 32,
    save_prefix: str = "cfg_lstm",
) -> Dict[str, Any]:
    """
    Build a simple sequential model: Embedding -> LSTM -> Dense(sigmoid)
    Train it and save model + metadata into current working directory.

    Returns training metadata dict.
    """
    maxlen = X.shape[1]
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size + 1, output_dim=embed_dim, input_length=maxlen
            ),
            LSTM(lstm_units),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        X, y, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es]
    )

    model_path = f"{save_prefix}_model.h5"
    tokenizer_path = f"{save_prefix}_tokenizer.pkl"
    meta_path = f"{save_prefix}_meta.pkl"

    model.save(model_path)

    meta = {
        "maxlen": int(maxlen),
        "vocab_size": int(vocab_size),
        "embed_dim": int(embed_dim),
        "lstm_units": int(lstm_units),
        "model_path": model_path,
        "meta_path": meta_path,
        "tokenizer_path": tokenizer_path,
    }

    # save meta later by caller when tokenizer exists
    return {"model": model, "history": history.history, "meta": meta}
def save_tokenizer(tokenizer: Tokenizer, path: str):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def save_meta(meta: Dict[str, Any], path: str):
    with open(path, "wb") as f:
        pickle.dump(meta, f)
def train_from_csv(
    csv_path: str,
    save_prefix: str = "cfg_lstm",
    epochs: int = 8,
    embed_dim: int = 64,
    lstm_units: int = 64,
):
    """
    High-level function that reads CSV, extracts training records, tokenizes,
    trains the model and saves (model, tokenizer, meta) into current directory
    with filenames using save_prefix.
    """
    print(f"Reading dataset from {csv_path} ...")
    df = extract_training_df_from_dataset(csv_path)
    if df.empty:
        raise RuntimeError("No training records were extracted — check data.csv format")

    texts = df["input"].astype(str).tolist()
    labels = df["label"].astype(int).values

    X, tokenizer, maxlen = None, None, None
    X, tokenizer, maxlen = tokenize_and_pad(texts, tokenizer=None, maxlen=None)
    vocab_size = len(tokenizer.word_index)

    # build and train
    print("Training model...")
    result = build_and_train_model(
        X,
        labels,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        lstm_units=lstm_units,
        epochs=epochs,
        save_prefix=save_prefix,
    )
    model = result["model"]
    meta = result["meta"]
    # save tokenizer and meta
    tok_path = f"{save_prefix}_tokenizer.pkl"
    meta_path = f"{save_prefix}_meta.pkl"
    save_tokenizer(tokenizer, tok_path)
    # extend meta with additional fields
    meta.update({"tokenizer_path": tok_path, "maxlen": int(maxlen)})
    save_meta(meta, meta_path)
    print("Saved: model ->", meta["model_path"])
    print("Saved: tokenizer ->", tok_path)
    print("Saved: meta ->", meta_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--save_prefix", type=str, default="cfg_lstm")
    args, unknown = parser.parse_known_args()
    train_from_csv(
        args.csv,
        save_prefix=args.save_prefix,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
        lstm_units=args.lstm_units,
    )