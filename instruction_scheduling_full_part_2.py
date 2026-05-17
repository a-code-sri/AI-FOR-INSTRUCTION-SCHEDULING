import pandas as pd
import random
import csv
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------
# Step 1: Personality-based code generation
# ---------------------------------------------------------------------


def _rand_var():
    """Return a random single-letter variable name."""
    return random.choice([chr(c) for c in range(ord("a"), ord("z") + 1)])


def _assign_statement(var: str = None, val: int = None) -> str:
    """Generate a single assignment statement like 'x = 4'."""
    if var is None:
        var = _rand_var()
    if val is None:
        val = random.randint(0, 5)
    return f"{var} = {val}"


def _if_block_chain(max_chain=3) -> Tuple[str, list[int]]:
    """
    Generate a random if–else-if–else block chain and corresponding labels.
    Labels represent whether each condition evaluates True (1) or False (0).
    """
    var = _rand_var()
    val = random.randint(0, 5)
    pre = f"{var} = {val}\n"
    chain_len = random.randint(1, max_chain)

    parts, labels = [], []
    for i in range(chain_len):
        c = random.randint(0, 5)
        op = random.choice(["<", ">", "==", "<=", ">="])
        cond = f"if({var} {op} {c})" if i == 0 else f"else if({var} {op} {c})"
        expr = f"{val} {op} {c}"
        try:
            label = 1 if eval(expr) else 0
        except Exception:
            label = 0
        labels.append(label)
        parts.append(cond + " {\n    " + _assign_statement() + "\n}")
    if random.choice([True, False]):
        parts.append("else {\n    " + _assign_statement() + "\n}")
    return pre + "\n".join(parts), labels


def _for_block() -> Tuple[str, int]:
    """Generate a small for-loop block and a binary label (loop executed or not)."""
    iter_count = random.randint(0, 3)
    loop = f"for(int i=0; i<{iter_count}; i++) {{\n    {_assign_statement()}\n}}"
    label = 1 if iter_count > 0 else 0
    return loop, label


# ---------------------------------------------------------------------
# Step 2: Personalities — define 4 programmer styles
# ---------------------------------------------------------------------


def _generate_personality_code(
    personality: str, min_statements=3, max_statements=7
) -> Tuple[str, list[int]]:
    """
    Generate code influenced by a specific 'personality' (style).
    Each personality has its own statistical preferences.
    """
    stmts, labels = [], []

    if personality == "A":  # Compact coder
        choices = ["assign"] * 5 + ["if"] * 2 + ["for"]
    elif personality == "B":  # Verbose coder
        choices = ["assign"] * 3 + ["if"] * 3 + ["for"] * 2
    elif personality == "C":  # Experimental coder
        choices = ["assign"] * 2 + ["if"] * 3 + ["for"] * 3
    else:  # D — Structured coder (balanced)
        choices = ["assign"] * 4 + ["if"] * 2 + ["for"] * 2

    stmt_count = random.randint(min_statements, max_statements)
    for _ in range(stmt_count):
        choice = random.choice(choices)
        if choice == "assign":
            stmts.append(_assign_statement())
        elif choice == "if":
            chain, l = _if_block_chain()
            stmts.append(chain)
            labels.extend(l)
        else:
            loop, l = _for_block()
            stmts.append(loop)
            labels.append(l)

    stmts.append("END")  # terminal marker
    return "\n".join(stmts), labels


def generate_codes(
    n: int = 100, personalities: List[str] = ["A", "B", "C", "D"]
) -> Tuple[List[str], List[list[int]]]:
    """
    Generate 'n' random code samples by mixing multiple programmer personalities.
    Returns lists of (codes, labels).
    """
    codes, labels_list = [], []
    for _ in range(n):
        personality = random.choice(personalities)
        code, labels = _generate_personality_code(personality)
        codes.append(code)
        labels_list.append(labels)
    return codes, labels_list


# ---------------------------------------------------------------------
# Step 3: Control Flow Graph (CFG) Builder
# ---------------------------------------------------------------------


def build_cfg(code_str: str) -> Tuple[Dict[int, str], Dict[int, list[int]]]:
    """
    Build a simple control-flow-graph (CFG) from a C++-like code string.

    Each assignment or control structure (if-chain, for-loop) is a node.
    The adjacency list (edges) represents possible control transfers.
    """
    lines = [ln.strip() for ln in code_str.splitlines() if ln.strip() != ""]

    items = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # Handle if/else-if/else chains
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

        # Handle for-loops
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

        # Plain assignment
        items.append({"type": "assign", "text": ln})
        i += 1

    # Convert items → nodes and edges
    nodes, start_indices = [], []
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

    # Edge creation logic (main control flow)
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
            edges[hdr] = [body, exit_idx] if exit_idx < len(nodes) else [body]
            edges[body] = [hdr]  # loop back

        elif it["type"] == "if_chain":
            pairs = len(it["conds"])
            for k in range(pairs):
                cond, body = start + 2 * k, start + 2 * k + 1
                nxt = (
                    start + 2 * (k + 1)
                    if k + 1 < pairs
                    else (start + 2 * pairs if it.get("has_else") else exit_idx)
                )
                edges[cond] = [body, nxt] if nxt < len(nodes) else [body]
                edges[body] = [exit_idx] if exit_idx < len(nodes) else []
            if it.get("has_else"):
                else_idx = start + 2 * pairs
                edges[else_idx] = [exit_idx] if exit_idx < len(nodes) else []

    node_dict = {i: nodes[i] for i in range(len(nodes))}
    for k in edges:
        edges[k] = [x for x in edges[k] if x < len(nodes)]
    return node_dict, edges


# ---------------------------------------------------------------------
# Step 4: Dataset creation — saving to CSV
# ---------------------------------------------------------------------


def save_dataset_to_csv(
    codes: List[str], labels_list: List[list[int]], filename="data.csv"
):
    """
    Save generated dataset to CSV.
    Each row: [code_str, labels]
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["code_str", "labels"])
        for code, labels in zip(codes, labels_list):
            writer.writerow([code, labels])
    print(f"✅ Dataset saved to {filename} (total {len(codes)} samples)")


# ---------------------------------------------------------------------
# Step 5: Main demo — end-to-end generation and saving
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Generate 100 random samples across 4 programmer personalities
    codes, labels_list = generate_codes(n=100)

    # Build CFG for the first sample (demo only)
    sample_code = codes[0]
    node_dict, adj_list = build_cfg(sample_code)

    print("Sample Generated Code:\n----------------------")
    print(sample_code, "\n")

    print("Node Dictionary:")
    for k, v in node_dict.items():
        print(f"  {k}: {v}")
    print("\nAdjacency List:")
    for k, v in adj_list.items():
        print(f"  {k} -> {v}")

    # Save dataset
    save_dataset_to_csv(codes, labels_list)

# -----------------------------
# File: train_model.py
# -----------------------------
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

# -----------------------------
# --- CFG builder (same logic)
# -----------------------------


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


# -----------------------------
# --- Safe expression evaluator
# -----------------------------


def _extract_assignments_from_nodes(nodes: List[str]) -> Dict[str, int]:
    """
    Parse assignment statements in node strings and return a simple context
    mapping variable -> integer value for those assignments that are simple
    integer assignments like 'a = 3' or 'x=5'. Ignore complex expressions.
    """
    ctx = {}
    assign_re = re.compile(r"\b([a-zA-Z_]\w*)\s*=\s*(-?\d+)\b")
    for n in nodes:
        for m in assign_re.finditer(n):
            var, val = m.group(1), int(m.group(2))
            ctx[var] = val
    return ctx


def _safe_eval_condition(cond_text: str, ctx: Dict[str, int]) -> Optional[int]:
    """
    Evaluate simple boolean conditions like 'if(a < 3)' or 'a < 3'.
    We remove 'if' and parentheses if present. Replace variable names with
    integers from ctx and evaluate the comparison using ast parsing to avoid
    arbitrary code execution.
    Returns 1 (true), 0 (false), or None if we cannot evaluate.
    """
    try:
        # strip 'if' and parentheses
        t = cond_text.strip()
        if t.startswith("if"):
            t = t[2:].strip()
        if t.startswith("(") and t.endswith(")"):
            t = t[1:-1].strip()
        # replace variable names with their numeric values
        # build a safe AST expression
        for var, val in ctx.items():
            # use word boundary replace
            t = re.sub(rf"\b{re.escape(var)}\b", str(int(val)), t)

        # Only allow comparison and numeric literals; parse AST and check nodes
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
        # find the middle clause between first and second semicolons
        inner = header[header.find("(") + 1 : header.rfind(")")]
        parts = [p.strip() for p in inner.split(";")]
        if len(parts) >= 2:
            cond = parts[1]  # usually 'i < N' or similar
            # replace context variables
            for var, val in ctx.items():
                cond = re.sub(rf"\b{re.escape(var)}\b", str(int(val)), cond)
            # find integer after '<' or '<=' etc
            m = re.search(r"<\s*(-?\d+)", cond)
            if m:
                return int(m.group(1))
        return None
    except Exception:
        return None


# -----------------------------
# --- Step 3: Extract training pairs from CFGs
# -----------------------------


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


# -----------------------------
# --- Step 4: Tokenize & Pad
# -----------------------------


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


# -----------------------------
# --- Step 5: Build and train LSTM
# -----------------------------


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


# -----------------------------
# --- Save helpers
# -----------------------------


def save_tokenizer(tokenizer: Tokenizer, path: str):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def save_meta(meta: Dict[str, Any], path: str):
    with open(path, "wb") as f:
        pickle.dump(meta, f)


# -----------------------------
# --- End-to-end train function
# -----------------------------


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


# -----------------------------
# --- CLI
# -----------------------------
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
import os
from instruction_scheduling_full_part_2 import * # -----------------------------
# File: predict_flow.py
# -----------------------------
"""
predict_flow.py

Loads saved model/tokenizer/meta from current working directory and predicts
the execution order of nodes for a given input code string. If no code file
is provided it will pick the first sample from data.csv (if present).

Usage:
  python predict_flow.py --code mycode.txt
  python predict_flow.py              # uses data.csv first sample as default

Outputs printed predicted node execution sequence.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# reuse build_cfg, _extract_assignments_from_nodes, _safe_eval_condition,
# _parse_for_iterations functions from train_model. For simplicity we import
# by copying logic here (so predict_flow.py is standalone).

# ---- copy build_cfg and helpers (identical to train_model) ----

import re
import ast

def build_cfg(code_str: str):
    lines = [ln.strip() for ln in code_str.splitlines() if ln.strip() != ""]
    items = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("if(") or ln.startswith("else if(") or ln.startswith("else"):
            conds, bodies, has_else = [], [], False
            while i < len(lines) and (lines[i].startswith("if(")
                                      or lines[i].startswith("else if(")
                                      or lines[i].startswith("else")):
                cur = lines[i]
                if cur.startswith("else") and not cur.startswith("else if("):
                    j = i + 1
                    if j < len(lines) and lines[j].startswith("{"): j += 1
                    body = []
                    while j < len(lines) and '}' not in lines[j]:
                        body.append(lines[j]); j += 1
                    if j < len(lines) and '}' in lines[j]: j += 1
                    bodies.append("\n".join(body) if body else "<EMPTY>")
                    has_else = True
                    i = j; break
                else:
                    conds.append(cur)
                    j = i + 1
                    if j < len(lines) and lines[j].startswith("{"): j += 1
                    body = []
                    while j < len(lines) and '}' not in lines[j]:
                        body.append(lines[j]); j += 1
                    if j < len(lines) and '}' in lines[j]: j += 1
                    bodies.append("\n".join(body) if body else "<EMPTY>")
                    i = j
            items.append({"type": "if_chain", "conds": conds,
                          "bodies": bodies, "has_else": has_else})
            continue
        if ln.startswith("for("):
            header = ln
            j = i + 1
            if j < len(lines) and lines[j].startswith("{"): j += 1
            body = []
            while j < len(lines) and '}' not in lines[j]:
                body.append(lines[j]); j += 1
            if j < len(lines) and '}' in lines[j]: j += 1
            items.append({"type": "for", "header": header,
                          "body": "\n".join(body) if body else "<EMPTY>"})
            i = j; continue
        items.append({"type": "assign", "text": ln})
        i += 1
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
        exit_idx = start_indices[idx + 1] if idx + 1 < len(start_indices) else len(nodes)
        if it["type"] == "assign":
            if exit_idx < len(nodes):
                edges[start].append(exit_idx)
        elif it["type"] == "for":
            hdr, body = start, start + 1
            edges[hdr] = [body]
            if exit_idx < len(nodes): edges[hdr].append(exit_idx)
            edges[body] = [hdr]
        elif it["type"] == "if_chain":
            pairs = len(it["conds"])
            for k in range(pairs):
                cond, body = start + 2*k, start + 2*k + 1
                if k + 1 < pairs:
                    nxt = start + 2*(k+1)
                else:
                    nxt = start + 2*pairs if it.get("has_else") else exit_idx
                edges[cond] = [body]
                if nxt < len(nodes): edges[cond].append(nxt)
                if exit_idx < len(nodes): edges[body].append(exit_idx)
            if it.get("has_else"):
                else_idx = start + 2*pairs
                if exit_idx < len(nodes): edges[else_idx].append(exit_idx)
    node_dict = {i: nodes[i] for i in range(len(nodes))}
    for k in edges:
        edges[k] = [x for x in edges[k] if x < len(nodes)]
    return node_dict, edges


def _extract_assignments_from_nodes(nodes: List[str]) -> Dict[str, int]:
    ctx = {}
    assign_re = re.compile(r"\b([a-zA-Z_]\w*)\s*=\s*(-?\d+)\b")
    for n in nodes:
        for m in assign_re.finditer(n):
            var, val = m.group(1), int(m.group(2))
            ctx[var] = val
    return ctx


def _safe_eval_condition(cond_text: str, ctx: Dict[str, int]) -> Optional[int]:
    try:
        t = cond_text.strip()
        if t.startswith('if'):
            t = t[2:].strip()
        if t.startswith('(') and t.endswith(')'):
            t = t[1:-1].strip()
        for var, val in ctx.items():
            t = re.sub(rf"\b{re.escape(var)}\b", str(int(val)), t)
        node = ast.parse(t, mode='eval')
        allowed = (ast.Expression, ast.Compare, ast.BinOp, ast.Num, ast.UnaryOp,
                   ast.NameConstant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
                   ast.Mod, ast.Pow, ast.Lt, ast.Gt, ast.Eq, ast.LtE, ast.GtE,
                   ast.NotEq, ast.BitAnd, ast.BitOr, ast.And, ast.Or, ast.BoolOp,
                   ast.Constant)
        for n in ast.walk(node):
            if not isinstance(n, allowed):
                return None
        val = eval(compile(node, '<string>', 'eval'))
        return 1 if bool(val) else 0
    except Exception:
        return None


def _parse_for_iterations(header: str, ctx: Dict[str, int]) -> Optional[int]:
    """
    Parse very small for headers like 'for(int i=0; i<N; i++)' and return
    N (iterations) as integer when possible; otherwise return None.
    """
    try:
        # find the middle clause between first and second semicolons
        inner = header[header.find('(')+1:header.rfind(')')]
        parts = [p.strip() for p in inner.split(';')]
        if len(parts) >= 2:
            cond = parts[1]  # usually 'i < N' or similar
            # replace context variables
            for var, val in ctx.items():
                cond = re.sub(rf"\b{re.escape(var)}\b", str(int(val)), cond)
            # find integer after '<' or '<=' etc
            m = re.search(r"<\s*(-?\d+)", cond)
            if m:
                return int(m.group(1))
        return None
    except Exception:
        return None

# -----------------------------
# --- Prediction function
# -----------------------------

def predict_flow_for_code(code_str: str, model, tokenizer, meta: Dict[str, Any], prob_thresh: float = 0.5, loop_threshold: int = 3) -> List[Tuple[int, str]]:
    """
    Given a code string, build its CFG and use the trained LSTM model to
    decide whether to enter conditional/loop bodies. Returns an ordered list
    of visited nodes: list of tuples (node_index, node_text).

    Rules:
      - For each condition node (if/else if) create input string as the
        concatenation of visited nodes (same as training) and ask model.
      - If probability >= prob_thresh -> treat as 'enter' else 'skip'.
      - For for-loops: if predicted enter, iterate body up to loop_threshold times
        (safeguard) and then exit.
    """
    node_dict, edges = build_cfg(code_str)
    nodes = [node_dict[i] for i in range(len(node_dict))]

    visited_nodes = []
    execution_path = []  # (idx, text)

    i = 0
    steps = 0
    max_steps = max(500, len(nodes)*10)
    while i < len(nodes) and steps < max_steps:
        steps += 1
        node = nodes[i]
        # condition nodes
        if node.startswith('if(') or node.startswith('else if('):
            inp_nodes = visited_nodes
            inp = ' || '.join(inp_nodes[-40:]) if inp_nodes else '<EMPTY_CONTEXT>'
            seq = tokenizer.texts_to_sequences([inp])
            padded = pad_sequences(seq, maxlen=meta.get('maxlen', 40), padding='pre')
            prob = float(model.predict(padded, verbose=0)[0,0])
            take = prob >= prob_thresh
            # if take -> we go to body (first successor), else go to next cond/exit
            succs = edges.get(i, [])
            if take and len(succs) >= 1:
                # go into body
                body_idx = succs[0]
                execution_path.append((i, node))
                # append body node itself and simulate simple visiting
                execution_path.append((body_idx, nodes[body_idx]))
                # update visited_nodes
                visited_nodes.append(node); visited_nodes.append(nodes[body_idx])
                # after body we go to exit (we rely on CFG edges from body)
                # find next after body
                next_nodes = edges.get(body_idx, [])
                if next_nodes:
                    i = next_nodes[0]
                else:
                    i += 1
                continue
            else:
                # skip to next node as per CFG (second successor) or next index
                execution_path.append((i, node + f"  # predicted_skip(prob={prob:.3f})"))
                visited_nodes.append(node)
                succs = edges.get(i, [])
                if len(succs) >= 2:
                    i = succs[1]
                else:
                    i += 1
                continue

        # else node
        if node.strip().startswith('else') or node.startswith('else'):
            # else is reached when previous conditions were false; model was not
            # asked directly for else during training but we created samples for else.
            inp_nodes = visited_nodes
            inp = ' || '.join(inp_nodes[-40:]) if inp_nodes else '<EMPTY_CONTEXT>'
            seq = tokenizer.texts_to_sequences([inp])
            padded = pad_sequences(seq, maxlen=meta.get('maxlen', 40), padding='pre')
            prob = float(model.predict(padded, verbose=0)[0,0])
            # we use prob to decide entering else body
            succs = edges.get(i, [])
            if len(succs) >= 1:
                body_idx = succs[0]
                if prob >= prob_thresh:
                    execution_path.append((i, node))
                    execution_path.append((body_idx, nodes[body_idx]))
                    visited_nodes.append(node); visited_nodes.append(nodes[body_idx])
                    i = edges.get(body_idx, [body_idx+1])[0]
                    continue
                else:
                    execution_path.append((i, node + f"  # predicted_skip_else(prob={prob:.3f})"))
                    visited_nodes.append(node)
                    i = i + 1
                    continue
            else:
                execution_path.append((i, node))
                i += 1
                continue


        # for-loops
        if node.startswith('for('):
            inp_nodes = visited_nodes
            inp = ' || '.join(inp_nodes[-40:]) if inp_nodes else '<EMPTY_CONTEXT>'
            seq = tokenizer.texts_to_sequences([inp])
            padded = pad_sequences(seq, maxlen=meta.get('maxlen', 40), padding='pre')
            prob = float(model.predict(padded, verbose=0)[0,0])
            succs = edges.get(i, [])
            enter = prob >= prob_thresh
            if enter and len(succs) >= 1:
                body_idx = succs[0]
                # simulate loop with threshold
                iterations = 0
                while iterations < loop_threshold:
                    execution_path.append((i, node + f"  # loop_iter({iterations})"))
                    execution_path.append((body_idx, nodes[body_idx]))
                    visited_nodes.append(node); visited_nodes.append(nodes[body_idx])
                    iterations += 1
                # after threshold exit to successor after header if exists
                if len(succs) >= 2:
                    i = succs[1]
                else:
                    i += 1
                continue
            else:
                # skip loop
                execution_path.append((i, node + f"  # predicted_skip_loop(prob={prob:.3f})"))
                visited_nodes.append(node)
                # move to exit successor if present
                succs = edges.get(i, [])
                if len(succs) >= 2:
                    i = succs[1]
                else:
                    i += 1
                continue

        # plain assignment or body
        execution_path.append((i, node))
        visited_nodes.append(node)
        # default move to next via edges or linear index
        succs = edges.get(i, [])
        if succs:
            i = succs[0]
        else:
            i += 1


    return execution_path

# -----------------------------
# --- CLI for predict_flow.py
# -----------------------------
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Predict CFG execution flow using trained LSTM model (reads from CWD)')
    # parser.add_argument('--code', type=str, default=None, help='Path to file containing code string; if omitted will use first sample in data.csv')
    # parser.add_argument('--model_prefix', type=str, default='cfg_lstm', help='Prefix used when saving model/tokenizer/meta during training')
    # parser.add_argument('--prob_thresh', type=float, default=0.5, help='Probability threshold to decide entering block')
    # parser.add_argument('--loop_threshold', type=int, default=3, help='Maximum simulated loop iterations during prediction')
    # args = parser.parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lstm_units', type=int, default=32)
    parser.add_argument('--save_prefix', type=str, default='cfg_lstm_model')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--code', type=str, default=None)

    # Important: parse known args, ignore unknown (like Jupyter’s -f)
    args, unknown = parser.parse_known_args()
    d=os.getcwd
    model_path = "cfg_lstm_model.h5"
    tok_path = "cfg_lstm_tokenizer.pkl"
    meta_path = "cfg_lstm_meta.pkl"

    # load model/tokenizer/meta
    model = load_model(model_path)
    with open(tok_path, 'rb') as f: tokenizer = pickle.load(f)
    with open(meta_path, 'rb') as f: meta = pickle.load(f)

    if args.code is None:
        # try to load first sample from data.csv
        import os
        if not os.path.exists('data.csv'):
            raise RuntimeError('No --code provided and data.csv not found in current directory')
        df = pd.read_csv('data.csv')
        if 'code_str' not in df.columns:
            raise RuntimeError("data.csv must have 'code_str' column")
        code_str = str(df.iloc[0]['code_str'])
    else:
        with open(args.code, 'r', encoding='utf-8') as f:
            code_str = f.read()

    seq = predict_flow_for_code(code_str, model, tokenizer, meta, prob_thresh=0.5, loop_threshold=3)

    print('\nPredicted execution sequence (index: node):\n')
    for idx, text in seq:
        print(f"  {idx}: {text}")

    print('\nDone.')

# for i in seq:
#   print(i[0])

# adj_list

# node_dict