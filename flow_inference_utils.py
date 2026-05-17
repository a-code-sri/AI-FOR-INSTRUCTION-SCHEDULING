import pickle
import re
import ast
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CFG Builder
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
        elif ln.startswith("for("):
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
        else:
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
            if exit_idx < len(nodes): edges[start].append(exit_idx)
        elif it["type"] == "for":
            hdr, body = start, start + 1
            edges[hdr] = [body, exit_idx] if exit_idx < len(nodes) else [body]
            edges[body] = [hdr]
        elif it["type"] == "if_chain":
            pairs = len(it["conds"])
            for k in range(pairs):
                cond, body = start + 2*k, start + 2*k + 1
                nxt = start + 2*(k+1) if k + 1 < pairs else (start + 2*pairs if it.get("has_else") else exit_idx)
                edges[cond] = [body, nxt]
                edges[body] = [exit_idx]
            if it.get("has_else"):
                else_idx = start + 2*pairs
                edges[else_idx] = [exit_idx]
    node_dict = {i: nodes[i] for i in range(len(nodes))}
    return node_dict, edges

#Prediction helpers

def load_artifacts(prefix="cfg_lstm"):
    model = load_model(f"{prefix}_model.h5")
    with open(f"{prefix}_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(f"{prefix}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, tokenizer, meta


def predict_decision(input_context: str, model, tokenizer, meta, threshold=0.5):
    seq = tokenizer.texts_to_sequences([input_context])
    padded = pad_sequences(seq, maxlen=meta.get("maxlen", 40), padding="pre")
    prob = float(model.predict(padded, verbose=0)[0, 0])
    return {"probability": prob, "decision": "ENTER" if prob >= threshold else "SKIP"}
