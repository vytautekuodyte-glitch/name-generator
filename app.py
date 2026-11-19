"""Name generator: a tiny Transformer that invents new names.

Transformers workshop, Artificial Intelligence and Machine Learning.

This app needs three files in the same folder:
    app.py             this file
    model.pt           your trained model (saved by the notebook)
    requirements.txt   the packages the app needs

Run it on your own computer, from a terminal opened in this folder:
    python -m streamlit run app.py
"""
import math
import threading
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
from torch import nn

st.set_page_config(page_title="Name Generator", layout="centered")
torch.set_num_threads(1)               # the model is tiny; one processor thread is enough
MODEL_FILE = Path(__file__).parent / "model.pt"


# ---------------------------------------------------------------------------
# The model. This is the same code as in the notebook: the app must build
# the same model before it can load the trained parameters into it.
# ---------------------------------------------------------------------------
def attention(q, k, v, mask):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)            # match
    scores = scores.masked_fill(~mask, float("-inf"))          # hide the future
    weights = torch.softmax(scores, dim=-1)                    # share
    return weights @ v, weights                                # blend


def causal_mask(size):
    return torch.tril(torch.ones(size, size, dtype=torch.bool))


def pick_next(scores, temperature=1.0):
    probabilities = torch.softmax(scores / temperature, dim=-1)
    return int(torch.multinomial(probabilities, 1))


class SelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.mix = nn.Linear(d, d, bias=False)

    def split(self, x):
        words_, positions, d = x.shape
        return x.reshape(words_, positions, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, mask):
        q, k, v = self.split(self.to_q(x)), self.split(self.to_k(x)), self.split(self.to_v(x))
        out, self.weights = attention(q, k, v, mask)
        words_, _, positions, _ = out.shape
        return self.mix(out.transpose(1, 2).reshape(words_, positions, -1))


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attention = SelfAttention(d, heads)
        self.norm1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        return self.norm2(x + self.dropout(self.mlp(x)))


class NameGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d=32, heads=2, blocks=2, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.lock = threading.Lock()                       # one visitor at a time
        self.letter_embedding = nn.Embedding(vocab_size, d)
        self.position_embedding = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(blocks)])
        self.to_scores = nn.Linear(d, vocab_size)

    def forward(self, ids):
        positions = ids.shape[1]
        x = self.letter_embedding(ids) + self.position_embedding(torch.arange(positions))
        mask = causal_mask(positions)
        for block in self.blocks:
            x = block(x, mask)
        return self.to_scores(x)


# ---------------------------------------------------------------------------
# Load the trained model once and keep it in memory.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(last_changed):
    # last_changed is the time model.pt was last written: a new file is loaded again.
    saved = torch.load(MODEL_FILE, weights_only=True)
    model = NameGPT(**saved["settings"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


if not MODEL_FILE.exists():
    st.error("The file **model.pt** is missing. It must be in the same folder as app.py. "
             "The notebook saves it in Step 10: upload it to your repository, next to app.py.")
    st.stop()
try:
    model, saved = load_model(MODEL_FILE.stat().st_mtime)
except Exception as error:
    st.error("The file **model.pt** could not be loaded. Run the notebook again from top to bottom "
             f"(Run All) and upload the new model.pt.\n\nDetails: {type(error).__name__}: {error}")
    st.stop()
chars = saved["chars"]
char_to_id = {c: i for i, c in enumerate(chars)}
known = set(saved["words"])
list_title = saved["list_name"].replace("_", " ").capitalize()


def generate(start="", temperature=0.8):
    # Invent one word, letter by letter, until the model says "stop" (symbol 0).
    ids = [0] + [char_to_id[c] for c in start.lower() if c in char_to_id and c != "."]
    ids = ids[: model.block_size - 1]
    with torch.no_grad(), model.lock:
        while len(ids) < model.block_size:
            scores = model(torch.tensor([ids]))[0, -1]
            next_id = pick_next(scores, temperature)
            if next_id == 0 and len(ids) > 1:              # "stop", but never an empty word
                break
            if next_id != 0:
                ids.append(next_id)
    return "".join(chars[i] for i in ids[1:])


def attention_table(word):
    # Attention weights of the last block while the model spells one word.
    ids = [0] + [char_to_id[c] for c in word if c in char_to_id]
    with torch.no_grad(), model.lock:
        model(torch.tensor([ids]))
        weights = model.blocks[-1].attention.weights[0].clone()
    labels = [f"{i:02d} {c}" for i, c in enumerate(["start"] + list(word))]
    rows = []
    for head in range(weights.shape[0]):
        for i, looking in enumerate(labels):
            for j, looked_at in enumerate(labels):
                if j <= i:
                    rows.append({"Head": f"Head {head + 1}", "Letter that is looking": looking,
                                 "Letter it looks at": looked_at, "Attention": float(weights[head, i, j])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The page.
# ---------------------------------------------------------------------------
st.title("Name Generator")
st.markdown(f"A small Transformer that learned from **{len(known):,} {list_title}** "
            f"and now invents new ones. Built by **{saved['author']}**.")

left, middle, right = st.columns(3)
left.metric("Parameters", f"{saved['parameters']:,}")
middle.metric("Names in its list", f"{len(known):,}")
right.metric("Letters in its alphabet", len(chars) - 1)

st.divider()
start = st.text_input("Start with (optional)", max_chars=6, placeholder="for example: " + saved["words"][0][:2])
temperature = st.slider("Creativity (temperature)", 0.3, 1.5, 0.8, 0.1,
                        help="Low: careful, close to real names. High: adventurous, sometimes strange.")
how_many = st.slider("How many names?", 1, 20, 8)

if st.button("Invent names", type="primary"):
    names = []
    for _ in range(how_many * 5):                  # a few extra tries, to avoid repeats
        name = generate(start, temperature)
        if name and name not in names:
            names.append(name)
        if len(names) == how_many:
            break
    st.session_state["names"] = names
    st.session_state["asked_for"] = how_many
    st.session_state["left_out"] = sorted({c for c in start.strip().lower() if c not in char_to_id or c == "."})

names = st.session_state.get("names", [])
if names:
    left_out = st.session_state.get("left_out", [])
    if left_out:
        st.caption("Not in this model's alphabet, so left out of the start: "
                   + ", ".join(f"'{c}'" for c in left_out))
    if len(names) < st.session_state.get("asked_for", 0):
        st.caption(f"The model found only {len(names)} different names with these settings. "
                   "Raise the creativity or shorten the start to get more.")
    invented = [name for name in names if name not in known]
    lines = []
    for name in names:
        if name in known:
            lines.append(f"- {name.capitalize()}  :gray[(a real one, already in the list)]")
        else:
            lines.append(f"- **{name.capitalize()}**  :blue[new]")
    st.markdown("\n".join(lines))
    st.caption(f"{len(invented)} of {len(names)} names are not in the list the model learned from.")

    with st.expander("Look inside the model"):
        st.write("While the model spells a name, every letter looks back at the letters before it. "
                 "A darker square means more attention. Each picture is one attention head of the last block.")
        chosen = st.selectbox("Name", names, format_func=str.capitalize)
        if chosen:
            table = attention_table(chosen)
            heads = sorted(table["Head"].unique())
            for column, head in zip(st.columns(len(heads)), heads):     # side by side; stacked on a phone
                chart = (alt.Chart(table[table["Head"] == head])
                         .mark_rect()
                         .encode(x=alt.X("Letter it looks at:N", sort=None),
                                 y=alt.Y("Letter that is looking:N", sort=None),
                                 color=alt.Color("Attention:Q", legend=None,
                                                 scale=alt.Scale(scheme="blues", domain=[0, 1])),
                                 tooltip=["Letter that is looking", "Letter it looks at",
                                          alt.Tooltip("Attention:Q", format=".2f")])
                         .properties(title=head, width="container", height=280))
                column.altair_chart(chart)

with st.expander("How does it work?"):
    st.markdown(
        "1. A name is spelled **one letter at a time**.\n"
        "2. For each new letter, the model looks back at the letters so far (**attention**) "
        "and gives every letter of the alphabet a score.\n"
        "3. One letter is drawn at random, with better-scoring letters more likely. "
        "The **temperature** decides how adventurous that draw is.\n"
        "4. This repeats until the model decides that the name is finished.\n\n"
        "Nobody gave the model any rules. It learned the style of the list by practising "
        "one task thousands of times: *predict the next letter*.")

st.divider()
st.caption("Artificial Intelligence and Machine Learning · Transformers workshop")
