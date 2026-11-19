# Name Generator

A small Transformer that invents new names. I built and trained it in the Transformers workshop of the course *Artificial Intelligence and Machine Learning*.

**Try it:** PASTE-YOUR-APP-LINK-HERE

## What it does

The model learned from a list of names by practising one task: predict the next letter. To invent a name it spells one letter at a time, and the temperature slider decides how adventurous each choice is.

## What is in this repository

| File | What it is |
|---|---|
| `app.py` | the web app (Streamlit) and the model code |
| `model.pt` | the trained parameters, the alphabet, and the list of names |
| `requirements.txt` | the packages the app needs |

## The model

- A decoder-only Transformer (the same design as GPT), written in PyTorch.
- 2 blocks, 2 attention heads per block, vectors of length 32, about 19,000 parameters.
- Trained on a laptop in less than a minute.

## The list

The lists of names come from the workshop materials. The lists of towns and cities are from the GeoNames geographical database ([geonames.org](https://www.geonames.org), Creative Commons Attribution 4.0); the English names are from the `names.txt` list of Andrej Karpathy's *makemore* project (MIT licence). If you trained the model on your own list, describe it here.

## Run it on your own computer

These are commands for a **terminal**, not for a notebook cell.

1. Open a terminal **in the folder that contains `app.py`**. In VS Code: right-click that folder and choose **Open in Integrated Terminal**.
2. Type these two lines, one after the other:

```
python -m pip install streamlit torch
python -m streamlit run app.py
```

On a Mac, write `python3` instead of `python` if the terminal answers `command not found: python`.

The app opens in your browser at `http://localhost:8501`. To stop it, click into the terminal and press **Ctrl + C**.

If it does not start:

| The terminal says | What to do |
|---|---|
| `command not found: pip` or `streamlit` | Use the two lines above exactly: `python -m ...` works even when `pip` and `streamlit` are not found. |
| `File does not exist: app.py` | The terminal is in another folder. Type `ls` (Windows: `dir`): you must see `app.py`. If not, open the terminal as in step 1. |
| `No module named streamlit` | The first line was not run, or it was run with another Python. Run both lines again in the same terminal. |
| The page says that `model.pt` is missing | Save your model first (Step 10 of the notebook), so that `model.pt` lies next to `app.py`. |
