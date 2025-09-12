
# Transfer Learning in NLP — Hands‑On with Hugging Face Transformers

This repo showcases what I learnt during training on **transfer learning for Natural Language Processing (NLP)**.  
It includes a single Jupyter notebook that walks through using **pretrained Transformer models** for common NLP tasks and peeks under the hood of tokenizers and model heads.

> Notebook: `Transfer_Learning_in_NLP.ipynb`

---

## 🔎 What you’ll learn

- Why transfer learning is powerful in NLP (data efficiency, faster time‑to‑value, and better performance)
- How to use the 🤗 **Transformers** `pipeline` API for quick wins (sentiment analysis, etc.)
- How tokenization works and how inputs are prepared for Transformer models
- How to load **TensorFlow** model variants (with an example of importing from PyTorch weights)
- How to customize model heads and configs for different downstream tasks

---

## 📓 Notebook sections

The notebook is structured as follows:

1. **Transfer Learning** — overview & motivation  
2. **Transformers** — a quick tour of the library  
3. **Getting started with a `pipeline`** — sentiment analysis demo  
4. **Under the Hood: pretrained models** — tokenizers, encodings, and batched inputs  
5. **Accessing the Code** — loading specific model/tokenizer classes  
6. **Customize the Model** — changing configs (e.g., number of labels)

---

## 🚀 Quick start (local)

1. **Create and activate a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**  
   ```bash
   pip install -U transformers tensorflow
   # Optional (only needed if you import PyTorch weights with from_pt=True):
   pip install torch
   ```

3. **Launch Jupyter**  
   ```bash
   pip install -U jupyter
   jupyter notebook
   ```

4. **Open** `Transfer_Learning_in_NLP.ipynb` and run the cells top‑to‑bottom.

> **Notes**
> - The notebook uses **TensorFlow** model classes (e.g., `TFAutoModelForSequenceClassification`).  
> - In one example, it demonstrates loading a PyTorch‑only checkpoint into TensorFlow with `from_pt=True`. If you run that section, you may need the `torch` package installed.

---

## ☁️ Run in Google Colab (no setup)

You can run the notebook in a browser with a GPU in **Google Colab**:

- Go to **https://colab.research.google.com** → *Upload* → select `Transfer_Learning_in_NLP.ipynb`.  
- Or, once this repo is live on GitHub, you can add a badge like this (update `<USER>/<REPO>`):

  ```markdown
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/Transfer_Learning_in_NLP.ipynb)
  ```

---

## 🧪 Example (sentiment analysis)

```python
from transformers import pipeline
clf = pipeline("sentiment-analysis")
clf("We are very happy to be a part of this LLM course")
# [{'label': 'POSITIVE', 'score': 0.99...}]
```

---

## 📁 Repo structure

```
.
├── README.md                          # This file
└── Transfer_Learning_in_NLP.ipynb     # Main tutorial notebook
```

---

## ✅ Learning outcomes

By the end of this notebook, you should be able to:

- Explain the benefits of transfer learning in NLP
- Use `pipeline` for quick inference across common tasks
- Inspect tokenization outputs (input IDs, attention masks)
- Load specific pretrained models and tokenizers
- Adjust model configs for different label spaces / tasks

---

## 🔧 Where to go next

- **Fine‑tune** a model on a small dataset (e.g., SST‑2 for sentiment, CoNLL for NER)  
- Try **other tasks**: question answering (`pipeline("question-answering")`), summarization, translation  
- Explore **model distillation** and **quantization** for efficiency  
- Compare **TensorFlow vs PyTorch** training loops and `Trainer`/`Keras` APIs

---

## 🙏 Acknowledgements

- Built with the excellent [🤗 Transformers](https://github.com/huggingface/transformers) library.

---

## 📝 License

Choose a license before publishing (e.g., MIT, Apache‑2.0). You can add a `LICENSE` file later.
