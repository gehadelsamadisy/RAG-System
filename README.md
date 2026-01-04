# 🔍 RAG System for CNN Document Q&A

A Retrieval-Augmented Generation (RAG) pipeline that enables intelligent question-answering over PDF documents about Convolutional Neural Networks. Built with LangChain, ChromaDB, and Groq's ultra-fast LLM inference.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)

---

## 📋 Overview

This project implements a complete RAG pipeline that:

1. **Loads** PDF documents and extracts text content
2. **Chunks** text using recursive character splitting with overlap
3. **Embeds** chunks using HuggingFace's sentence-transformers
4. **Stores** embeddings in a ChromaDB vector database
5. **Retrieves** relevant context using similarity search
6. **Generates** accurate answers using Groq's LLaMA 3.1 model

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  PDF Input  │────▶│ Text Splitter│────▶│ Chunk Embeddings│
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Answer    │◀────│   LLM (Groq) │◀────│ ChromaDB Vector │
│             │     │  LLaMA 3.1   │     │     Store       │
└─────────────┘     └──────────────┘     └─────────────────┘
                           ▲
                           │
                    ┌──────┴──────┐
                    │   Question  │
                    └─────────────┘
```

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **📄 PDF Processing** | Seamlessly extracts and processes text from PDF documents |
| **🧩 Smart Chunking** | Recursive text splitting with 500-char chunks and 50-char overlap |
| **🔢 Local Embeddings** | Uses `all-MiniLM-L6-v2` for fast, free embedding generation |
| **🗄️ Vector Storage** | ChromaDB for efficient similarity search |
| **⚡ Ultra-Fast LLM** | Groq's LLaMA 3.1-8B for blazing-fast inference |
| **📊 Evaluation Metrics** | Built-in precision, recall, and F1 score calculation |
| **💾 Export Options** | Answers exported to JSON, CSV, and TXT formats |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- A free Groq API key ([Get one here](https://console.groq.com/keys))

### Dependencies

```bash
pip install langchain langchain-core langchain-community langchain-groq
pip install chromadb pypdf tiktoken sentence-transformers
```

---

## ⚙️ Configuration

1. **Get your free Groq API key** from [console.groq.com/keys](https://console.groq.com/keys)

2. **Set your API key** in the notebook:
   ```python
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

3. **Specify your PDF path**:
   ```python
   pdf_path = "path/to/your/document.pdf"
   ```

---

## 📖 Usage

### Running the Notebook

1. Open `dl-rag-final.ipynb` in Jupyter, Kaggle, or Google Colab
2. Set your Groq API key and PDF path
3. Run all cells sequentially

### Interactive Q&A

```python
# Ask any question about your document
question = "What are the main components of a CNN?"
answer, sources = ask_question(question)
```

### Batch Processing

The notebook includes 15 pre-defined test questions covering:
- CNN architecture components
- Famous architectures (AlexNet, VGG, ResNet, GoogleNet)
- Pooling layers and activation functions
- Batch normalization and dropout
- Transfer learning and data augmentation

---

## 📊 Evaluation

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Fraction of retrieved chunks that are relevant |
| **Recall** | Fraction of relevant chunks that were retrieved |
| **F1 Score** | Harmonic mean of precision and recall |

### Answer Quality Assessment

The system uses LLM-based evaluation scoring:
- **Faithfulness** (1-5): Is the answer grounded in context?
- **Relevance** (1-5): Does the answer address the question?
- **Completeness** (1-5): Is the answer thorough?

---

## 📁 Output Files

| File | Format | Description |
|------|--------|-------------|
| `rag_answers.json` | JSON | Structured Q&A pairs |
| `rag_answers.csv` | CSV | Tabular format for analysis |
| `rag_answers.txt` | TXT | Human-readable formatted output |

---

## 🔧 Technical Details

### Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50,    # Overlap for context continuity
    separators=["\n\n", "\n", " ", ""]
)
```

### Embedding Model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Size**: ~80MB
- **Speed**: Very fast, runs on CPU

### LLM Configuration

- **Provider**: Groq (Free tier available)
- **Model**: `llama-3.1-8b-instant`
- **Temperature**: 0 (deterministic)
- **Max Tokens**: 512

---

## 📸 LangSmith Traces

The project includes LangSmith execution traces for monitoring:

| Trace | Description |
|-------|-------------|
| `langsmith1.png` | Full execution pipeline |
| `langsmith2.png` | Retriever component |
| `langsmith3.png` | LLM generation step |

---

## 🎯 Sample Results

**Question**: *What are the main components of a CNN?*

**Answer**: The main components of a Convolutional Neural Network (CNN) are:
1. **Convolutional Layers** - Apply learnable filters to detect features
2. **Pooling Layers** - Reduce spatial dimensions (max/average pooling)
3. **Activation Functions** - Introduce non-linearity (ReLU, Leaky ReLU)
4. **Fully Connected Layers** - Combine features for final classification
5. **Batch Normalization** - Stabilizes training, allows higher learning rates

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Protobuf conflicts | The notebook includes automatic fixes for protobuf version conflicts |
| Slow embeddings | First run downloads the model (~400MB), subsequent runs use cached version |
| API rate limits | Groq's free tier has generous limits; space out batch requests if needed |

---

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Groq Console](https://console.groq.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 📄 License

This project was created for educational purposes as part of a Deep Learning course assignment.

---

<p align="center">
  Built with ❤️ using LangChain & Groq
</p>

