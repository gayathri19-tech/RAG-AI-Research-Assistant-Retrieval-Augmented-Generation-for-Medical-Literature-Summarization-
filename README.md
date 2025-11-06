# RAG-AI-Research-Assistant-Retrieval-Augmented-Generation-for-Medical-Literature-Summarization-
An Intelligent Retrieval-Augmented Generation (RAG) System for Summarizing Medical Research Papers

**OVERVIEW**:
This project demonstrates the design and implementation of an AI-powered research assistant that leverages Retrieval-Augmented Generation (RAG) to extract, summarize, and explain insights from medical research papers.The system helps clinicians, researchers, and policymakers quickly access verified, context-aware evidence from large PDF collections, transforming hours of manual review into seconds of AI-driven retrieval and summarization.

**OBJECTIVES**:
Enable semantic retrieval of medical literature using Sentence-BERT embeddings.
Generate factual, evidence-based summaries with LLMs guided by retrieved context.
Evaluate system performance using BLEU, ROUGE, and RAGAS metrics.
Deliver an intuitive Streamlit interface for real-time Q&A with uploaded PDFs.

**System Architecture**:
The solution integrates retrieval and generation components within a modular RAG pipeline:
| Component             | Description                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| **Retriever**         | Uses Sentence-BERT (all-mpnet-base-v2) embeddings and FAISS vector database for dense semantic search. |
| **Generator**         | Leverages OpenRouter’s GPT-based models for natural-language generation grounded in retrieved text.    |
| **Evaluation Module** | Computes BLEU, ROUGE, RAGAS, and SBERT similarity scores for retrieved Q&A pairs.                      |
| **UI Layer**          | Built using Streamlit for interactive document upload, question answering, and metric visualization.   |

**🧩 Technologies Used**

Python 3.10+

LangChain – retrieval & generation chaining

Sentence-BERT (all-mpnet-base-v2) – semantic embedding

FAISS / Chroma – vector storage for dense retrieval

OpenRouter LLM API – for grounded text generation

Streamlit – front-end interface

Pandas, NumPy, Scikit-learn, Matplotlib – evaluation & visualization

**Evaluation Metrics**:
| Metric                          | Purpose                        | Result                   |
| ------------------------------- | ------------------------------ | ------------------------ |
| **Precision@1**                 | Measures retrieval accuracy    | **0.94**                 |
| **R@10**                        | Recall of relevant chunks      | **0.94**                 |
| **Semantic Similarity (SBERT)** | Conceptual alignment           | **0.57**                 |
| **BLEU / ROUGE**                | Lexical & content accuracy     | Moderate alignment       |
| **RAGAS Suite**                 | Faithfulness, answer relevancy | Stable factual grounding |
The RAG system achieved high retrieval precision and strong contextual faithfulness, validating its utility for real-world healthcare data applications

**Streamlit Interface**
Upload multiple research PDFs
Ask domain-specific questions
View evidence-grounded answers
Evaluate retrieval & generation results instantly

**📈 Key Results**
Reduced manual review time by up to 80% for clinicians and researchers.
Demonstrated high retrieval precision (P@1 = 0.94) and consistent semantic coherence.
Validated a replicable, domain-adapted RAG pipeline suitable for healthcare and research industries.

**How to Run Locally**
# Clone the repository
git clone https://github.com/<your-username>/RAG-AI-Research-Assistant.git
cd RAG-AI-Research-Assistant

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

**📂 Repository Structure**
RAG-AI-Research-Assistant/
│
├── app.py                 # Streamlit interface
├── retriever.py           # FAISS retriever logic
├── generator.py           # OpenRouter LLM response generator
├── evaluation.py          # Evaluation metrics (BLEU, ROUGE, RAGAS)
├── data/                  # Sample PDFs
├── docs/                  # Screenshots and report links
├── requirements.txt
└── README.md

**Impact**

This project demonstrates how responsible, evidence-grounded AI can help researchers and clinicians access trustworthy insights faster. The framework can be extended to domains like law, finance, or education for efficient knowledge retrieval.

**References**

[Streamlit Documentation](https://docs.streamlit.io/)

[LangChain Documentation](https://docs.langchain.com/)

[OpenRouter API](https://openrouter.ai/docs)

[Sentence Transformers](https://www.sbert.net/)

[FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
