<h1>Hi, I'm Aishwarya! <br/><a href="https://www.linkedin.com/in/aishwarya-chennabathni/">Data Scientist</a></h1>

<h2>👨‍💻 Introduction:</h2>

I’m a Data Scientist with 6+ years of experience delivering end-to-end AI/ML and GenAI solutions across healthcare, logistics, insurance, real estate, and the public sector. I build production systems—from forecasting and pricing to RAG assistants and document intelligence—grounded in strong MLOps, measurable impact, and clear communication with business leaders.&#x20;

**💡 What I Do:**

* **GenAI & NLP**: RAG apps (LangChain, Llamaindex, FAISS), GPT-4–powered assistants, semantic search, summarization, NER, OCR + entity extraction.
* **Forecasting & optimization**: SARIMA/Prophet/LSTM for demand, inventory, and capacity; elasticity modeling and causal A/B testing for pricing.
* **Predictive ML**: XGBoost/LightGBM/Scikit-learn pipelines for churn, risk, and classification with SHAP/LIME for explainability.
* **Analytics to action**: translate insights into workflows and dashboards; lead Agile sprints, reviews, and stakeholder readouts.

**⚙ Tech Stack:**

* **Languages:** Python, SQL
* **ML/DL:** Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow, XGBoost, LightGBM, Hugging Face, Statsmodels
* **Data & Orchestration:** Snowflake, Redshift, BigQuery, Spark/PySpark, dbt, Airflow
* **MLOps & Cloud:** MLflow, Docker, GitHub Actions CI/CD; AWS SageMaker, Azure ML, GCP Vertex AI
* **GenAI/RAG:** GPT-4, LangChain, FAISS, Prompt Engineering; governance with SHAP/LIME and compliance (FOIP/HIA/GDPR)

**📈 Recent Highlights:**

* Shipped a secure GPT-4 virtual assistant for Ontario’s Ministry of Health (RAG over policy + Snowflake), cutting inquiry resolution time by 60% and improving accessibility.
* Built multi-seasonal LTC bed-demand forecasting (SARIMA) with drift diagnostics and MLflow tracking, embedded in operational dashboards.
* Delivered churn prediction and pricing optimization for logistics, improving early churn detection and informing revenue-positive pricing moves.

**🔎 Interests:**
Responsible AI, LLMOps (prompt versioning, drift monitoring), retrieval quality, and turning analytical models into reliable, user-centric products.

## 💼 Work Experience
- 🏢 **Ministry of Health Ontario, Ministry of Long-Term Care** · 📍 **Toronto, ON** · 📅 **Sept 2024 – Present**
- 🏢 **Tiger Analytics** · 📅 **May 2022– June 2024**
- 🏢 **Optum Global Solutions, United Health Group** · 📅 **July 2017– August 2020**

## 🎓 Education
- 🎓 **Master of Business Administration in STEM - Data Science, AI/ML and Analytics** · 📅 **May 2020– May 2022**             
Indian Institute of Management Tiruchirappalli              
Focused on applied machine learning, statistics, and business analytics      

- 🎓 **Bachelor of Technology in Electrical and Electronics Engineering** · 📅 **Sept 2013–July 2017**          
VNR Vignana Jyothi Institute of Engineering and Technology

<h2>💡 My Portfolio Projects </h2>

Welcome to my GitHub portfolio! Here are some of the projects I've worked on.

### 🧠 Advanced RAG Pipeline — LlamaIndex + TruLens

**Description:** Built and evaluated a production-minded RAG system that pairs **Sentence-Window Retrieval (SWR)** for high-precision local context with **Auto-Merging Retrieval (AMR)** for coherent multi-chunk context. Quality is measured via the **RAG-Triad**—**Context Relevance, Groundedness, Answer Relevance**—with a leaderboard tracking latency, tokens, and cost.

**Models / Tools / Techniques:** LlamaIndex · TruLens · OpenAI/embedding models · Sentence-Window Retrieval · Auto-Merging Retrieval · Hierarchical chunking · Cross-encoder reranking (BGE) · VectorStoreIndex (persisted) · Prompt budgeting
Evaluation: RAG-Triad (CR/G/AR) · Leaderboard (latency/tokens/cost) · A/B retrieval experiments (window size, hierarchy depth)

**Highlights:** SWR + AMR outperform dense baseline on **grounding** and **context relevance** · Coherent, non-fragmented context via AMR · Token/cost reduction vs naïve top-k · Reproducible pipeline with persisted indexes and TruLens dashboards

[**Open Colab — Advanced\_RAG\_Pipeline**](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques/blob/main/Advanced_RAG_Pipeline.ipynb)

[**Open Colab — RAG\_Triad\_of\_metrics**](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques/blob/main/RAG_Triad_of_metrics.ipynb)

[**Open Colab — Sentence\_window\_retrieval**](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques/blob/main/Sentence_window_retrieval.ipynb)

[**Open Colab — Auto-merging\_Retrieval**](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques/blob/main/Auto-merging_Retrieval.ipynb)

[**GitHub Repo**](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques)

### 📩 Spam SMS Classification & Data Analysis — GPT-2 Fine-Tuning

**Description:** Fine-tuned **GPT-2 (124M)** with a lightweight classifier head to detect **spam vs ham** on the **UCI SMS Spam Collection**. Built a **repeatable, auditable** workflow: deterministic preprocessing & splits, balanced training subset for pedagogy, and clear evaluation with accuracy/loss curves plus optional PR/ROC on the original imbalance.

**Models / Tools / Techniques:** GPT-2 small (decoder-only) · PyTorch · `tiktoken` (byte-level BPE) · AdamW · Selective unfreezing (last block + LN + Linear 768→2) · Last-token logits for CE loss · Pandas/NumPy · Matplotlib
Evaluation & Ops: Accuracy/Loss curves · (Optional) Precision/Recall/F1, ROC-AUC · Threshold calibration · Seeded splits · CPU/GPU friendly

**Highlights:** Fast convergence (≈4–5 epochs) on balanced subset · Robust to obfuscation (emoji/unicode) thanks to byte-level BPE · Minimal compute, production-style head · Artifacts saved (`loss-plot.pdf`, `accuracy-plot.pdf`, `review_classifier.pth`) · Ready for drift monitoring & recalibration

[**Open Colab**](https://github.com/Aishwarya-chen11/Fine-tuned-LLM-Classification-Model/blob/main/Fine_tuned_LLM_classification_model.ipynb)  ·  [**GitHub Repo**](https://github.com/Aishwarya-chen11/Fine-tuned-LLM-Classification-Model)

### 🖼️ Multi-Modal RAG with LangChain (PDF + Images)

**Description:** Built a **multimodal RAG** pipeline that unifies **text and images** from PDFs in one vector space using **CLIP ViT-B/32**. Extracts page text & figures with **PyMuPDF**, embeds both into **512-d unit vectors**, retrieves cross-modal matches via **FAISS**, then formats a **multimodal prompt** (text + base64 images) for **GPT-4.1** to produce grounded answers citing the exact page and figure.

**Models / Tools / Techniques:** LangChain · PyMuPDF (fitz) · Pillow · CLIP ViT-B/32 (Transformers) · FAISS · GPT-4.1 (vision)
Unified embeddings (text+image) · L2-normalization (512-d) · RecursiveCharacterTextSplitter (500/100) · similarity\_search\_by\_vector · Multimodal prompt (image\_url + base64)

**Highlights:** One embedding model → **cross-modal retrieval** · Answers grounded with **page snippets + figures** · No captioning required · Simple, fast stack (CLIP + FAISS) · Reproducible notebook with example queries (“What does the chart on page 1 show?”)

[**Open Colab**](https://github.com/Aishwarya-chen11/Build-MultiModal-RAG-with-Langchain/blob/main/multimodalopenai.ipynb) · [**GitHub Repo**](https://github.com/Aishwarya-chen11/Build-MultiModal-RAG-with-Langchain)

### 📝 Instruction Fine-Tuning a Decoder-Only LLM (GPT-2 Medium)

**Description:** End-to-end **SFT** pipeline that turns a pretrained **GPT-2 Medium (355M)** into a better instruction follower. Converts raw JSON to **Alpaca-style** prompts, applies **dynamic padding** with **label masking** (train on **response tokens only**), fine-tunes with **AdamW**, runs deterministic inference, and performs lightweight **LLM-as-a-judge** scoring for quick quality checks.

**Models / Tools / Techniques:** PyTorch · GPT-2 Medium (355M) · `tiktoken` (GPT-2 BPE) · Dynamic padding · `ignore_index=-100` label masking · Cross-entropy over response tokens · AdamW (5e-5) · Deterministic generation · Optional **Ollama Llama-3** judge · `tqdm`

**Highlights:** Visible instruction adherence after **1 epoch** · Reproducible seeds & saved artifacts (JSON preds + checkpoint) · Clean, didactic code (no LoRA; easy to extend) · CPU/Single-GPU friendly · Simple to swap schedulers/PEFT later

[**Open Colab**](https://github.com/Aishwarya-chen11/LLM-Instruction-Fine-tuning/blob/main/Instruction_Fine_Tuning_LLM.ipynb) · [**GitHub Repo**](https://github.com/Aishwarya-chen11/LLM-Instruction-Fine-tuning)

### 🧩 Building LLMs From Scratch — GPT-Style (PyTorch)

**Description:** Implemented a GPT-style decoder-only Transformer **from first principles**: custom tokenizer → input/target maker → dataloader → token + positional embeddings → **causal multi-head self-attention** (fused QKV) → **GELU** MLP → **pre-norm + residuals** → decoding (**temperature / top-k**) → loss & **perplexity**. Validated wiring by mapping **GPT-2 (124M)** weights.

**Models / Tools / Techniques:** PyTorch · NumPy · (optional) `tiktoken` / `transformers` · Causal mask · Multi-head attention (fused QKV) · GELU MLP · LayerNorm (pre-norm) · Tied embeddings · Top-k sampling & temperature · Perplexity tracking

**Highlights:** End-to-end working GPT in a single annotated notebook · Correct causal masking & pre-norm stack · GPT-2 weight mapping sanity-checks the implementation · Colab-friendly; lightweight text demo (`the-verdict.txt`) for quick runs

[**Open Colab**](https://github.com/Aishwarya-chen11/Build-LLM-architecture-from-scratch/blob/main/Building_LLM_from_Scratch.ipynb) · [**GitHub Repo**](https://github.com/Aishwarya-chen11/Build-LLM-architecture-from-scratch)

<h2> 🤳 Connect with me:</h2>

Feel free to reach out if you have any questions or would like to collaborate on a project!

[<img align="left" alt="JoshMadakor | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="Aishwarya Chennabathni | Gmail" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/gmail.svg" />][gmail]

[linkedin]:https://www.linkedin.com/in/aishwarya-chennabathni/
[gmail]:mailto:aishwarya.chen11@gmail.com


