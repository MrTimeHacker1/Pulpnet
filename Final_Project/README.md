# 🎓 IIT Kanpur Academic Chatbot (RAG-Powered)

This project is an **AI-powered chatbot** built using:
- **Retrieval-Augmented Generation (RAG)**
- FAISS vector similarity search
- Sentence Transformer embeddings
- A lightweight Language Model (Falcon-RW-1B or alternatives)

It can answer academic queries related to:
- **CSE Courses**
- **UG Manual**
- **PG Manual**

---

## 📂 Project Structure
├── data/
│ ├── ALL_DATA.jsonl # CSE Courses Data
│ ├── UG_Manual.pdf # UG Manual PDF
│ └── PG_Manual.pdf # PG Manual PDF
│
├── Data_Loader.ipynb # Jupyter Notebook for preprocessing & saving FAISS index
├── faiss_index.bin # Saved FAISS vector index (generated from notebook)
├── final_corpus.pkl # Saved processed text corpus
├── iitk_logo.png # (Optional) IIT Kanpur Logo for UI header
├── app.py # Streamlit App (Chatbot Interface)
├── README.md # Project Documentation (this file)
└── requirements.txt # Required Python Packages

## Step:1 Clone the repository
```bash
git clone https://github.com/MrTimeHacker1/Pulpnet.git
cd Pulpnet/Final_Project
```
## Step:2 Install the Dependencies

```bash
pip install -r requirements.txt
```

## Step:3

```bash
jupyter notebook Data_Loader.ipynb
```

Execute each and every cell of the notebook

## Step:4 Run the app

```bash
streamlit run app.py
```
