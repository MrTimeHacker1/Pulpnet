import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Clear CUDA memory (IMPORTANT)
torch.cuda.empty_cache()

# Load Models
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_falcon_pipeline():
    falcon_model_name = "tiiuae/falcon-rw-1b"
    falcon_tokenizer = AutoTokenizer.from_pretrained(falcon_model_name)
    falcon_model = AutoModelForCausalLM.from_pretrained(
        falcon_model_name, trust_remote_code=True, device_map="auto"
    )
    return pipeline("text-generation", model=falcon_model, tokenizer=falcon_tokenizer)

embedding_model = load_embedding_model()
falcon_pipeline = load_falcon_pipeline()

# Load FAISS index and corpus
index = faiss.read_index("faiss_index.bin")
with open("final_corpus.pkl", "rb") as f:
    final_corpus = pickle.load(f)

def retrieve_documents(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            'text': final_corpus[idx],
            'score': float(dist)
        })
    return results

def rag_query(query, top_k=3, max_new_tokens=256):
    retrieved_docs = retrieve_documents(query, top_k=top_k)
    context = "\n\n".join([doc['text'] for doc in retrieved_docs])

    prompt = f"""You are an expert assistant answering questions about IIT Kanpur courses and academic manuals.
Here is some relevant context:
{context}

Answer the following question clearly and accurately:

Question: {query}
Answer:"""

    result = falcon_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=True)
    answer = result[0]['generated_text'][len(prompt):].strip()
    return answer

# ✅ Interface Design
st.set_page_config(page_title="IITK Chatbot", page_icon="🎓", layout="wide")

# ✅ Custom CSS (Killer UI)
st.markdown("""
    <style>
    .main {
        background-color: #f4f7fb;
    }
    header, footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #003262;
        color: white;
        border-radius: 8px;
        padding: 0.75em 1.5em;
        border: none;
        font-size: 1.1em;
    }
    .stTextInput>div>div>input {
        padding: 1em;
        border-radius: 10px;
        border: 1px solid #cccccc;
        font-size: 1.1em;
    }
    .response-box {
        background-color: white;
        padding: 1.5em;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 1.5em;
        font-size: 1.05em;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ IITK Logo (You can upload 'iitk_logo.png' and place it in app folder)
st.image("iitk_logo.png", width=120)  # Optional: IIT Kanpur logo
st.title("IIT Kanpur Academic Chatbot 🎓")
st.subheader("Ask anything about CSE Courses, UG Manual, or PG Manual")

# ✅ Query Input
query = st.text_input("Ask your question:")

if query:
    with st.spinner("Generating answer..."):
        answer = rag_query(query)
    st.markdown('<div class="response-box">{}</div>'.format(answer), unsafe_allow_html=True)

# ✅ Footer (optional)
st.markdown(
    """
    <center><small>Powered by SentenceTransformer, FAISS, and Falcon-RW-1B (RAG Pipeline)<br>IIT Kanpur Chatbot Prototype</small></center>
    """,
    unsafe_allow_html=True,
)
