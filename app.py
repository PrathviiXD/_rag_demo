import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# ---------------- STREAMLIT CONFIG ----------------

st.set_page_config(page_title="PDF Q&A with Gemini")

st.title("📄 PDF Q&A with Gemini + OCR")

# ---------------- GEMINI CONFIG ----------------

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("✅ Gemini API Connected")

except Exception:
    st.error("❌ Gemini API Key not found")
    st.stop()

# ---------------- TESSERACT PATH ----------------
# WINDOWS USERS ONLY

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---------------- LOAD EMBEDDING MODEL ----------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model_emb = load_model()

# ---------------- CHUNKING FUNCTION ----------------

def chunk_text(text, chunk_size=700, overlap=120):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

# ---------------- MAIN PROCESS ----------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

        tmp.write(uploaded_file.read())

        tmp_path = tmp.name

    try:

        # ---------------- TEXT EXTRACTION ----------------

        with st.spinner("📖 Extracting text from PDF..."):

            reader = PdfReader(tmp_path)

            full_text = ""

            # ---------- NORMAL TEXT EXTRACTION ----------

            for page in reader.pages:

                text = page.extract_text()

                if text and len(text.strip()) > 30:
                    full_text += text + "\n"

        # ---------------- OCR FALLBACK ----------------

        if len(full_text.strip()) < 100:

            st.info("⚡ Scanned/Image PDF detected. Running OCR...")

            images = convert_from_path(tmp_path)

            ocr_text = ""

            progress = st.progress(0)

            custom_config = r'--oem 3 --psm 6'

            for i, image in enumerate(images):

                extracted = pytesseract.image_to_string(
                    image,
                    config=custom_config
                )

                ocr_text += extracted + "\n"

                progress.progress((i + 1) / len(images))

            full_text = ocr_text

        # ---------------- SAFETY CHECK ----------------

        if len(full_text.strip()) < 50:

            st.error("❌ Could not extract readable text from PDF.")

            st.stop()

        # ---------------- CHUNKING ----------------

        texts = chunk_text(full_text)

        # ---------------- EMBEDDINGS + FAISS ----------------

        with st.spinner("🧠 Creating embeddings and FAISS index..."):

            if len(texts) == 0:

                st.error("❌ No text chunks created.")

                st.stop()

            embeddings = model_emb.encode(texts)

            embeddings = np.array(embeddings).astype("float32")

            if len(embeddings.shape) != 2:

                st.error("❌ Embedding creation failed.")

                st.stop()

            index = faiss.IndexFlatL2(embeddings.shape[1])

            index.add(embeddings)

        st.success(
            f"✅ PDF processed successfully! {len(texts)} chunks created."
        )

        # ---------------- QUESTION INPUT ----------------

        question = st.text_input(
            "Ask a question about your PDF"
        )

        # ---------------- ANSWER GENERATION ----------------

        if st.button("Get Answer"):

            if question.strip() == "":

                st.warning("⚠️ Please enter a question")

            else:

                with st.spinner("🔍 Searching relevant context..."):

                    q_emb = model_emb.encode([question]).astype("float32")

                    distances, indices = index.search(q_emb, k=4)

                    context = "\n\n".join(
                        [texts[i] for i in indices[0]]
                    )

                prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the PDF."

---------------- CONTEXT ----------------

{context}

---------------- QUESTION ----------------

{question}

---------------- ANSWER ----------------
"""

                try:

                    llm = genai.GenerativeModel(
                        "gemini-2.5-flash-lite"
                    )

                    with st.spinner("🤖 Gemini is generating answer..."):

                        response = llm.generate_content(prompt)

                    st.subheader("📌 Answer")

                    st.write(response.text)

                except Exception as e:

                    st.error(f"❌ Gemini Error: {e}")

    finally:

        os.unlink(tmp_path)