import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader

st.set_page_config(page_title="PDF Q&A with Gemini")

st.title("📄 PDF Q&A with Gemini")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("✅ Gemini API Connected")

except Exception as e:
    st.error("❌ Gemini API Key not found")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:

        with st.spinner("Extracting text from PDF..."):

            reader = PdfReader(tmp_path)

            full_text = ""

            for page in reader.pages:
                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        def chunk_text(text, chunk_size=500, overlap=50):

            chunks = []

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunks.append(text[start:end])

                start += chunk_size - overlap

            return chunks

        texts = chunk_text(full_text)

        with st.spinner("Creating embeddings and FAISS index..."):

            model_emb = SentenceTransformer("all-MiniLM-L6-v2")

            embeddings = model_emb.encode(texts).astype("float32")

            index = faiss.IndexFlatL2(embeddings.shape[1])

            index.add(embeddings)

        st.success(f"✅ PDF processed successfully! {len(texts)} chunks created.")

        question = st.text_input("Ask a question about your PDF")

        if st.button("Get Answer"):

            if question.strip() == "":
                st.warning("Please enter a question")

            else:

                q_emb = model_emb.encode([question]).astype("float32")

                distances, indices = index.search(q_emb, k=4)

                context = "\n\n".join([texts[i] for i in indices[0]])

                prompt = f"""
Use only the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

                try:

                    llm = genai.GenerativeModel("gemini-2.5-flash-lite")

                    with st.spinner("Gemini is thinking..."):

                        response = llm.generate_content(prompt)

                        st.subheader("Answer")

                        st.write(response.text)

                    with st.expander("📋 Context Used"):
                        st.write(context)

                except Exception as e:
                    st.error(f"Gemini Error: {e}")

    finally:

        os.unlink(tmp_path)
