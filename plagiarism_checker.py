import streamlit as st
import PyPDF2
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

st.set_page_config(page_title="PDF Plagiarism Checker", layout="wide")

st.title("🕵️‍♂️ PDF Plagiarism Checker for Students")

st.write("Upload your students' PDF assignments to check how similar they are!")

# Step 1: Upload PDFs
uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    texts = []
    filenames = []

    # Step 2: Extract text from each PDF
    for pdf in uploaded_files:
        reader = PyPDF2.PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        texts.append(text)
        filenames.append(pdf.name)

    # Step 3: Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Step 4: Compute similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 5: Convert to DataFrame
    df = pd.DataFrame(similarity_matrix, columns=filenames, index=filenames)

    st.subheader("📊 Similarity Between All Uploaded PDFs")
    st.dataframe(df.style.background_gradient(cmap="YlGnBu"))

    st.markdown("✅ **Note:** 1.00 = same, 0.00 = completely different")

    # Step 6: Optional - Select one file and compare with all
    selected_file = st.selectbox("Select a file to compare with others:", filenames)
    selected_index = filenames.index(selected_file)
    st.subheader(f"📈 Similarity of '{selected_file}' with others:")
    
    result = pd.DataFrame({
        "File": filenames,
        "Similarity": similarity_matrix[selected_index]
    })
    st.table(result.sort_values(by="Similarity", ascending=False))
