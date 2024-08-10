import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model
import os

def main():
    st.set_page_config(page_title="QA with Documents")

    st.header("QA with Documents (Information Retrieval)")

    doc = st.file_uploader("Upload your document", type=['pdf', 'txt', 'docx'])

    user_question = st.text_input("Ask your question")

    if st.button("Submit & Process"):
        if doc is not None:
            with st.spinner("Processing..."):
                try:
                    # Save the uploaded file to the Data/ directory
                    doc_path = os.path.join("Data", doc.name)
                    with open(doc_path, "wb") as f:
                        f.write(doc.getbuffer())

                    documents = load_data(doc_path)
                    model = load_model()
                    query_engine = download_gemini_embedding(model, documents)
                    response = query_engine.query(user_question)
                    st.write(response.response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a document first.")

if __name__ == "__main__":
    main()
