import streamlit as st
from dotenv import load_dotenv
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# start
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# all uploaded pdfs text extraction function
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# big text chunking function
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks


# chunks to vector store function
def get_vector_store(text_chunks):
    """
    Build and save FAISS index. Choose embedding backend with environment variable:
      USE_LOCAL_EMBEDDINGS=true  -> use local HuggingFace model (no Google quota)
      (default)                  -> use Google embeddings (may hit quota)
    """
    
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    if use_local:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


# prompt template
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    and make sure to provide all the details. If the answer is not provided in context just
    say, "answer not available in context please ask something related to what you have uploaded" in a creative, funny way.

    Context :
    {context}

    Question :
    {question}

    Answer :
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# user input function
def user_input(user_question):
    # ensure index exists
    if not os.path.exists("faiss_index"):
        st.warning("No index found. Upload and process PDFs first.")
        return

    # use same embedding backend used to create the index
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    if use_local:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # allow unsafe pickle deserialization only if explicitly trusted
    allow_danger = os.getenv("TRUST_FAISS_INDEX", "false").lower() == "true"

    try:
        if allow_danger:
            # some langchain/FAISS versions accept this kwarg
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            new_db = FAISS.load_local("faiss_index", embeddings)
    except TypeError:
        # fallback if implementation signature does not accept the kwarg
        try:
            new_db = FAISS.load_local("faiss_index", embeddings)
        except Exception as e:
            st.error(f"Failed to load vector index: {e}")
            st.info("If you created the index locally and trust it, set TRUST_FAISS_INDEX=true in your .env and restart the app.")
            st.info("Or delete the 'faiss_index' folder and click Process to rebuild the index from your PDFs.")
            return
    except Exception as e:
        msg = str(e).lower()
        if "quota" in msg or "limit" in msg:
            st.error("Failed to load vector index: API/quota error detected.")
            st.info("Consider setting USE_LOCAL_EMBEDDINGS=true in your .env to build a local embedding index (no Google quota).")
        else:
            st.error(f"Failed to load vector index: {e}")
        st.info("If you created the index locally and trust it, set TRUST_FAISS_INDEX=true in your .env and restart the app.")
        st.info("Or delete the 'faiss_index' folder and click Process to rebuild the index from your PDFs.")
        return

    try:
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Similarity search failed: {e}")
        return

    if not docs:
        st.info("No relevant documents found for this question.")
        return

    chain = get_conversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
    except Exception as e:
        st.error(f"QA chain failed: {e}")
        return

    st.write("Reply:", response.get("output_text", "No output from model"))


# streamlit app interface
def main():
    st.set_page_config(page_title="PDF Chatbot with Gemini Pro", page_icon=":books:")
    st.header("PDF Chatbot with Gemini Pro :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("PDF Chatbot with Gemini Pro")
        st.markdown("## Upload your PDF files here")

        pdf_docs = st.file_uploader("Upload your PDF documents here and click on process", accept_multiple_files=True)

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF document.")
            else:
                with st.spinner("Extracting text from PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)

                    if text_chunks:
                        try:
                            get_vector_store(text_chunks)
                        except Exception as e:
                            msg = str(e).lower()
                            if "quota" in msg or "limit" in msg or "exceeded" in msg:
                                st.error("API quota or limit exceeded while creating embeddings.")
                                st.info("Options: enable billing/request quota increase for Google, or set USE_LOCAL_EMBEDDINGS=true to use a local HF model.")
                            else:
                                st.error(f"Failed to create vector index: {e}")
                        else:
                            st.success("Processing complete! You can now ask questions about your documents.")
                    else:
                        st.warning("No text chunks were created from the uploaded documents.")


if __name__ == "__main__":
    main()
