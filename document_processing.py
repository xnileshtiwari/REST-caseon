from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from add_one_column import add_one_to_column
from index_creator import create_index
import requests
from contextlib import contextmanager
import tempfile
import gc

@contextmanager
def safe_pdf_download(url):
    """Safely download a PDF and clean up the temporary file."""
    temp_file = None
    response = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        yield temp_file.name
    finally:
        if response:
            response.close()
        if temp_file:
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

def process_pdf_safely(loader):
    """Yield PDF pages one-by-one with adjusted page numbers."""
    try:
        for page in loader.lazy_load():
            page.metadata['page'] += 1  # Adjust to 1-based page numbers
            yield page
    finally:
        if hasattr(loader, 'pdf_reader') and hasattr(loader.pdf_reader, 'stream'):
            loader.pdf_reader.stream.close()

def document_chunking_and_uploading_to_vectorstore(link, name_space):
    """Process PDF page-by-page, upload chunks in batches, and manage memory."""
    vector_store = None
    loader = None
    text_splitter = None
    accumulated_splits = []
    batch_size = 100  # Batch size for vector store uploads
    page_count = 0
    chunk_count = 0
    
    try:
        with safe_pdf_download(link) as pdf_path:
            # Initialize vector store and embeddings
            index_name_from_env = os.environ["INDEX_NAME"]
            create_index(index_name_from_env)
            add_one_to_column(name_space)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            index = pc.Index(os.environ["INDEX_NAME"])
            vector_store = PineconeVectorStore(
                embedding=embeddings,
                index=index,
                namespace=name_space
            )

            # Set up loader and splitter
            loader = PyPDFLoader(file_path=pdf_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                add_start_index=True,
            )

            # Process pages incrementally
            for page in process_pdf_safely(loader):
                page_count += 1
                splits = text_splitter.split_documents([page])
                accumulated_splits.extend(splits)
                chunk_count += len(splits)
                
                if len(accumulated_splits) >= batch_size:
                    vector_store.add_documents(documents=accumulated_splits)
                    del accumulated_splits
                    accumulated_splits = []
                
                del splits
                del page

            # Upload remaining splits
            if accumulated_splits:
                vector_store.add_documents(documents=accumulated_splits)
                del accumulated_splits

            print(f"Processed {page_count} pages into {chunk_count} chunks")
            return f"This PDF ID is: {name_space}"

    finally:
        if vector_store and hasattr(vector_store, 'close'):
            vector_store.close()
        if loader:
            del loader
        if text_splitter:
            del text_splitter
        if vector_store:
            del vector_store
        gc.collect()
