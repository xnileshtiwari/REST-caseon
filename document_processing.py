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
    """
    Context manager for safely downloading and cleaning up PDF files.
    """
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
            try:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file: {e}")

def process_pdf_safely(loader):
    """
    Generator to process PDF pages one-by-one with proper resource management.
    Yields pages with adjusted page numbers.
    """
    try:
        for page in loader.lazy_load():
            # Adjust page number (from 0-based to 1-based) for citation
            page.metadata['page'] += 1
            yield page
    except Exception as e:
        print(f"Error loading PDF page: {e}")
        raise
    finally:
        # Ensure PDF reader's stream is closed
        if hasattr(loader, 'pdf_reader') and hasattr(loader.pdf_reader, 'stream'):
            loader.pdf_reader.stream.close()

def document_chunking_and_uploading_to_vectorstore(link, name_space):
    """
    Process PDF document page-by-page, upload to vector store, and manage memory.
    """
    vector_store = None
    loader = None
    text_splitter = None
    
    try:
        with safe_pdf_download(link) as pdf_path:
            # Initialize Pinecone and vector store
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

            # Initialize PDF loader and text splitter
            loader = PyPDFLoader(file_path=pdf_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                add_start_index=True,
            )

            # Process PDF page-by-page
            page_count = 0
            chunk_count = 0
            for page in process_pdf_safely(loader):
                page_count += 1
                splits = text_splitter.split_documents([page])
                if splits:
                    vector_store.add_documents(documents=splits)
                    chunk_count += len(splits)
                # Explicitly delete page and splits to free memory
                del splits
                del page

            # Report results
            print(f"Processed {page_count} pages into {chunk_count} chunks")
            return f"This PDF ID is: {name_space}"

    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

    finally:
        # Clean up resources and force garbage collection
        if vector_store and hasattr(vector_store, 'close'):
            vector_store.close()
        # Explicitly delete large objects
        if loader:
            del loader
        if text_splitter:
            del text_splitter
        if vector_store:
            del vector_store
        gc.collect()
