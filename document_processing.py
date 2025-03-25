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
    Safely process PDF pages with proper resource management
    """
    try:
        pages = loader.load()
        for page in pages:
            page.metadata['page'] = page.metadata['page'] + 1
        return pages
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise
    finally:
        if hasattr(loader, 'pdf_reader'):
            loader.pdf_reader.stream.close()

def document_chunking_and_uploading_to_vectorstore(link, name_space):
    """
    Process PDF document in batches to reduce memory usage
    """
    vector_store = None
    
    try:
        with safe_pdf_download(link) as pdf_path:
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

            # Load and process PDF
            loader = PyPDFLoader(file_path=pdf_path)
            docs = process_pdf_safely(loader)

            # Configure text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                add_start_index=True,
            )

            # Process documents in batches
            batch_size = 10  # Adjust based on your server's memory capacity
            total_chunks = 0
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                splits = text_splitter.split_documents(batch)
                if splits:
                    vector_store.add_documents(documents=splits)
                    total_chunks += len(splits)
                    print(f"Processed batch {i//batch_size + 1}: {len(batch)} pages into {len(splits)} chunks")
                # Optional: Force garbage collection to release memory
                gc.collect()

            if total_chunks > 0:
                print(f"Processed {len(docs)} pages into {total_chunks} chunks")
                return f"This PDF ID is: {name_space}"
            else:
                raise ValueError("No document splits were created")

    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

    finally:
        if vector_store and hasattr(vector_store, 'close'):
            vector_store.close()
