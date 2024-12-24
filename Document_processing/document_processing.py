import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import getpass
import os
from langchain_community.document_loaders import PyPDFLoader
import asyncio
from Database.unique_id_generator import PDFIdentifier
from pinecone_vector_database.index_creator import create_index
from DASHBOARD.add_one_column import add_one_to_column
from dotenv import load_dotenv

load_dotenv()
pdf_identifier = PDFIdentifier()

start_time = time.time()



def document_chunking_and_uploading_to_vectorstore(filepath):
    try:

        id1 = pdf_identifier.generate_id(filepath) # generate unique id

        create_index(id1) # create index with unique idc

        add_one_to_column(id1)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"]) # get api key
        index = pc.Index(id1) # get index

        vector_store = PineconeVectorStore(embedding=embeddings, index=index)

        # Define document loader
        filepath = filepath # file of pdf document

        loader = PyPDFLoader(filepath)
        async def load_pages(loader):
            pages = []
            async for page in loader.alazy_load():
                # PyPDFLoader automatically includes page numbers in metadata
                # We can verify the metadata here
                # Adjust the page number by adding 1 to start from 1 instead of 0
                page.metadata['page'] = page.metadata['page'] + 1
                print(f"Loaded page {page.metadata['page']} with metadata: {page.metadata}")
                pages.append(page)
            return pages

        docs = asyncio.run(load_pages(loader))
        
        # Configure text splitter to preserve metadata during splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            add_start_index=True,  # This will help track chunk positions
        )
        
        all_splits = text_splitter.split_documents(docs)
        
        # Verify metadata in splits
        for split in all_splits:
            if 'page' not in split.metadata:
                print(f"Warning: page number missing in split metadata: {split.metadata}")
        
        # Add documents to vector store with metadata
        vector_store.add_documents(documents=all_splits)
        
        # Print some statistics
        print(f"Processed {len(docs)} pages into {len(all_splits)} chunks")
        return f"This PDF ID is: {id1}"
    except Exception as e:
        print(f"An error occurred: {str(e)}")



# new_file = document_chunking_and_uploading_to_vectorstore(filepath="SC538.pdf")
# print(new_file)