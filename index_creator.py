import os
from pinecone import Pinecone, PineconeApiException
from pinecone import ServerlessSpec


# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# Create Index
def create_index(index_name):

    try:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        return print(f"Index '{index_name}' created.")
    except PineconeApiException as e:
        if e.status == 409 and "ALREADY_EXISTS" in e.body:
            return print("Resource already exists")
        else:
            raise
