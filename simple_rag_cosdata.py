import ollama
from cosdata.client import Client
from markitdown import MarkItDown
from cosdata.index import Index


dataset = []

with open('cat_facts.txt', 'r') as file:
   dataset = file.readlines()
   print(f'Loaded {len(dataset)} entries')


# print("dataset", dataset)

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'


client = Client(
        host="http://127.0.0.1:8443"
    )

def get_collection(vector_db_name, dimension=768, description="Rag app"):
    """
    Try to get an existing collection, create it if it doesn't exist.
    Returns a tuple of (collection, is_new_collection)
    """
    is_new_collection = False
    try:
        # Try to get the existing collection
        collection = client.get_collection(vector_db_name)
        print(f"Collection '{vector_db_name}' already exists")
    except:
        # Create a new collection if it doesn't exist
        collection = client.create_collection(
            name=vector_db_name,
            dimension=dimension,
            description=description
        )
        print(f"Collection '{vector_db_name}' created successfully")
        is_new_collection = True
    
    return collection, is_new_collection

# Get or create the collection
simple_rag_collection, is_new_collection = get_collection("simple_rag", 768, "Rag app")

print("collection details ", simple_rag_collection)

all_embeddings = []

def add_chunk_to_database(chunk, i):
  if chunk != "":
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    all_embeddings.append({
        "id": i, 
        "values": embedding
    })

# Only process chunks if we're working with an existing collection
if is_new_collection:
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk, i)
        print(f'Added chunk {i+1}/{len(dataset)} to the database')

    index = simple_rag_collection.create_index(
        distance_metric="cosine"
    )

    # Only upsert if we have embeddings
    if all_embeddings:
        with index.transaction() as txn:
            txn.upsert(all_embeddings)
            print(f"Upserting complete - all vectors inserted in a single transaction")
else:
    index = Index(client, simple_rag_collection)


def retrieve(query, top_n=3):
   query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
   results = index.query(
     vector=query_embedding,
     nn_count=20
   )
   print(f"Query results: {results}")
   return results


# Chatbot
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)
