import os
from pinecone import Pinecone
from openai import OpenAI  # Or another embedding service
import time

# Initialize OpenAI client
os.environ["OPENAI_API_KEY"] = "##"
openai_client = OpenAI()

# Initialize Pinecone
pc = Pinecone(api_key="##")
index = pc.Index(host="https://nkidocs-7ygnls2.svc.aped-4627-b74a.pinecone.io")

# Function to get embeddings
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # Choose appropriate model
    )
    return response.data[0].embedding

# 3. Prepare a list to store vectors to upsert
vectors_to_upsert = []

# 4. Loop through all .txt files in your folder
folder_path = "/Users/rgopalam/Desktop/AWS-NKI/torch2nki/documentation/nki_documentation/nki_language_apis_parsed"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # Read the contents of the text file
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        
        # Generate embedding for the text
        embedding_vector = get_embedding(text_content)
        
        # 5. Add the embedding vector and metadata
        vectors_to_upsert.append({
            "id": filename,  # Use filename or a unique ID
            "values": embedding_vector,  # This is the embedding vector
            "metadata": {
                "text": text_content,  # Store the original text in metadata
                "filename": filename
            }
        })
        
        # Add a small delay to avoid rate limits
        time.sleep(0.5)

# 6. Upsert the vectors into Pinecone
# Process in batches if you have many documents
batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i+batch_size]
    index.upsert(
        vectors=batch,
        namespace="nki language apis"  # Optional, specify if you use namespaces
    )
    print(f"Upserted batch {i//batch_size + 1} into Pinecone.")

print(f"Upserted {len(vectors_to_upsert)} text files into Pinecone.")