from openai import OpenAI
from pinecone import Pinecone
import os
import time

def test_rag_query_generation(system_prompt_path, user_prompt_path, output_path, 
                             pinecone_api_key, pinecone_index_name):
    # Initialize clients
    client = OpenAI()
    
    # Load prompts with moderate limits
    system_prompt = open(system_prompt_path, "r").read()[:800]
    user_prompt = open(user_prompt_path, "r").read()[:500]
    
    print(f"Starting RAG process for: {user_prompt[:50]}...")
    
    # 1. Generate query - THIS IS THE PART YOU WANT TO TEST
    try:
        print("Generating retrieval query...")
        query_response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using 3.5 to save tokens
            messages=[
                {"role": "system", "content": "Identify key technical concepts for CUDA kernel. Be brief (max 100 words)."},
                {"role": "user", "content": f"What technical concepts should I retrieve for this kernel task? {user_prompt}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        retrieval_query = query_response.choices[0].message.content
        print(f"Query generated: {retrieval_query[:100]}...")
        
        # 2. Get embedding
        print("Getting embedding...")
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=retrieval_query
        )
        query_embedding = embedding_response.data[0].embedding
        
        # 3. Query Pinecone
        print("Querying Pinecone...")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # This line is causing the error - need to pass name explicitly, not index_name
        index = pc.Index(name=pinecone_index_name)
        
        results = index.query(
            vector=query_embedding,
            top_k=2,
            include_metadata=True
        )
        
        # 4. Format context
        print(f"Retrieved {len(results.matches)} documents.")
        context = ""
        for i, doc in enumerate(results.matches):
            context += f"Doc{i+1}: "
            
            # Get title and content
            title = doc['metadata'].get('title', 'No title')
            content = doc['metadata'].get('content', 'No content')
            
            # Truncate if needed
            if len(content) > 200:
                content = content[:200] + "..."
                
            context += f"{title} - {content}\n"
            
        if not context:
            context = "No relevant documents found."
            
        # Save the query and context for inspection
        with open(output_path + ".query_log", "w") as f:
            f.write(f"USER PROMPT:\n{user_prompt}\n\n")
            f.write(f"GENERATED QUERY:\n{retrieval_query}\n\n")
            f.write(f"RETRIEVED CONTEXT:\n{context}\n\n")
            
        print(f"Query and context saved to {output_path}.query_log")
            
    except Exception as e:
        print(f"Error during retrieval process: {e}")
        context = "Retrieval process failed. Check query_log for details."
        with open(output_path + ".query_log", "w") as f:
            f.write(f"ERROR: {str(e)}")
    
    # 5. Generate a minimal kernel just to complete the process
    try:
        print("Generating simple kernel...")
        # Minimal generation to save tokens
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using 3.5 to save tokens
            messages=[
                {"role": "system", "content": "You write CUDA kernels."},
                {"role": "user", "content": f"Write a basic kernel for: {user_prompt[:100]}"}
            ],
            temperature=0.7,
            max_tokens=500  # Very limited to avoid rate limits
        )
        
        result = completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating kernel: {e}")
        result = f"Error generating kernel. RAG query process completed, but generation failed: {str(e)}"
    
    # Save kernel result
    with open(output_path, "w") as f:
        f.write(result)
    
    print(f"Process completed. Kernel saved to {output_path}")
    print("Check the query_log file to see the retrieval process results.")

if __name__ == "__main__":
    # Get credentials
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')
    
    if not pinecone_api_key or not pinecone_index_name:
        print("Error: Environment variables not set.")
        exit(1)
    
    # Run test
    test_rag_query_generation(
        "/home/ubuntu/torch2nki/prompts/system_prompt_naive.txt",
        "/home/ubuntu/torch2nki/prompts/user_prompt_add.txt",
        "/home/ubuntu/torch2nki/generation/samples/vector_add.txt",
        pinecone_api_key,
        pinecone_index_name
    )