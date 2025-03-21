"""
Pinecone Diagnostic Script (Modified to return closest vectors)

This standalone script tests your Pinecone connection and index, returning the closest vectors
regardless of relevance score.
It will:
1. Check your connection to Pinecone
2. Display index statistics 
3. Test a direct query and return top 5 closest vectors
4. Test query through LangChain integration and return top 5 closest documents
"""

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import sys

# ANSI colors for better readability
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

def print_header(message):
    print(f"\n{BOLD}{BLUE}=== {message} ==={ENDC}\n")

def print_success(message):
    print(f"{GREEN}✓ {message}{ENDC}")

def print_warning(message):
    print(f"{YELLOW}⚠ {message}{ENDC}")

def print_error(message):
    print(f"{RED}✖ {message}{ENDC}")

def print_info(message):
    print(f"{BLUE}ℹ {message}{ENDC}")

def main():
    print_header("PINECONE DIAGNOSTIC TOOL")
    
    # Get API key and index name from environment or arguments
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')
    
    if len(sys.argv) > 1:
        pinecone_api_key = sys.argv[1]
    if len(sys.argv) > 2:
        pinecone_index_name = sys.argv[2]
    
    if not pinecone_api_key:
        print_error("No Pinecone API key found! Please set PINECONE_API_KEY environment variable or provide as first argument.")
        return
        
    if not pinecone_index_name:
        print_error("No Pinecone index name found! Please set PINECONE_INDEX_NAME environment variable or provide as second argument.")
        return
    
    print_info(f"Using index name: {pinecone_index_name}")
    
    # Step 1: Test basic connection
    print_header("TESTING PINECONE CONNECTION")
    try:
        print("Initializing Pinecone client...")
        pc = Pinecone(api_key=pinecone_api_key)
        print_success("Successfully initialized Pinecone client")
        
        # List available indexes
        indexes = pc.list_indexes()
        if indexes:
            print_success(f"Found {len(indexes)} indexes: {', '.join([idx.name for idx in indexes])}")
            
            # Check if our target index exists
            if not any(idx.name == pinecone_index_name for idx in indexes):
                print_error(f"Index '{pinecone_index_name}' not found among available indexes!")
        else:
            print_warning("No indexes found in your Pinecone account")
            
    except Exception as e:
        print_error(f"Failed to connect to Pinecone: {e}")
        return
    
    # Step 2: Get index information
    print_header("CHECKING INDEX DETAILS")
    try:
        index = pc.Index(name=pinecone_index_name)
        stats = index.describe_index_stats()
        print_info(f"Index statistics: {stats}")
        
        total_vectors = stats.get('total_vector_count', 0)
        if total_vectors == 0:
            print_error("Your index is empty! This explains why no documents are being retrieved.")
            print_info("You need to populate your vector store with documents before retrieval will work.")
            return
        else:
            print_success(f"Index contains {total_vectors} vectors")
            
            # Check namespaces if available
            if 'namespaces' in stats:
                namespaces = list(stats['namespaces'].keys())
                print_info(f"Index has {len(namespaces)} namespaces: {namespaces}")
                
                # Store namespace for later use
                if namespaces:
                    active_namespace = namespaces[0]
                    print_info(f"Using namespace: {active_namespace} for queries")
                else:
                    active_namespace = None
        
        # Get dimension info
        dimension = stats.get('dimension')
        if dimension:
            print_info(f"Vector dimension: {dimension}")
    except Exception as e:
        print_error(f"Failed to get index details: {e}")
        return
    
    # Step 3: Test direct query
    print_header("TESTING DIRECT QUERY - TOP 5 CLOSEST VECTORS")
    try:
        # Create a test embedding
        print("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        print_success("OpenAI embeddings initialized")
        
        test_query = "language apis"  # Changed to be more relevant to stored content
        print(f"Creating embedding for test query: '{test_query}'")
        test_embedding = embeddings.embed_query(test_query)
        print_success(f"Successfully created embedding with {len(test_embedding)} dimensions")
        
        # Query the index directly
        print("Querying Pinecone index directly...")
        query_params = {
            "vector": test_embedding,
            "top_k": 5,  # Get top 5 closest vectors
            "include_metadata": True,
            "include_values": False
        }
        
        # Add namespace if it exists
        if active_namespace:
            query_params["namespace"] = active_namespace
            
        results = index.query(**query_params)
        
        if results.get('matches'):
            print_success(f"Query returned {len(results['matches'])} results")
            for i, match in enumerate(results['matches']):
                print_info(f"Match {i+1}: Score {match['score']:.4f}")
                if 'metadata' in match and match['metadata']:
                    print_info(f"  Metadata: {match['metadata']}")
                    if 'text' in match['metadata']:
                        # Print a snippet of the text
                        text_snippet = match['metadata']['text'][:100] + "..." if len(match['metadata']['text']) > 100 else match['metadata']['text']
                        print_info(f"  Text snippet: {text_snippet}")
        else:
            print_warning("Query returned no results even when requesting the top 5 closest vectors")
            print_info("This is unusual and might indicate an issue with the index configuration.")
    except Exception as e:
        print_error(f"Failed during direct query test: {e}")
    
    # Step 4: Test LangChain integration
    print_header("TESTING LANGCHAIN INTEGRATION - TOP 5 CLOSEST DOCUMENTS")
    try:
        print("Setting up PineconeVectorStore...")
        vectorstore_params = {
            "embedding": embeddings,
            "index": index
        }
        
        # Add namespace if it exists
        if active_namespace:
            vectorstore_params["namespace"] = active_namespace
            
        vectorstore = PineconeVectorStore(**vectorstore_params)
        print_success("Successfully created PineconeVectorStore")
        
        print("Creating retriever...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Ensure no score threshold filters results
        )
        print_success("Successfully created retriever")
        
        print("Testing retrieval with the query...")
        docs = retriever.invoke(test_query)
        
        if docs:
            print_success(f"LangChain retriever returned {len(docs)} documents")
            for i, doc in enumerate(docs):
                print_info(f"Document {i+1}: {len(doc.page_content)} chars")
                print_info(f"  Content snippet: {doc.page_content[:100]}..." if len(doc.page_content) > 100 else doc.page_content)
                if doc.metadata:
                    print_info(f"  Metadata: {doc.metadata}")
        else:
            print_warning("LangChain retriever returned no documents even with no score threshold")
            print_info("This might indicate an issue with your LangChain configuration.")
    except Exception as e:
        print_error(f"Failed during LangChain integration test: {e}")
    
    # Conclusion
    print_header("DIAGNOSTIC SUMMARY")
    print_info("If you're still seeing empty results, here are possible issues and solutions:")
    print_info("1. Namespace issues: Verify you're querying the correct namespace")
    print_info("2. Embedding mismatch: Ensure you use the same embedding model for storage and retrieval")
    print_info("3. Index configuration: Check if your index settings might be preventing retrieval")
    print_info("4. Permission issues: Ensure your API key has read access to the index")
    print_info("5. Try examining a specific vector ID directly if available")

if __name__ == "__main__":
    main()