from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

def setup_rag_components(pinecone_api_key, pinecone_index_name):
    """Set up and return the RAG components."""
    # Initialize LLMs
    query_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )
    
    kernel_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7
    )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    
    # Set up vector store and retriever with improved error handling
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Get the index instance
        index = pc.Index(name=pinecone_index_name)
        
        # Check for namespaces in the index
        stats = index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        active_namespace = namespaces[0] if namespaces else None
        
        print(f"Index contains {stats.get('total_vector_count', 0)} vectors")
        
        if active_namespace:
            print(f"Using namespace: {active_namespace}")
            # Create the vector store using the index with namespace
            vectorstore = PineconeVectorStore(
                embedding=embeddings,
                index=index,
                namespace=active_namespace
            )
        else:
            # Create the vector store without namespace
            vectorstore = PineconeVectorStore(
                embedding=embeddings,
                index=index
            )
        
        # Create retriever with increased k to ensure we get results
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Increased from 2 to 5 to get more results
        )
        
        # Test the retriever with a simple query to validate it works
        test_results = retriever.invoke("language apis")
        if test_results:
            print(f"Successfully connected to Pinecone and retrieved {len(test_results)} documents")
        else:
            print("Connected to Pinecone but retrieval returned no results - continuing anyway")
        
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        print("Falling back to dummy retriever")
        # Create a dummy retriever that returns empty results
        class DummyRetriever:
            def invoke(self, _):
                return []
        
        retriever = DummyRetriever()
    
    # Create the query generation chain
    query_generation_prompt = ChatPromptTemplate.from_template(
        "Identify key technical concepts for NKI kernel. Be brief (max 100 words).\n\n"
        "What technical concepts should I retrieve for this kernel task? {user_prompt}"
    )
    
    query_generation_chain = (
        query_generation_prompt 
        | query_llm 
        | StrOutputParser()
    )
    
    return query_generation_chain, retriever, kernel_llm


def format_context(docs):
    """Format the retrieved documents into a context string."""
    context = ""
    for i, doc in enumerate(docs):
        context += f"Doc{i+1}: "
        
        # Get content
        content = doc.page_content
        metadata = doc.metadata
        
        # Get title from metadata if available
        title = metadata.get('title', 'No title')
        
        # Check if content is too long
        if len(content) > 500:
            content = content[:500] + "..."
            
        context += f"{title} - {content}\n\n"
        
    if not context:
        context = "No relevant documents found."
        
    return context
