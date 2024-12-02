import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the PDF
loader = PyPDFLoader("github-copilot.pdf")
documents = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert texts to embeddings
try:
    embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
    print("Vector Embeddings created successfully")
except Exception as e:
    print(f"Error creating vector embeddings: {e}")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")

# Add documents to the vector store
vector_store.add_documents(documents=texts)

# Validate the setup
try:
    # Test queries to validate data retrieval
    test_queries = [
        "What are the limitations of AI code completion tools in programming?",
        "How does code suggestion impact novice programmers' learning?"
    ]

    for test_query in test_queries:
        results = vector_store.search(query=test_query, search_type='similarity')

        # Deduplicate results
        unique_results = OrderedDict()
        for doc in results:
            if doc.page_content not in unique_results:
                unique_results[doc.page_content] = doc

        # Convert unique results to a list and limit to top 3
        final_results = list(unique_results.values())[:3]
        print(f"Unique query results for '{test_query}': {final_results}")

except Exception as e:
    print(f"Error during test query: {e}")