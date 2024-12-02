# 🤖 CopilotBot: Your AI Coding Assistant 💻
----------------------------------------------------------
### Research Report: [github-copilot.pdf](https://raw.githubusercontent.com/goushaa/github-copilot-chatbot/main/github-copilot.pdf)
----------------------------------------------------------

## What is CopilotBot?
CopilotBot 🤖 is an intelligent assistant designed to provide detailed insights into GitHub Copilot's performance, functionality, and limitations. It enables users to ask queries about AI-assisted programming and receive responses based on empirical research data.

## How It Is Built?
**CopilotBot** leverages the following technologies:

- **Streamlit**: Provides a user-friendly interface for real-time interaction.
- **LangChain**: Seamlessly integrates language models (LLMs) with a vector database.
- **RecursiveCharacterTextSplitter**: Splits the research PDF into manageable chunks for effective indexing.
- **Hugging Face's sentence-transformers/all-MiniLM-L6-v2**: Generates precise embeddings for semantic understanding.
- **Chroma Vector Database**: Efficiently stores and retrieves embeddings for query handling.
- **Meta-Llama-3-8B-Instruct**: Powers natural language understanding for answering queries.
- **Custom Prompt Engineering**: Guides the model to deliver insightful and contextually accurate responses.

## Architecture Overview
![CopilotBot Architecture](./architecture.png)

## Setup and Execution

### Step 1: Install Dependencies
Run the following command to install the required libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Step 2: Configure API Keys

Add your **HuggingFace API Token** to the `.env` file. If the file doesn’t exist in the project directory, create it and include the following line:

```bash
HUGGINGFACEHUB_API_TOKEN=<your_api_token>
```

## Step 3: Generate Vector Embeddings

Run the following script to split the research paper, generate embeddings, and store them in the vector database:

```bash
python embeddings_generator.py  
```

Step 4: Launch CopilotBot

Start the CopilotBot Streamlit app by running:

```bash
streamlit run app.py  
```

## Features and Examples

**CopilotBot** is equipped to handle a variety of queries and provide insightful responses based on the indexed document:

### Key Features:
1. **Performance Analysis**: Understand how GitHub Copilot impacts development workflows.
2. **Comparative Insights**: Compare Copilot with other AI coding assistants.
3. **Best Practices**: Learn optimal strategies for integrating Copilot into software development processes.
4. **Limitations**: Explore the challenges and constraints of using Copilot for different scenarios.

### Example Queries:
- "_How effective is GitHub Copilot in reducing development time?_"
- "_What are the benefits of using Copilot in collaborative coding?_"
- "_What are the limitations of Copilot in debugging large-scale applications?_"
- "_Can Copilot assist with code optimization tasks?_"

### Output Categories:
- **Relevant Responses**: Provides concise answers based on the document.
- **Out-of-Scope Queries**: Politely informs the user if a query is unrelated to the research content.

**Out-of-Scope Example:**
- "_What is the future of AI in global software development?_"
