import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the vector store from disk
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens": 1024},
)

prompt_template = """
As a knowledgeable assistant focused on AI-assisted programming, your role is to accurately interpret programming-related queries 
and provide insightful responses based on the findings from the empirical study on GitHub Copilot. Follow these directives:
1. Precision in Answers: Respond solely with information directly relevant to the user's query about GitHub Copilot's capabilities.
2. Areas of Focus: Address queries related to:
    - Performance and Functionality of GitHub Copilot
    - Comparison with Human Programmers
    - Best Practices for Using AI Tools in Programming
    - Common Issues and Limitations of Copilot
3. Handling Off-topic Queries: For questions unrelated to the subject or not addressed in the paper, politely inform the user that their query falls outside the scope of the chatbot's knowledge or the document's content.
4. Contextual Accuracy: Ensure responses are directly related to AI-assisted programming, utilizing information from the research paper.
5. Streamlined Communication: Deliver clear, concise, and direct answers without unnecessary comments or sign-offs.

GitHub Copilot Query:
{context}

Question: {question}

Answer:
"""
custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),
    chain_type_kwargs={"prompt": custom_prompt}
)

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app setup
st.markdown(
    """
    <style>
        .appview-container .main .block-container {{
            padding-top: {padding_top}rem;
            padding-bottom: {padding_bottom}rem;
        }}
    </style>""".format(
        padding_top=1, padding_bottom=1
    ),
    unsafe_allow_html=True,
)

st.markdown("""
    <h3 style='text-align: left; color: white; padding-top: 35px; border-bottom: 3px solid blue;'>
        GitHub Copilot Insights 🤖
    </h3>""", unsafe_allow_html=True)

side_bar_message = """
Hi there! 👋 I'm CopilotBot, your assistant for exploring GitHub Copilot's capabilities. All insights I provide are based on 
the research paper by Dr. Zhen Ming and collaborators. Here’s what I can help with:

1. 🚀 **Performance**: How well does Copilot solve problems?
2. 👨‍💻 **Comparison**: Copilot vs. human programmers.
3. 📚 **Best Practices**: Tips for effective AI-assisted coding.
4. ⚠️ **Limitations**: Common issues and workarounds.

Ask me anything about GitHub Copilot and AI-assisted programming!
"""


with st.sidebar:
    st.title('🤖 CopilotBot: Your AI Coding Assistant')
    st.markdown(side_bar_message)

initial_message = """
Hi! 🤖 I'm CopilotBot, here to provide insights on GitHub Copilot and AI-assisted programming. 
You can ask about:

- 🚀 How does Copilot perform on algorithmic tasks?
- 👨‍💻 How does it compare to human developers?
- 📚 Best practices for using Copilot effectively.
- ⚠️ Common issues and how to address them.

Let’s get started—ask away!
"""

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching insights about GitHub Copilot for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = response  # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
