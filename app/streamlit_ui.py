import streamlit as st
import os
from dotenv import load_dotenv

# Import our custom modules (assume they are in the python path)
# In a real setup, we might use sys.path.append or install the package
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.rag.rag_engine import RAGEngine, RAGConfig
from core.agents.agent_manager import AgentManager, AgentConfig
from core.finetuning.lora_trainer import LoraTrainer, LoraTrainingConfig

# Page config
st.set_page_config(
    page_title="Advanced GenAI Lab",
    page_icon="🤖",
    layout="wide",
)

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# Initialize session state for our components
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'agent_manager' not in st.session_state:
    st.session_state.agent_manager = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
    st.divider()
    
    mode = st.radio("Lab Mode", ["RAG & Semantic Search", "Agentic Workflows", "PEFT/LoRA Training"])
    
    st.info("Enterprise-grade GenAI PEFT/RAG Lab. Modular architecture with Pydantic configuration.")

# Main content
st.title(f"🚀 {mode}")

if mode == "RAG & Semantic Search":
    st.subheader("Context-Aware Retrieval")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Document Indexing")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
        
        if uploaded_file:
            # Save temporary file
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            if st.button("Index Document"):
                with st.spinner("Indexing..."):
                    try:
                        config = RAGConfig()
                        st.session_state.rag_engine = RAGEngine(config)
                        st.session_state.rag_engine.load_and_index(uploaded_file.name)
                        st.success("Indexing complete!")
                        os.remove(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with col2:
        st.markdown("### Ask your Documents")
        query = st.text_input("Enter your question here")
        
        if query:
            if st.session_state.rag_engine:
                with st.spinner("Retrieving answer..."):
                    response = st.session_state.rag_engine.query(query)
                    st.markdown("**Answer:**")
                    st.write(response["result"])
                    
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Source {i+1}**")
                            st.write(doc.page_content[:500] + "...")
            else:
                st.warning("Please index a document first.")

elif mode == "Agentic Workflows":
    st.subheader("Multi-Tool Agentic Reasoning")
    
    if not st.session_state.agent_manager:
        config = AgentConfig()
        st.session_state.agent_manager = AgentManager(config)
        
    chat_container = st.container()
    
    with chat_container:
        user_input = st.chat_input("Ask the agent anything...")
        if user_input:
            with st.spinner("Agent is thinking..."):
                response = st.session_state.agent_manager.run(user_input)
                
    # Display chat history
    for msg in st.session_state.agent_manager.memory.chat_memory.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

elif mode == "PEFT/LoRA Training":
    st.subheader("Fine-Tuning Configuration (LoRA/QLoRA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Selection")
        model_id = st.selectbox("Base Model", ["mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-hf"])
        r = st.slider("LoRA r (Rank)", 4, 64, 8)
        alpha = st.slider("LoRA Alpha", 8, 128, 32)
        
    with col2:
        st.markdown("### Training Hyperparameters")
        lr = st.number_input("Learning Rate", value=2e-4, format="%.4f")
        epochs = st.number_input("Epochs", 1, 10, 3)
        batch_size = st.number_input("Batch Size", 1, 32, 4)
        
    if st.button("Start Fine-Tuning"):
        config = LoraTrainingConfig(
            model_name=model_id,
            r=r,
            lora_alpha=alpha,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size
        )
        trainer = LoraTrainer(config)
        st.info(f"Initializing training for {model_id}...")
        # trainer.setup() # Actual model loading would happen here
        st.warning("Note: Full training requires GPU environment. Logic initialized successfully.")
        st.success("LoRA adapters saved successfully (Mocked).")
