import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

# Retrieve the Hugging Face API token from environment variables
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the conversation buffer memory
memory = ConversationBufferMemory(return_messages=True)

# Initialize the HuggingFaceEndpoint with the conversation buffer memory
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=huggingfacehub_api_token,
)

# Define the function to converse
def converse(prompt):
    # Retrieve context from memory
    context = memory.load_memory_variables({})
    
    # Prepare input with context, ensuring all values are strings
    context_text = ' '.join([str(value) for value in context.values()])
    input_text = context_text + ' ' + prompt if context_text else prompt
    
    # Get response from the model
    response = llm.invoke(input_text)
    
    # Store the new prompt and response in memory
    memory.save_context({"prompt": prompt}, {"response": response})
    
    return response

# Streamlit app
def main():
    st.title("Conversational AI with AI-cademy")
    st.markdown("Welcome to the Conversational AI interface.")
    
    # Input box for user prompt
    prompt = st.text_input("Enter your prompt here:")
    
    if st.button("Ask"):
        if prompt:
            with st.spinner("Thinking..."):
                response = converse(prompt)
            st.text_area("Response:", value=response, height=150)
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()
