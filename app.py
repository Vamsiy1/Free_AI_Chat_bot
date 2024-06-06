import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
import time
load_dotenv()


# Initialize the conversation buffer memory
memory = ConversationBufferMemory(return_messages=True)

# Retrieve the API token
#huggingfacehub_api_token = os.getenv('HUGGINGFACE_API_TOKEN')
os.environ['HUGGINGFACE_API_TOKEN'] = st.secrets['HUGGINGFACE_API_TOKEN']
huggingfacehub_api_token = os.environ['HUGGINGFACE_API_TOKEN']                                     
print('VALUE OF TOKEN -->',huggingfacehub_api_token)


# Initialize the HuggingFaceEndpoint with the conversation buffer memory
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
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
    # Custom CSS for styling
    st.markdown("""
        <style>
        .title {
            font-size: 50px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subtitle {
            font-size: 20px;
            color: #FF5722;
        }
        .prompt-box {
            font-size: 18px;
            color: #3F51B5;
        }
        .response-box {
            font-size: 18px;
            color: #E91E63;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title and description
    st.markdown('<p class="title">Conversational AI with AI-cademy 🤖</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Welcome to the Conversational AI interface.</p>', unsafe_allow_html=True)

    # Input box for user prompt
    prompt = st.text_input("Enter your prompt here:", key='prompt', placeholder="Type something...", help="Ask anything you want!")

    if st.button("Ask"):
        if prompt:
            with st.spinner("Thinking..."):
                response = converse(prompt)
            display_response(response)
        else:
            st.warning("Please enter a prompt.")



# Function to display response letter by letter
def display_response(response):
    response_placeholder = st.empty()
    typed_text = ""
    for char in response:
        typed_text += char
        response_placeholder.markdown(f'<p class="response-box">{typed_text}</p>', unsafe_allow_html=True)
        time.sleep(0.01)  # Adjust the delay to control typing speed

if __name__ == "__main__":
    main()
