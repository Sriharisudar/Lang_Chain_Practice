import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to get response from LLaMA 2 model
def get_llama_response(input_text, no_words, blog_style):
    try:
        # Specify the path to the model file
        model_path = 'models/llama-2-7b-chat.ggmlv3.q2_K.bin'

        # Initialize LLaMA 2 model with CTransformers
        llm = CTransformers(
            model=model_path,
            model_type='llama',
            config={
                'max_new_tokens': 256,
                'temperature': 0.01
            }
        )

        # Define the prompt template
        template = """
        Write a blog for a {blog_style} job profile on the topic '{input_text}' 
        within {no_words} words.
        """
        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", "no_words"],
            template=template
        )

        # Generate response from the model
        prompt_text = prompt.format(
            blog_style=blog_style,
            input_text=input_text,
            no_words=no_words
        )
        response = llm(prompt_text)
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit App Configuration
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Generate Blogs 🤖")

# Input for Blog Topic
input_text = st.text_input("Enter the Blog Topic")

# Create two columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("Number of Words")
with col2:
    blog_style = st.selectbox(
        "Writing the blog for",
        options=["Researchers", "Data Scientists", "Common People"],
        index=0
    )

# Button to trigger blog generation
submit = st.button("Generate")

# Final Response
if submit:
    if input_text and no_words.isdigit():
        with st.spinner("Generating blog..."):
            response = get_llama_response(input_text, no_words, blog_style)
        st.write(response)
    else:
        st.error("Please enter a valid blog topic and number of words.")

