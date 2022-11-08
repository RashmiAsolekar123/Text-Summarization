import nltk
import validators
import streamlit as st
from transformers import AutoTokenizer, pipeline

# local modules
import sys
sys.path.append("/content")
from utils import (
    clean_text,
    fetch_article_text,
    preprocess_text_for_abstractive_summarization,
    read_text_from_file,
    bart_summarize,
)

if __name__ == "__main__":
    # ---------------------------------
    # Main Application
    # ---------------------------------
    st.title("Text Summarizer 📝")

    summarize_type = st.sidebar.selectbox(
        "Summarization type", options=["Abstractive"]
    )
    st.markdown(
        "Enter a text or a url to get a concise summary of the article while conserving the overall meaning. This app supports text in the following formats:"
    )
    st.markdown(
        """- Raw text in text box 
- URL of article/news to be summarized 
- .txt, .pdf, .docx file formats"""
    )
    st.markdown(
        """This app supports two type of summarization:
1. *Extractive Summarization*: The extractive approach involves picking up the most important phrases and lines from the documents. It then combines all the important lines to create the summary. So, in this case, every line and word of the summary actually belongs to the original document which is summarized.
2. *Abstractive Summarization*: The abstractive approach involves rephrasing the complete document while capturing the complete meaning of the document. This type of summarization provides more human-like summary"""
    )
    st.markdown("---")
    # ---------------------------
    # SETUP & Constants
    nltk.download("punkt")
    abs_tokenizer_name = "facebook/bart-large-cnn"
    abs_model_name = "facebook/bart-large-cnn"
    abs_tokenizer = AutoTokenizer.from_pretrained(abs_tokenizer_name)
    abs_max_length = 90
    abs_min_length = 30
    # ---------------------------

    inp_text = st.text_input("Enter text or a url here")
    st.markdown(
        "<h3 style='text-align: center; color: green;'>OR</h3>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload a .txt, .pdf, .docx file for summarization"
    )

    is_url = validators.url(inp_text)
    if is_url:
        # complete text, chunks to summarize (list of sentences for long docs)
        text, clean_txt = fetch_article_text(url=inp_text)
    elif uploaded_file:
        clean_txt = read_text_from_file(uploaded_file)
        clean_txt = clean_text(clean_txt)
    else:
        clean_txt = clean_text(inp_text)

    # view summarized text (expander)
    with st.expander("View input text"):
        if is_url:
            st.write(clean_txt[0])
        else:
            st.write(clean_txt)
    summarize = st.button("Summarize")

    # called on toggle button [summarize]

    if summarize:
        if summarize_type == "Abstractive":
            with st.spinner(
                text="Creating abstractive summary. This might take a few seconds ..."
            ):
                text_to_summarize = clean_txt
                abs_summarizer = pipeline(
                    "summarization", model=abs_model_name, tokenizer=abs_tokenizer_name
                )

                if is_url is False:
                    # list of chunks
                    text_to_summarize = preprocess_text_for_abstractive_summarization(
                        tokenizer=abs_tokenizer, text=clean_txt
                    )

                
                tmp_sum2 =  bart_summarize(
                    text_to_summarize,
                    num_beams=4,
                    length_penalty=2.0,
                    max_length=abs_max_length,
                    min_length=abs_min_length,
                    no_repeat_ngram_size=3,
                )




                #summarized_text = " ".join([summ["summary_text"] for summ in tmp_sum2])
                summarized_text = tmp_sum2

        # final summarized output
        st.subheader("Summarized text")
        st.info(summarized_text)
