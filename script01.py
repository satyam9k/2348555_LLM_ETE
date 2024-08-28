import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import os

# Load models
@st.cache_resource
def load_models():
    # Load plagiarism model
    tokenizer_plagiarism = AutoTokenizer.from_pretrained("./models/plagiarism_tokenizer")
    model_plagiarism = AutoModel.from_pretrained("./models/plagiarism_model")
    
    # Load summarization model
    tokenizer_summary = AutoTokenizer.from_pretrained("./models/pegasus_tokenizer")
    model_summary = AutoModelForSeq2SeqLM.from_pretrained("./models/pegasus_model")
    
    # Load translation model
    tokenizer_translation = AutoTokenizer.from_pretrained("./models/translation_tokenizer")
    model_translation = AutoModelForSeq2SeqLM.from_pretrained("./models/translation_model")
    
    # Load QA model
    tokenizer_qa = AutoTokenizer.from_pretrained("./models/qa_tokenizer")
    model_qa = AutoModelForQuestionAnswering.from_pretrained("./models/qa_model")
    
    return (tokenizer_plagiarism, model_plagiarism, 
            tokenizer_summary, model_summary, 
            tokenizer_translation, model_translation, 
            tokenizer_qa, model_qa)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Similarity Calculation
def compute_similarity(text1, text2, tokenizer, model):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(text, tokenizer, model):
        encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        return mean_pooling(model_output, encoded_input['attention_mask'])
    
    embedding1 = get_embedding(text1, tokenizer, model)
    embedding2 = get_embedding(text2, tokenizer, model)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

# Summarization
def summarize_text(text, tokenizer, model, max_length):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Translation
def translate_text(text, target_language, tokenizer, model):
    inputs = tokenizer(f"translate English to {target_language}: {text}", return_tensors="pt", truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# QA
def answer_question(context, question, tokenizer, model):
    inputs = tokenizer.encode_plus(
        question, context,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax() + 1
    answer_tokens = inputs['input_ids'][0][answer_start_index:answer_end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

# Streamlit App
def main():
    st.title("📚 ScholarAssist 🔍")
    st.write("""
    **ScholarAssist** is a versatile tool that allows you to perform various tasks on text extracted from PDF files. 
    You can check the similarity between documents, summarize content, translate text into multiple languages, 
    and even get answers to questions based on the provided document. 
    This application leverages advanced LLM models to enhance your document processing capabilities.
    """)
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.write("Text extracted from PDF:")
        st.write(text[:2000])  # Show a preview of the extracted text
    
        (tokenizer_plagiarism, model_plagiarism,
         tokenizer_summary, model_summary,
         tokenizer_translation, model_translation,
         tokenizer_qa, model_qa) = load_models()

        operation = st.selectbox("Choose an operation", ["Similarity", "Summarization", "Translation", "Question Answering"])

        if operation == "Similarity":
            uploaded_file2 = st.file_uploader("Choose a second PDF file", type="pdf")
            if uploaded_file2:
                text2 = extract_text_from_pdf(uploaded_file2)
                similarity = compute_similarity(text, text2, tokenizer_plagiarism, model_plagiarism)
                st.write(f"Similarity score: {similarity:.4f}")

        elif operation == "Summarization":
            summary_type = st.selectbox("Summary type", ["Long", "Short"])
            max_length = 500 if summary_type == "Long" else 200
            summary = summarize_text(text, tokenizer_summary, model_summary, max_length)
            st.write("Summary:")
            st.write(summary)

        elif operation == "Translation":
            target_language = st.selectbox("Target language", ["Hindi", "Malayalam", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Bengali"])
            translated_text = translate_text(text, target_language, tokenizer_translation, model_translation)
            st.write("Translated text:")
            st.write(translated_text)

        elif operation == "Question Answering":
            question = st.text_input("Enter your question")
            if question:
                answer = answer_question(text, question, tokenizer_qa, model_qa)
                st.write("Answer:")
                st.write(answer)

    # Collapsible Sidebar for User Input
    with st.sidebar:
        st.header("Text Input & Download")
        st.write("You can type text here and download it locally.")
        
        user_text = st.text_area("Type your text here...")
        if st.button("Download Text"):
            with open("user_text.txt", "w") as f:
                f.write(user_text)
            st.success("Text saved as user_text.txt")

if __name__ == "__main__":
    main()
