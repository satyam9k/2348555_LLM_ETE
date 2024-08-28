import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import os

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Get embedding for a given text using the provided model and tokenizer
def get_embedding(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Compute similarity between two PDF files
def compute_similarity(pdf_path1, pdf_path2, tokenizer, model):
    text1 = extract_text_from_pdf(pdf_path1)
    text2 = extract_text_from_pdf(pdf_path2)
    
    embedding1 = get_embedding(text1, tokenizer, model)
    embedding2 = get_embedding(text2, tokenizer, model)
    
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

def main():
    # Load the tokenizer and model from the local directory
    tokenizer_plagiarism = AutoTokenizer.from_pretrained("./models/plagiarism_tokenizer")
    model_plagiarism = AutoModel.from_pretrained("./models/plagiarism_model")
    
    pdf_path1 = input("Enter the path to the first PDF file: ")
    pdf_path2 = input("Enter the path to the second PDF file: ")
    
    if not os.path.exists(pdf_path1) or not os.path.exists(pdf_path2):
        print("Error: One or both of the specified PDF files do not exist.")
        return
    
    similarity = compute_similarity(pdf_path1, pdf_path2, tokenizer_plagiarism, model_plagiarism)
    
    print(f"PDF 1: {pdf_path1}")
    print(f"PDF 2: {pdf_path2}")
    print(f"Similarity score: {similarity:.4f}")

if __name__ == "__main__":
    main()
