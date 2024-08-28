import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import os

def load_models():
    print("Loading models from local directory...")
    
    # Load the tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained("./models/pegasus_tokenizer")
    model = AutoModelForSeq2SeqLM.from_pretrained("./models/pegasus_model")
    
    print("Models loaded successfully.")
    return tokenizer, model

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def summarize_text(text, tokenizer, model, max_length):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    # Load the tokenizer and model from the local directory
    tokenizer, model = load_models()
    
    pdf_path = input("Enter the path to the PDF file: ")
    
    if not os.path.exists(pdf_path):
        print("Error: The specified PDF file does not exist.")
        return
    
    text = extract_text_from_pdf(pdf_path)
    
    summary_type = input("Choose summary type (long/short): ").lower()
    while summary_type not in ['long', 'short']:
        summary_type = input("Invalid choice. Please enter 'long' or 'short': ").lower()
    
    max_length = 500 if summary_type == 'long' else 200
    
    summary = summarize_text(text, tokenizer, model, max_length)
    
    print(f"\nSummary ({summary_type}, approximately {max_length} words):")
    print(summary)
    
    # Count words in the summary
    word_count = len(summary.split())
    print(f"\nActual word count: {word_count}")

if __name__ == "__main__":
    main()
