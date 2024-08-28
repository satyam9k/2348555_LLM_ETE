import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import PyPDF2
import os

def load_models():
    print("Loading models from local directory...")
    
    # Load the tokenizer and model from the local directory
    tokenizer_qa = AutoTokenizer.from_pretrained("./models/qa_tokenizer")
    model_qa = AutoModelForQuestionAnswering.from_pretrained("./models/qa_model")
    
    print("Models loaded successfully.")
    return tokenizer_qa, model_qa

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_input(text, question, tokenizer, max_length=512):
    inputs = tokenizer.encode_plus(
        question, text,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    return inputs

def answer_question(context, question, tokenizer, model):
    inputs = preprocess_input(context, question, tokenizer)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax() + 1
    
    # Decode the tokens to get the answer
    answer_tokens = inputs['input_ids'][0][answer_start_index:answer_end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer

def main():
    # Load the tokenizer and model from the local directory
    tokenizer, model = load_models()
    
    pdf_path = input("Enter the path to the PDF file: ")
    
    if not os.path.exists(pdf_path):
        print("Error: The specified PDF file does not exist.")
        return
    
    text = extract_text_from_pdf(pdf_path)
    
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        answer = answer_question(text, question, tokenizer, model)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
