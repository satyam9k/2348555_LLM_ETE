import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import os

def load_models():
    print("Loading translation models...")
    tokenizer_translation = AutoTokenizer.from_pretrained("./models/translation_tokenizer")
    model_translation = AutoModelForSeq2SeqLM.from_pretrained("./models/translation_model")
    print("Models loaded successfully.")
    return tokenizer_translation, model_translation

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def translate_text(text, target_language, tokenizer, model):
    # Handle different target languages if needed
    if target_language == "Malayalam":
        target_language_code = "ml"
    elif target_language == "Hindi":
        target_language_code = "hi"
    elif target_language == "Tamil":
        target_language_code = "ta"
    elif target_language == "Telugu":
        target_language_code = "te"
    elif target_language == "Marathi":
        target_language_code = "mr"
    elif target_language == "Gujarati":
        target_language_code = "gu"
    elif target_language == "Kannada":
        target_language_code = "kn"
    elif target_language == "Bengali":
        target_language_code = "bn"
    else:
        raise ValueError(f"Unsupported language: {target_language}")

    # Tokenize the input text
    inputs = tokenizer(f"translate English to {target_language_code}: {text}", return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate the translation
    translated = model.generate(**inputs, max_length=1024)
    
    # Decode the translated text
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def main():
    tokenizer, model = load_models()
    
    # Define supported Indian languages
    supported_languages = {
        "1": "Hindi",
        "2": "Malayalam",
        "3": "Tamil",
        "4": "Telugu",
        "5": "Marathi",
        "6": "Gujarati",
        "7": "Kannada",
        "8": "Bengali"
    }
    
    pdf_path = input("Enter the path to the PDF file: ")
    
    if not os.path.exists(pdf_path):
        print("Error: The specified PDF file does not exist.")
        return
    
    print("\nSupported Indian languages:")
    for key, value in supported_languages.items():
        print(f"{key}. {value}")
    
    language_choice = input("\nChoose the target language (enter the number): ")
    while language_choice not in supported_languages:
        language_choice = input("Invalid choice. Please enter a valid number: ")
    
    target_language = supported_languages[language_choice]
    
    text = extract_text_from_pdf(pdf_path)
    
    print(f"\nTranslating to {target_language}...")
    translated_text = translate_text(text, target_language, tokenizer, model)
    
    print("\nTranslated text:")
    print(translated_text)
    
    # Save the translated text to a file
    output_file = f"translated_{os.path.basename(pdf_path)}.txt"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translated_text)
    print(f"\nTranslation saved to {output_file}")

if __name__ == "__main__":
    main()
