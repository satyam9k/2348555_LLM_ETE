import PyPDF2
import os

def pdf_to_text(pdf_path, output_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Extract text from each page
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
    # Write the extracted text to a .txt file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(text)

def process_pdf():
    # Get input PDF file path
    pdf_path = input("Enter the path to the PDF file: ")
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print("Error: The specified PDF file does not exist.")
        return False
    
    # Generate output .txt file path
    output_path = os.path.splitext(pdf_path)[0] + '.txt'
    
    # Convert PDF to text
    pdf_to_text(pdf_path, output_path)
    
    print(f"Conversion complete. Text saved to: {output_path}")
    return True

def main():
    while True:
        if process_pdf():
            # Ask if the user wants to process another PDF
            another = input("Do you want to process another PDF? (yes/no): ").lower()
            if another != 'yes':
                break
        else:
            # If there was an error, ask if they want to try again
            retry = input("Do you want to try another PDF? (yes/no): ").lower()
            if retry != 'yes':
                break
    
    print("Thank you for using the PDF to Text converter!")

if __name__ == "__main__":
    main()