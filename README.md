# ScholarAssist 📚🔍

**ScholarAssist** is a powerful and versatile Streamlit-based application designed to assist scholars, researchers, and students in processing text extracted from PDF documents. The application offers a suite of functionalities, including plagiarism checking, summarization, translation, and question answering.

## Features

- **Similarity Checking**: Compares two documents and calculates a similarity score to detect potential plagiarism or content overlap.
- **Summarization**: Generates concise summaries of documents, with options for long or short summaries.
- **Translation**: Translates extracted text into multiple languages including Hindi, Malayalam, Tamil, Telugu, Marathi, Gujarati, Kannada, and Bengali.
- **Question Answering**: Provides precise answers to user queries based on the content of the uploaded PDF document.

## Models Used

- **Summarization**: [`google/pegasus-arxiv`](https://huggingface.co/google/pegasus-arxiv)
- **Similarity Checking**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Translation**: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **Question Answering**: [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2)

## Installation

1. **Clone the repository**:


2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained models**:
    Place the downloaded models in the `./models/` directory:
    - Plagiarism Model
    - Summarization Model
    - Translation Model
    - Question Answering Model

## Usage

1. **Run the Streamlit application**:
    ```bash
    streamlit run script01.py
    ```

2. **Upload a PDF file**:
    - The application allows you to upload a PDF file and extract text from it.

3. **Select an operation**:
    - Choose from similarity checking, summarization, translation, or question answering.

4. **Process the document**:
    - Depending on the operation selected, the app will perform the desired task and display the results.

## Project Structure

- `script01.py`: Main application script.
- `models/`: Directory to store the pre-trained models. Download from HuggingFace locally.
- `requirements.txt`: List of Python packages required to run the application.

## How It Works

### Loading Models
The models are loaded and cached using Streamlit's `@st.cache_resource` to optimize performance.

### Text Extraction
The `extract_text_from_pdf` function uses PyPDF2 to extract text from the uploaded PDF files.

### Similarity Calculation
The application computes the cosine similarity between document embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model.

### Summarization
Text is summarized using the `google/pegasus-arxiv` model, with options for long or short summaries.

### Translation
The application translates text to the selected language using the `facebook/nllb-200-distilled-600M` model.

### Question Answering
Given a context from the document, the application answers questions using the `deepset/roberta-base-squad2` model.


## License

This project is [MIT licensed](LICENSE).

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the pre-trained models.
- [Streamlit](https://streamlit.io/) for the interactive web framework.
- [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF text extraction.

---

Made with ❤️ by Satyam Kumar(https://github.com/satyam9k)
