import string
import pickle
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

# Download NLTK stopwords safely
try:
    nltk.download("stopwords")
except Exception as e:
    print("NLTK download error:", e)

# Load the model and TF-IDF vectorizer safely
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

def preprocess_text(text):
    """Preprocess input text by removing punctuation, converting to lowercase, and removing stopwords."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def detect(input_text):
    """Detect plagiarism in the given input text."""
    if not input_text.strip():
        return "No text provided. Please enter valid content."
    
    input_text = preprocess_text(input_text)
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism"

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text("text")
    except Exception as e:
        return f"Error reading PDF: {e}"
    return text.strip()

def extract_text_from_image(image_file):
    """Extract text from an image using Tesseract OCR."""
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
    except Exception as e:
        return f"Error processing image: {e}"
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    result = None
    input_text = ""
    
    if 'text' in request.form and request.form['text'].strip():
        input_text = request.form['text'].strip()
        result = detect(input_text)
    
    if 'pdf' in request.files and request.files['pdf'].filename:
        pdf_text = extract_text_from_pdf(request.files['pdf'])
        if pdf_text:
            result = detect(pdf_text)
    
    if 'image' in request.files and request.files['image'].filename:
        image_text = extract_text_from_image(request.files['image'])
        if image_text:
            result = detect(image_text)
    
    return render_template('index.html', result=result, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)