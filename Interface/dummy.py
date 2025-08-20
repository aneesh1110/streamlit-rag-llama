import os
import time
import fitz  # PyMuPDF
import pytesseract  # OCR for PNG images
from PIL import Image  # For opening PNG images
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken  # For token counting and context truncation
from rank_bm25 import BM25Okapi
import numpy as np
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env
HF_TOKEN = os.getenv("HF_TOKEN")  # retrieves your token from .env
PINECONE_TOKEN = os.getenv("PINECONE_TOKEN")  # retrieves your token from .env

# Disable tokenizers parallelism to avoid deadlock issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Pinecone and Sentence Transformer
pc = Pinecone(api_key=PINECONE_TOKEN)
index_name = "icici2-reports"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1'))
index = pc.Index(index_name)
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # Model's output vector dimension

# Use the Llama-2 model for chat-based interactions
huggingface_token = HF_TOKEN
model_name = "meta-llama/Llama-2-7b-chat-hf"

client = InferenceClient(model_name, token=huggingface_token)

# Initialize BM25 search for keyword-based search
documents = []  # To store documents after reading them
vectorizer = None
bm25 = None
MAX_TOKENS = 4096  # Max tokens for Llama-2
PROMPT_TOKENS = 500  # Tokens reserved for LLM response

# Initialize the Llama-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def truncate_context(context, max_tokens=MAX_TOKENS - PROMPT_TOKENS):
    # Use Llama-2 tokenizer for encoding and truncating
    input_tokens = tokenizer.encode(context)

    # Truncate the input to the last 'max_tokens' tokens
    truncated_tokens = input_tokens[-max_tokens:]

    # Decode the truncated tokens back into a string with explicit clean_up_tokenization_spaces
    return tokenizer.decode(truncated_tokens, clean_up_tokenization_spaces=True)


# Extract text and images from PDFs and PNGs
def extract_content(file_path, image_folder):
    text, images = "", []
    try:
        if file_path.endswith('.pdf'):
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                for img in page.get_images(full=True):
                    image_bytes = doc.extract_image(img[0])["image"]
                    img_path = f"{os.path.splitext(file_path)[0]}_p{page_num + 1}_img{img[0]}.png"
                    with open(os.path.join(image_folder, img_path), "wb") as img_file:
                        img_file.write(image_bytes)
                    images.append(img_path)
        elif file_path.endswith('.png'):
            text = pytesseract.image_to_string(Image.open(file_path))
            images.append(file_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        print(f"Images extracted so far: {images}")
    return text, images


# Read and process files from folder
def read_files(folder_path, image_folder):
    global documents
    documents, images = [], []
    for filename in os.listdir(folder_path):
        text, img = extract_content(os.path.join(folder_path, filename), image_folder)
        if text:
            documents.append(text)
        images.extend(img)
    print(f"Documents: {documents}")  # Debug: Print documents to ensure they are being read
    return documents, images


def index_documents(docs, batch_size=100):
    global vectorizer, bm25

    # Check if there are documents to index
    if not docs:
        print("No documents to index.")
        return

    # Indexing documents into Pinecone
    for batch in [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]:
        vectors = [(str(i), model.encode(doc).tolist(), {'text': doc}) for i, doc in enumerate(batch)]
        index.upsert(vectors=vectors)

    # Initialize the keyword-based search models (TF-IDF or BM25)
    if docs:
        vectorizer = TfidfVectorizer().fit_transform(docs)
        bm25 = BM25Okapi([doc.split(" ") for doc in docs])  # Tokenize each document for BM25
    print(f"BM25 initialized with {len(docs)} documents.")  # Debug: Print confirmation of BM25 initialization


# Perform hybrid search (semantic + keyword-based) and Llama 2-based QA
def document_based_qa(query, top_k=3):
    query_vector = model.encode(query).tolist()

    # Step 1: Perform Semantic Search
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)['matches']

    # Initialize an empty list for semantic texts
    semantic_texts = []

    # Safely extract texts from the semantic results, checking for metadata
    for res in results:
        # Check if 'metadata' exists and contains 'text'
        if 'metadata' in res and 'text' in res['metadata']:
            semantic_texts.append(res['metadata']['text'])
        else:
            print(f"Warning: Missing 'metadata' or 'text' in result: {res}")

    # Step 2: Perform Keyword-based Search (BM25 or TF-IDF)
    if bm25:
        bm25_scores = bm25.get_scores(query.split(" "))
        top_k_bm25_indices = np.argsort(bm25_scores)[-top_k:]  # Get top-k documents from BM25
        keyword_results = [documents[i] for i in top_k_bm25_indices]
    else:
        print("BM25 is not initialized. Make sure documents are indexed properly.")
        keyword_results = []

    # Step 3: Combine Semantic and Keyword-based Results
    hybrid_results = semantic_texts + keyword_results

    # Step 4: Use GPT-4 to answer based on the combined results
    for context in hybrid_results:
        if context:
            answer = get_llama2_answer(context, query)  # Directly call get_llama2_answer here
            if "Sorry" not in answer:
                return answer

    return "No sufficient context to answer the query."


# Get Llama-2 response using Hugging Face InferenceClient
# Get Llama-2 response using Hugging Face InferenceClient
def get_llama2_answer(context, query):
    try:
        # Ensure context fits within the Llama-2 token limit
        truncated_context = truncate_context(context)  # Use the function to truncate context
        prompt = truncated_context + '\n' + query

        # Send the request to Llama-2 for chat-based completion
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "Answer as truthfully as possible with the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  # Adjust max tokens as needed
            stream=False  # No streaming for now, get full response at once
        )

        # Access response using dataclass attributes instead of dict
        if response and response.choices:
            generated_text = response.choices[0].message.content  # Access using attributes
            return generated_text
        else:
            print("Error: Received an empty or invalid response from Llama-2.")
            return "Error: Received an empty or invalid response from Llama-2."

    except Exception as e:
        print(f"Error getting Llama-2 response: {e}")
        return "Error in Llama-2 response."


# Upload and index folder content
def upload_folder(folder_path):
    image_folder = os.path.join(folder_path, 'Images')
    os.makedirs(image_folder, exist_ok=True)
    docs, _ = read_files(folder_path, image_folder)
    index_documents(docs)
    return "Documents uploaded and indexed successfully."
