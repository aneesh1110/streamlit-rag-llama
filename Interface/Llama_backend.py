import json
import os
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import camelot
from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env
HF_TOKEN = os.getenv("HF_TOKEN")  # retrieves your token from .env
import asyncio

# Fix for asyncio event loop issues in Streamlit
nest_asyncio.apply()

# Hugging Face LLM and embedding models
llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    token=HF_TOKEN,
    temperature=0.3,  # Setting temperature for deterministic answers
    top_p=0.9,  # Setting top_p for controlled creativity
    max_new_tokens=256  # Limit the length of the response
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")

# Declare global query engine and index variable
query_engine = None
index = None


# Convert PDF to images
def pdf_to_images(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.alpha:  # Check if the image has an alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            images.append(img)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


# Function to vectorize images
def vectorize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Convert PDF pages to vectorized images
def pdf_to_vectorized_images(pdf_path):
    images = pdf_to_images(pdf_path)
    if not images:
        print("No images generated from the PDF.")
        return [], []

    vectorized_images = [vectorize_image(img) for img in images]
    return images, vectorized_images


# Process PDF: extract text, images, tables, and vectorized images
def process_pdf(file_path):
    try:
        texts = []
        images = []
        tables = []
        vectorized_images = []
        pdf_images = []

        # Convert PDF to vectorized images
        pdf_images, vectorized_images = pdf_to_vectorized_images(file_path)

        for page_num, (page_image, contours) in enumerate(zip(pdf_images, vectorized_images)):
            # Save the image of the page
            image_path = f"Documents/page_{page_num}.png"
            cv2.imwrite(image_path, page_image)
            images.append(image_path)

            # Perform OCR to extract text from the image
            text = pytesseract.image_to_string(page_image)
            texts.append(text)

            # Extract tables using Camelot (handles case where no tables are found)
            try:
                table_list = camelot.read_pdf(file_path, pages=str(page_num + 1))
                if table_list:
                    for table in table_list:
                        tables.append(table)
            except Exception as e:
                print(f"No tables found on page {page_num + 1}: {e}")

            # Save the vectorized contours as an image
            contour_image = page_image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            contour_image_path = f"Documents/page_{page_num}_contours.png"
            cv2.imwrite(contour_image_path, contour_image)
            images.append(contour_image_path)  # Save the path to the contour image

        return texts, images, tables, vectorized_images, pdf_images
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return [], [], [], [], []


# Enhanced folder upload function
def upload_folder(folder_path,
                  output_dir='/Users/aneeshvinod/PycharmProjects/StreamLit_LLAMA/LLAMA_Streamlit/Interface/Documents/Docs'):
    global query_engine, index

    if index is not None:
        # Skip re-indexing if already done
        return "Documents are already indexed."

    try:
        # Create a directory to save the processed documents if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                texts, images, tables, vectorized_images, pdf_images = process_pdf(pdf_path)

                # Save texts
                with open(os.path.join(output_dir, f"{filename}_texts.txt"), 'w') as f:
                    f.write("\n".join(texts))

                # Save images
                for i, image_path in enumerate(images):
                    img = cv2.imread(image_path)
                    if img is not None:
                        cv2.imwrite(os.path.join(output_dir, f"{filename}_image_{i}.png"), img)

                # Save tables
                for i, table in enumerate(tables):
                    if not table.df.empty:
                        table.df.to_csv(os.path.join(output_dir, f"{filename}_table_{i}.csv"), index=False)

                # Save vectorized images
                for i, contours in enumerate(vectorized_images):
                    img = pdf_images[i].copy()
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
                    vector_image_path = os.path.join(output_dir, f"{filename}_vectorized_image_{i}.png")
                    cv2.imwrite(vector_image_path, img)

        # Reload documents and build index
        documents = SimpleDirectoryReader(output_dir).load_data()
        if not documents:
            raise ValueError("No documents loaded for indexing.")

        # Rebuild the index and create query engine
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, chunk_size=256)
        query_engine = index.as_query_engine(llm=llm)

        if query_engine is None:
            raise ValueError("Query engine initialization failed.")

        return "Folder uploaded and indexed successfully."
    except Exception as e:
        return f"An error occurred: {e}"


# Query the model using the query engine with a custom system prompt
def query_model(query_text):
    global query_engine  # Ensure it's accessible globally
    if query_engine is None:
        return "Query engine is not initialized. Please upload a folder first."
    try:
        # Custom system prompt
        system_prompt = "You are a financial assistant bot. Answer questions truthfully according to the provided documents and avoid answering anything unnecessary."

        # Include the system prompt and query in the message
        formatted_query = f"{system_prompt}\n{query_text}"

        # Query the model with the formatted query
        response = query_engine.query(formatted_query)
        return str(response)
    except Exception as e:
        return f"An error occurred: {e}"
