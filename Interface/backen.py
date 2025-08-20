# backend.py
import json
import os
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import camelot
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import (ServiceContext, SimpleDirectoryReader, VectorStoreIndex, set_global_service_context)
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.llms.clarifai import Clarifai
import nest_asyncio
from clarifai.client.model import Model
import asyncio

# Fix for asyncio event loop issues in Streamlit
nest_asyncio.apply()

# llm_model = Clarifai(
#     model_url=params["model_url"]
# )

# # Define Gradient's Model Adapter for LLAMA-2
# llm = GradientBaseModelLLM(
#     access_token='13ojf7kBETGmTqcqH6eMcEBJeaPJp1HZ',
#     workspace_id='245d0f63-4663-42f6-8e52-60fc95967702_workspace',
#     base_model_slug="llama2-7b-chat",
#     max_tokens=400
# )

# # Configure Gradient embeddings
# embed_model = GradientEmbedding(
#     gradient_access_token="13ojf7kBETGmTqcqH6eMcEBJeaPJp1HZ",
#     gradient_workspace_id="245d0f63-4663-42f6-8e52-60fc95967702_workspace",
#     gradient_model_slug="bge-large",
# )

# Set up the service context
# service_context = ServiceContext.from_defaults(
#     llm=llm,
#     embed_model=embed_model,
#     chunk_size=256,
# )
#
# set_global_service_context(service_context)


# Convert PDF to images
def pdf_to_images(pdf_path):
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


# Function to vectorize images
def vectorize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Main function to convert PDF to vectorized images
def pdf_to_vectorized_images(pdf_path):
    images = pdf_to_images(pdf_path)
    vectorized_images = [vectorize_image(img) for img in images]
    return images, vectorized_images


# Function to process PDF
def process_pdf(file_path):
    texts = []
    images = []
    tables = []
    vectorized_images = []

    # Convert PDF to vectorized images
    pdf_images, vectorized_images = pdf_to_vectorized_images(file_path)

    for page_num, page_image in enumerate(pdf_images):
        # Save the image of the page
        image_path = f"Documents/page_{page_num}.png"
        cv2.imwrite(image_path, page_image)
        images.append(image_path)

        # Perform OCR to extract text from the image
        text = pytesseract.image_to_string(page_image)
        texts.append(text)

        # Extract tables using camelot
        tables.extend(camelot.read_pdf(file_path, pages=str(page_num + 1)))

    return texts, images, tables, vectorized_images, pdf_images


# Enhanced upload function to handle a folder
def upload_folder(folder_path,
                  output_dir='/Users/aneeshvinod/PycharmProjects/StreamLit_LLAMA/LLAMA_Streamlit/Interface/Documents'):
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
                    cv2.imwrite(os.path.join(output_dir, f"{filename}_image_{i}.png"), img)

                # Save tables
                for i, table in enumerate(tables):
                    table.df.to_csv(os.path.join(output_dir, f"{filename}_table_{i}.csv"), index=False)

                # Save vectorized images
                for i, contours in enumerate(vectorized_images):
                    img = pdf_images[i].copy()
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
                    vector_image_path = os.path.join(output_dir, f"{filename}_vectorized_image_{i}.png")
                    cv2.imwrite(vector_image_path, img)

        # Reload documents and rebuild index
        documents = SimpleDirectoryReader(output_dir).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        global query_engine
        query_engine = index.as_query_engine()

        return "Folder uploaded and indexed successfully."
    except Exception as e:
        return f"An error occurred: {e}"


# Function to query the model
def query_model(query_text):
    try:
        prompt = query_text
        # You can set the model using model URL or model ID.
        model_url = "https://clarifai.com/meta/Llama-2/models/llama2-7b-chat"

        # Model Predict
        model_prediction = Model(url=model_url, pat="ebdf8921632e4553ba7b9e8e97fc8d9f").predict_by_bytes(
            prompt.encode(), input_type="text")
        response = model_prediction.outputs[0].data.text.raw
        return str(response)
    except Exception as e:
        return f"An error occurred: {e}"


# def query_model(query_text):
#     try:
#         response = query_engine.query(query_text)
#         return str(response)
#     except Exception as e:
#         return f"An error occurred: {e}"
