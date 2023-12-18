import streamlit as st
import torch
import pickle
from PIL import Image
import io

from model_execute import preprocess_images, output_to_names
from summarization import init_model_and_tokenizer, summarize
from wikipedia_api import getWikipedia

@st.cache_resource
def load_recognition_model():
    """
    Loads the translation model pipeline.
    """
    filename = "pickle_model.pkl"
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_summarizer():
    """
    Loads the summarization model.
    """

    summarizer, tokenizer = init_model_and_tokenizer()
    return summarizer, tokenizer

def predict_images(images, model):
    """
    Predicts each landmark name in `images` list.
    """
    images = preprocess_images(images)

    with torch.no_grad():
        output = model(images)
    
    names = output_to_names(output)

    return names

def load_images():
    """
    Loads user's images.
    """
    uploaded_files = st.file_uploader(
            label="Загрузите ваши фотографии.",
            type=['png', 'jpg'],
            accept_multiple_files=True
        )
    if uploaded_files is not None:
        images = []
        for file in uploaded_files:
            image_data = file.getvalue()
            st.image(image_data)
            images.append(image_data)
        
        return [Image.open(io.BytesIO(image_data)) for image_data in images]
    else:
        return None


# Load models
landmark_model = load_recognition_model()
summarizer, tokenizer = load_summarizer()

st.title("Распознавание достопримечательностей")

# Images input.
images = load_images()

result = st.button('Распознать')

if result:
    # Get predictions
    names = predict_images(images, landmark_model)
    st.write(names)

    # Request descriptions and coordinates from Wikipedia.
    wiki_data = getWikipedia(names)

    # Summarize descriptions for each landmark.
    for landmark in wiki_data:
        description = landmark['summary']
        summarized = summarize(description, summarizer, tokenizer)
        landmark['summarized'] = summarized

    st.write(wiki_data)

    # Draw a map.
