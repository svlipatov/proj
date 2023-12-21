import streamlit as st
import torch
import pickle
from PIL import Image
import io

from model_execute import preprocess_images, output_to_names
from summarization import init_model_and_tokenizer, summarize
from wikipedia_api import getWikipedia
from mapbox_map import plot_map

@st.cache_resource
def load_recognition_model():
    """
    Loads the translation model pipeline.
    """
    filename = "model/pickle_model.pkl"
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
        cols_list = []
        for file in uploaded_files:
            image_data = file.getvalue()
            images.append(image_data)
            container = st.container(border=True)
            cols = container.columns([1, 3])
            cols[0].image(image_data, width=300)
            cols_list.append(cols[1])
        
        return [Image.open(io.BytesIO(image_data)) for image_data in images], cols_list
    else:
        return None


st.set_page_config(layout="wide")
# Load models
landmark_model = load_recognition_model()
summarizer, tokenizer = load_summarizer()

st.title("Распознавание достопримечательностей")

# Images input.
images, cols_list = load_images()

summarize_checkbox = st.checkbox("Короткое описание")
result = st.button('Распознать')

if images and result:
    # Get predictions
    names = predict_images(images, landmark_model)

    # Request descriptions and coordinates from Wikipedia.
    wiki_data = getWikipedia(names)

    # Summarize descriptions for each landmark.
    if summarize_checkbox:
        for landmark in wiki_data:
            description = landmark['summary']
            summarized = summarize(description, summarizer, tokenizer)
            landmark['summarized'] = summarized
    for posts, cols in zip(wiki_data, cols_list):
        cols.markdown('**' + posts['find'] + '**')
        if summarize_checkbox:
            cols.markdown(posts['summarized'])
        else:
            cols.markdown(posts['summary'])

    # Draw a map.
    with st.container():
        plot_map(wiki_data)
