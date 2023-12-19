import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def preprocess_df(df):
    """
    Убрать строки без координат, преобразовать координаты в float.
    """
    no_coords = df['latitude'] == ''
    df = df.drop(df[no_coords].index)
    df['latitude'] = df['latitude'].astype('float')
    df['longitude'] = df['longitude'].astype('float')
    return df

def plot_map(landmarks):
    # Загрузка данных
    data = pd.DataFrame(landmarks)
    data = preprocess_df(data)
    st.write(data)
    # Создание маркеров
    marker_size = 12  # Размер маркера

    fig = go.Figure()

    for i, row in data.iterrows():
        if type(row['latitude']) is not float:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=[row['latitude']],
            lon=[row['longitude']],
            mode='markers',
            marker=dict(
                size=marker_size,
                color='red',
                sizemode='diameter',  # Размер маркера в диаметрах
            ),
            text=row['find']
        ))

    # Вычисление границы области, содержащей все маркеры
    min_lat, max_lat = data['latitude'].min(), data['latitude'].max()
    min_lon, max_lon = data['longitude'].min(), data['longitude'].max()

    # Рассчитываем центр и масштаб для отображения всех маркеров
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    zoom_level = 12  # Масштаб (может потребоваться настройка)

    # Настройка карты
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level,
        ),
    )

    # Отображение карты в Streamlit
    st.plotly_chart(fig)
