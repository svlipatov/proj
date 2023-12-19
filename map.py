import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Загрузка данных
data = pd.DataFrame({
    'latitude': [37.7749, 34.0522],  # Пример координат для точек
    'longitude': [-122.4194, -118.2437],
    'name': ['San Francisco', 'Los Angeles']
})

# Заголовок приложения
st.title('Приложение с картой MapBox')

# Создание маркеров
marker_size = 12  # Размер маркера

fig = go.Figure()

for i, row in data.iterrows():
    fig.add_trace(go.Scattermapbox(
        lat=[row['latitude']],
        lon=[row['longitude']],
        mode='markers',
        marker=dict(
            size=marker_size,
            color='red',
            sizemode='diameter',  # Размер маркера в диаметрах
        ),
        text=row['name']
    ))

# Вычисление границы области, содержащей все маркеры
min_lat, max_lat = data['latitude'].min(), data['latitude'].max()
min_lon, max_lon = data['longitude'].min(), data['longitude'].max()

# Рассчитываем центр и масштаб для отображения всех маркеров
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2
zoom_level = 1  # Масштаб (может потребоваться настройка)

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
