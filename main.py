import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
import xmltodict

st.set_page_config(page_title="Explorador de Precios de Gasolina", layout="wide")

st.title("🌟 Explorador de Precios de Gasolina en México")

@st.cache_data

def load_data():
    # Descarga de precios
    url_prices = 'https://publicacionexterna.azurewebsites.net/publicaciones/prices'
    r_prices = requests.get(url_prices)
    xml_prices = xmltodict.parse(r_prices.text.lstrip('ï»¿'))
    prices = xml_prices['places']['place']

    data_precios = []
    for p in prices:
        pid = p['@place_id']
        if isinstance(p['gas_price'], list):
            for g in p['gas_price']:
                data_precios.append({
                    'place_id': pid,
                    'gas_type': g['@type'],
                    'price': g['#text']
                })
        else:
            g = p['gas_price']
            data_precios.append({
                'place_id': pid,
                'gas_type': g['@type'],
                'price': g['#text']
            })

    base_precios = pd.DataFrame(data_precios)

    # Descarga de ubicaciones
    url_places = 'https://publicacionexterna.azurewebsites.net/publicaciones/places'
    r_places = requests.get(url_places)
    xml_places = xmltodict.parse(r_places.content.decode('utf-8-sig'))
    places = xml_places['places']['place']

    data_places = []
    for place in places:
        data_places.append({
            'place_id': place['@place_id'],
            'name': place['name'],
            'cre_id': place['cre_id'],
            'x': place['location']['x'],
            'y': place['location']['y']
        })

    base_id = pd.DataFrame(data_places)
    df = pd.merge(base_precios, base_id, on='place_id', how='inner')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df

df = load_data()

# Filtros laterales
gas_type = st.sidebar.selectbox("Selecciona el tipo de gasolina", df['gas_type'].unique())
price_min, price_max = st.sidebar.slider("Rango de precio", float(df['price'].min()), float(df['price'].max()), (float(df['price'].min()), float(df['price'].max())))

# Filtro de datos
df_filtered = df[(df['gas_type'] == gas_type) & (df['price'] >= price_min) & (df['price'] <= price_max)]

st.markdown(f"### Visualización de precios para: `{gas_type}`")
st.write(f"Observaciones filtradas: {len(df_filtered)}")

# Histogram
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df_filtered['price'], bins=30, kde=True, ax=ax1)
ax1.set_title(f"Histograma de precios para {gas_type}")
ax1.set_xlabel("Precio (MXN)")
ax1.set_ylabel("Frecuencia")
st.pyplot(fig1)

# CDF
fig2, ax2 = plt.subplots(figsize=(8, 4))
prices = np.sort(df_filtered['price'])
cdf = np.arange(1, len(prices) + 1) / len(prices)
ax2.plot(prices, cdf, marker='.', linestyle='none', color='blue')
ax2.set_title(f"CDF de precios para {gas_type}")
ax2.set_xlabel("Precio (MXN)")
ax2.set_ylabel("Distribución acumulada")
ax2.grid(True)
st.pyplot(fig2)

# KDE por tipo de gasolina (fijo)
st.markdown("### Comparación de distribuciones por tipo de gasolina")
fig3, ax3 = plt.subplots(figsize=(10, 5))
for gtype, color in zip(['regular', 'premium', 'diesel'], ['blue', 'red', 'green']):
    sns.kdeplot(df[df['gas_type'] == gtype]['price'], cumulative=True, label=gtype.capitalize(), ax=ax3, color=color)
ax3.set_xlim(18, 30)
ax3.set_xlabel("Precio (MXN)")
ax3.set_ylabel("Distribución acumulada")
ax3.set_title("CDF por tipo de gasolina")
ax3.legend()
st.pyplot(fig3)

# Tabla
st.markdown("### Vista tabular de datos filtrados")
st.dataframe(df_filtered.head(100))
