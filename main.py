import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import folium
from folium import Map, CircleMarker, Tooltip
from sklearn.cluster import DBSCAN
import requests
import xmltodict

# ----------------------
# CARGAR DATOS DE LA API DE LA CRE
# ----------------------
@st.cache_data
def cargar_datos_cre():
    # Precios
    url_precios = "https://publicacionexterna.azurewebsites.net/publicaciones/prices"
    res1 = requests.get(url_precios)
    data1 = xmltodict.parse(res1.content.decode('utf-8-sig'))
    precios = []
    for lugar in data1['places']['place']:
        if isinstance(lugar['gas_price'], list):
            for p in lugar['gas_price']:
                precios.append({'place_id': lugar['@place_id'], 'gas_type': p['@type'], 'price': p['#text']})
        else:
            p = lugar['gas_price']
            precios.append({'place_id': lugar['@place_id'], 'gas_type': p['@type'], 'price': p['#text']})
    df_precios = pd.DataFrame(precios)

    # Ubicaciones
    url_ubicaciones = "https://publicacionexterna.azurewebsites.net/publicaciones/places"
    res2 = requests.get(url_ubicaciones)
    data2 = xmltodict.parse(res2.content.decode('utf-8-sig'))
    lugares = []
    for lugar in data2['places']['place']:
        lugares.append({
            'place_id': lugar['@place_id'],
            'name': lugar['name'],
            'cre_id': lugar['cre_id'],
            'x': lugar['location']['x'],
            'y': lugar['location']['y']
        })
    df_lugares = pd.DataFrame(lugares)

    # Merge
    df = pd.merge(df_precios, df_lugares, on='place_id', how='inner')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['price', 'x', 'y'])
    return df

# ----------------------
# FUNCIONES DE CLUSTER
# ----------------------
def generar_mapa_clusters(df, tipo_gasolina='regular', price_range_max=0.02, eps_km=3):
    df = df[df['gas_type'] == tipo_gasolina].copy()
    df = df.dropna(subset=['x', 'y', 'price'])

    coords = np.radians(df[['y', 'x']].values)
    kms_per_radian = 6371.0088
    eps = eps_km / kms_per_radian
    db = DBSCAN(eps=eps, min_samples=3, algorithm='ball_tree', metric='haversine').fit(coords)
    df['cluster_geo'] = db.labels_

    df_valid = df[df['cluster_geo'] != -1]
    agg = df_valid.groupby('cluster_geo').agg(
        count=('price', 'count'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        num_names=('name', pd.Series.nunique)
    ).reset_index()
    agg['price_range'] = agg['max_price'] - agg['min_price']

    sospechosos = agg[(agg['count'] >= 3) & (agg['price_range'] <= price_range_max) & (agg['num_names'] >= 2)]
    df_sospechosos = df[df['cluster_geo'].isin(sospechosos['cluster_geo'])].copy()

    m = Map(location=[df_sospechosos['y'].mean(), df_sospechosos['x'].mean()], zoom_start=6, tiles="CartoDB positron")
    cluster_ids = df_sospechosos['cluster_geo'].unique()
    color_palette = matplotlib.cm.get_cmap('tab20', len(cluster_ids))
    cluster_colors = {cid: matplotlib.colors.rgb2hex(color_palette(i)[:3]) for i, cid in enumerate(cluster_ids)}

    for _, row in df_sospechosos.iterrows():
        CircleMarker(
            location=[row['y'], row['x']],
            radius=4,
            color=cluster_colors.get(row['cluster_geo'], "#000000"),
            fill=True,
            fill_opacity=0.8,
            tooltip=Tooltip(f"<b>{row['name']}</b><br>Precio: ${row['price']}<br>Cluster: {row['cluster_geo']}")
        ).add_to(m)

    return m, sospechosos

# ----------------------
# INTERFAZ STREAMLIT
# ----------------------
st.set_page_config(page_title="Clusters de gasolina", layout="wide")
st.title("🌟 Clusters de estaciones de gasolina")
st.markdown("Explora agrupaciones de estaciones con precios similares y proximidad geográfica, usando datos reales de la CRE.")

df_base = cargar_datos_cre()

tipo = st.selectbox("Selecciona el tipo de gasolina", ['regular', 'premium', 'diesel'])
price_limit = st.slider("Diferencia máxima de precios dentro del cluster (MXN)", 0.00, 0.10, 0.02, step=0.01)
radius_km = st.slider("Radio máximo entre estaciones para formar cluster (km)", 1, 10, 3)

if st.button("🌿 Generar mapa"):
    mapa, resumen = generar_mapa_clusters(df_base, tipo_gasolina=tipo, price_range_max=price_limit, eps_km=radius_km)
    st.markdown(f"### Se detectaron **{len(resumen)}** clusters.")
    st.dataframe(resumen)
    st.components.v1.html(mapa._repr_html_(), height=700, scrolling=True)

