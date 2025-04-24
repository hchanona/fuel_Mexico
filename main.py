# === CONFIGURACIÓN INICIAL ===
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(page_title="Product Market Regulator Sandbox", layout="wide")
st.title("PMR Sandbox")

@st.cache_data
def load_data():
    df = pd.read_excel("PMR_with_GDP.xlsx")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

df = load_data()
st.write(f"🌍 This dataset includes **{df['Country'].nunique()} countries**.")

medium_level_indicators = [
    "Distortions Induced by Public Ownership",
    "Involvement in Business Operations",
    "Regulations Impact Evaluation",
    "Administrative and Regulatory Burden",
    "Barriers in Service & Network sectors",
    "Barriers to Trade and Investment"
]

low_level_indicators = [col for col in df.columns if col not in ["Country", "OECD", "GDP_PCAP_2023", "PMR_2023"] + medium_level_indicators]

st.sidebar.header("Options")
mode = st.sidebar.radio("What do you want to do?", ["Guided simulation", "Autonomous simulation", "Stats"])
countries = df["Country"].tolist()
selected_country = st.sidebar.selectbox("Select a country", countries, index=countries.index("Australia") if "Australia" in countries else 0)

# === MODO: SIMULACIÓN GUIADA ===
if mode == "Guided simulation":
    row = df[df["Country"] == selected_country].iloc[0]
    pmr_score = row["PMR_2023"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
    with col2:
        oecd_avg = df[df['OECD'] == 1]['PMR_2023'].mean()
        non_oecd_avg = df[df['OECD'] == 0]['PMR_2023'].mean()
        st.metric(label='OECD Average PMR', value=round(oecd_avg, 3))
        st.metric(label='Non-OECD Average PMR', value=round(non_oecd_avg, 3))

    st.subheader("📊 PMR Profile: Country vs OECD Average (Medium-level indicators)")
    oecd_avg_vals = df[df["OECD"] == 1][medium_level_indicators].mean()
    country_vals = row[medium_level_indicators]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=country_vals.values, theta=medium_level_indicators, fill='toself', name=selected_country, line=dict(color='blue')))
    radar_fig.add_trace(go.Scatterpolar(r=oecd_avg_vals.values, theta=medium_level_indicators, fill='toself', name='OECD Average', line=dict(color='gray')))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,6])), showlegend=True)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("🔎 Regulatory Subcomponent Overview – Current Position by Rank")
    ranks = {ind: df[ind].rank(method="min").astype(int) for ind in low_level_indicators}
    rank_df = pd.DataFrame(ranks)
    summary = []
    for ind in low_level_indicators:
        score = row[ind]
        rank = int(rank_df[df["Country"] == selected_country][ind])
        summary.append({
            "Indicator": ind,
            "Score": round(score, 2),
            "Rank": rank
        })

    df_summary = pd.DataFrame(summary).sort_values("Rank")
    st.dataframe(df_summary.reset_index(drop=True), use_container_width=True)

    st.subheader("📌 Suggested Reform Priorities")
    top3 = df_summary.tail(3)["Indicator"].tolist()

    sliders = {}
    for ind in top3:
        current = row[ind]
        rank = int(rank_df[df["Country"] == selected_country][ind])
        st.markdown(f"**{ind}**\n\nCurrent score: {round(current,2)} | Rank: {rank}")
        sliders[ind] = st.slider(ind, 0.0, 6.0, float(current), 0.1)

    simulated_row = row.copy()
    for ind, val in sliders.items():
        simulated_row[ind] = val

    new_medium_avg = simulated_row[medium_level_indicators].mean()
    original_medium = row[medium_level_indicators].mean()

    df_simulated = df.copy()
    df_simulated.loc[df_simulated["Country"] == selected_country, medium_level_indicators] = simulated_row[medium_level_indicators]
    df_simulated["PMR_simulated"] = df_simulated[medium_level_indicators].mean(axis=1)
    valid_simulated = df_simulated[df_simulated["PMR_simulated"].notna()].copy()
    valid_simulated["rank_simulated"] = valid_simulated["PMR_simulated"].rank(method="min")


    sliders = {}
    for ind in top3:
        current = row[ind]
        rank = int(rank_df[df["Country"] == selected_country][ind])
        st.markdown(f"**{ind}**\n\nCurrent score: {round(current,2)} | Rank: {rank}")
        sliders[ind] = st.slider(ind, 0.0, 6.0, float(current), 0.1)

    simulated_row = row.copy()
    for ind, val in sliders.items():
        simulated_row[ind] = val

# 🔧 Actualiza indicadores de nivel medio
    for medium in medium_level_indicators:
        subcomponents = [col for col in low_level_indicators if col in df.columns and (col in medium or medium in col)
    ]
        if subcomponents:
            simulated_row[medium] = simulated_row[subcomponents].mean()

# ✅ Ahora sí calcula el nuevo PMR correctamente
    new_medium_avg = simulated_row[medium_level_indicators].mean()
    original_medium = row[medium_level_indicators].mean()

    df_simulated = df.copy()
    df_simulated.loc[df_simulated["Country"] == selected_country, medium_level_indicators] = simulated_row[medium_level_indicators]
    df_simulated["PMR_simulated"] = df_simulated[medium_level_indicators].mean(axis=1)
    valid_simulated = df_simulated[df_simulated["PMR_simulated"].notna()].copy()
    valid_simulated["rank_simulated"] = valid_simulated["PMR_simulated"].rank(method="min")

    # Asegurar que siempre se define new_rank
    new_rank = None
    if selected_country in valid_simulated["Country"].values:
        new_rank = int(valid_simulated.loc[valid_simulated["Country"] == selected_country, "rank_simulated"].values[0])
        st.metric("Simulated Rank", f"{new_rank}")
    else:
        st.warning("⚠️ Could not compute rank: missing or invalid simulated data for this country.")

    st.write("---")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Original PMR Estimate", round(original_medium, 3))
    with col5:
        st.metric("Simulated PMR Estimate", round(new_medium_avg, 3), delta=round(new_medium_avg - original_medium, 3))
    with col6:
        if new_rank is not None:
            st.metric("Simulated Rank", f"{int(new_rank)}")
        else:
            st.metric("Simulated Rank", "N/A")
elif mode == "Autonomous simulation":
    st.subheader("🧭 Autonomous Simulation – Choose Reform Areas Hierarchically")

    # Mensaje introductorio
    st.markdown("Selecciona hasta **tres indicadores de nivel medio** para simular reformas:")

    # Multiselect para elegir hasta 3 indicadores de nivel medio
    selected_rubros = st.multiselect(
        "Indicadores de nivel medio",
        options=medium_level_indicators,
        default=[],
        max_selections=3,
        help="Selecciona los rubros que deseas reformar"
    )

    if selected_rubros:
        # Se obtiene la fila del país seleccionado
        simulated_row = df[df["Country"] == selected_country].iloc[0].copy()

        st.write("### 🎯 Ajusta los subcomponentes de cada rubro seleccionado:")

        for rubro in selected_rubros:
            st.markdown(f"**{rubro}**")
            # Encuentra subcomponentes por coincidencia de nombre
            subcomponents = [
                col for col in low_level_indicators
                if col in df.columns and col in simulated_row.index and (col in rubro or rubro in col)
            ]

            if not subcomponents:
                st.warning("No se encontraron subcomponentes directamente ligados. Se usó coincidencia de texto.")

            for sub in subcomponents:
                current_val = simulated_row[sub]
                new_val = st.slider(f"{sub}", 0.0, 6.0, float(current_val), 0.1)
                simulated_row[sub] = new_val

        # Recalcular cada indicador medio a partir del promedio de sus subcomponentes
        for rubro in medium_level_indicators:
            sublist = [
                col for col in low_level_indicators
                if col in df.columns and (col in rubro or rubro in col)
            ]
            if sublist:
                simulated_row[rubro] = simulated_row[sublist].mean()

        # Recalcular el PMR simulado como promedio de los indicadores medios
        new_pmr = simulated_row[medium_level_indicators].mean()
        original_pmr = df[df["Country"] == selected_country][medium_level_indicators].mean(axis=1).values[0]

        # Se calcula un nuevo PMR para todos los países para poder estimar percentiles
        df["PMR_simulated"] = df[medium_level_indicators].mean(axis=1)
        new_percentile = (df["PMR_simulated"] > new_pmr).mean() * 100

        # Muestra los resultados finales
        st.write("---")
        col7, col8, col9 = st.columns(3)
        with col7:
            st.metric("Original PMR", round(original_pmr, 3))
        with col8:
            st.metric("Simulated PMR", round(new_pmr, 3), delta=round(new_pmr - original_pmr, 3))
        with col9:
            st.metric("Simulated Percentile", f"{round(new_percentile)}%")
    else:
        st.info("Selecciona al menos un rubro de nivel medio para comenzar.")
elif mode == "Stats":
    st.header("📈 PMR Trends")

    # Subtítulo y explicación del análisis
    st.subheader("🔎 PMR Score vs. GDP per capita (log-log) & OECD Membership")
    st.write("""
    Este análisis examina cómo el **ingreso per cápita (logarítmico)** y la **pertenencia a la OCDE** afectan el **logaritmo del PMR**. 
    Los coeficientes se interpretan como **elasticidades** o diferencias porcentuales aproximadas.
    """)

    # Filtrar solo países con datos válidos para logaritmos
    df_log = df[(df["PMR_2023"] > 0) & (df["GDP_PCAP_2023"] > 0)].copy()
    
    # Crear columnas transformadas en logaritmos
    df_log["log_pmr"] = np.log(df_log["PMR_2023"])
    df_log["log_gdp"] = np.log(df_log["GDP_PCAP_2023"])

    # Crear las variables independientes (X) y dependiente (y)
    X = sm.add_constant(df_log[["log_gdp", "OECD"]])  # constante + log(PIB) + dummy OCDE
    y = df_log["log_pmr"]  # log(PMR)

    # Ajustar modelo de regresión lineal
    model = sm.OLS(y, X).fit()

    # Mostrar resultados de la regresión como texto plano
    st.text("OLS Regression Results (log-log)")
    st.text(model.summary())

    # Gráfico de dispersión con línea de regresión
    st.subheader("📊 Distribución log(PMR) vs log(ingreso per cápita)")

    # Crear puntos para predicción de la curva
    x_vals = np.linspace(df_log["log_gdp"].min(), df_log["log_gdp"].max(), 100)
    X_pred = pd.DataFrame({
        "const": 1.0,
        "log_gdp": x_vals,
        "OECD": df_log["OECD"].mean()  # mantener constante la OCDE dummy
    })
    y_vals = model.predict(X_pred)

    # Crear gráfico de dispersión con línea de regresión
    fig = px.scatter(
        df_log,
        x="log_gdp",
        y="log_pmr",
        text="Country",
        labels={"log_gdp": "log(Income per capita)", "log_pmr": "log(PMR Score)"},
        title="log(PMR) vs log(Income per capita)"
    )

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Regresión lineal log-log',
        line=dict(color='red')
    ))

    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)

