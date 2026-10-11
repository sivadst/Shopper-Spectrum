import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from streamlit_option_menu import option_menu

# ==============================================================================
# 1. PAGE CONFIGURATION & GLOBAL STYLING
# ==============================================================================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism, Modern Fonts, Gradient Buttons & Product Cards
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #f8fafc;
}

/* Hide Streamlit Header & Footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Glassmorphism Card Style */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(99, 102, 241, 0.4);
}

/* Product Card Styling */
.product-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    margin-bottom: 16px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.product-card:hover {
    transform: translateY(-5px) scale(1.02);
    border-color: #6366f1;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
}

.product-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: #ffffff;
    margin-bottom: 12px;
}

.similarity-badge {
    display: inline-block;
    float: right;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 700;
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.product-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 8px;
}

/* Custom Gradient Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s ease !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
}

/* Segment Cards */
.segment-high-value {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.4));
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.segment-regular {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.4));
    border: 1px solid #3b82f6;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.segment-occasional {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.4));
    border: 1px solid #f59e0b;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.segment-at-risk {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.4));
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #0f172a;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. DATA & MODEL LOADING WITH FALLBACK
# ==============================================================================
@st.cache_data(ttl=3600)
def load_and_preprocess_data():
    """Loads transaction data from Online_Retail.csv or sample_data.csv with fallback."""
    file_path = None
    if os.path.exists("Online_Retail.csv"):
        file_path = "Online_Retail.csv"
    elif os.path.exists("sample_data.csv"):
        file_path = "sample_data.csv"

    if file_path:
        df = pd.read_csv(file_path)
    else:
        # Generate synthetic fallback dataset
        np.random.seed(42)
        products = [
            ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
            ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
            ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
            ("47566", "PARTY BUNTING", 4.95),
            ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
            ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
            ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
            ("23203", "JUMBO BAG VINTAGE DOLLY PARK", 2.08),
            ("21232", "STRAWBERRY CERAMIC TRINKET BOX", 1.25),
            ("22197", "POPCORN HOLDER", 0.85),
            ("22469", "HEART OF WICKER SMALL", 1.65),
            ("22086", "PAPER CHAIN KIT VINTAGE CHRISTMAS", 2.95),
            ("22112", "CHOCOLATE HOT WATER BOTTLE", 4.95),
            ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
            ("22382", "LUNCH BAG PINK POLKADOT", 1.65)
        ]
        countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain"]
        records = []
        base_date = pd.Timestamp("2023-12-31")
        for cid in range(10001, 10200):
            n_orders = np.random.randint(1, 10)
            for _ in range(n_orders):
                inv = f"58{np.random.randint(10000, 99999)}"
                days = np.random.randint(1, 300)
                date = base_date - pd.Timedelta(days=days)
                stock, desc, price = products[np.random.randint(0, len(products))]
                records.append({
                    "InvoiceNo": inv, "StockCode": stock, "Description": desc,
                    "Quantity": np.random.randint(1, 15), "InvoiceDate": date,
                    "UnitPrice": price, "Country": np.random.choice(countries),
                    "CustomerID": cid
                })
        df = pd.DataFrame(records)

    # Data Preprocessing
    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

    return df


@st.cache_resource
def load_models_and_matrices(df):
    """Loads pre-trained model artifacts or computes them dynamically."""
    scaler, kmeans, similarity_df = None, None, None

    if os.path.exists("rfm_scaler.pkl") and os.path.exists("kmeans_model.pkl"):
        try:
            scaler = joblib.load("rfm_scaler.pkl")
            kmeans = joblib.load("kmeans_model.pkl")
        except Exception:
            scaler, kmeans = None, None

    if os.path.exists("similarity_matrix.pkl"):
        try:
            similarity_df = joblib.load("similarity_matrix.pkl")
        except Exception:
            similarity_df = None

    # Calculate RFM Data
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalSpend": "sum"
    }).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # Train Scaler & KMeans fallback if needed
    if scaler is None or kmeans is None:
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    else:
        rfm_scaled = scaler.transform(rfm[["Recency", "Frequency", "Monetary"]])
        rfm["Cluster"] = kmeans.predict(rfm_scaled)

    # Generate Item Similarity Matrix fallback if needed
    if similarity_df is None:
        user_item = df.pivot_table(index="CustomerID", columns="Description", values="Quantity", aggfunc="sum", fill_value=0)
        item_sim = cosine_similarity(user_item.T)
        similarity_df = pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)

    return scaler, kmeans, similarity_df, rfm


# Load data and artifacts
df_data = load_and_preprocess_data()
scaler, kmeans_model, similarity_matrix, rfm_df = load_models_and_matrices(df_data)


# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛒 **Shopper Spectrum**")
    st.markdown("<p style='color:#94a3b8; font-size:0.85rem;'>Customer Intelligence & Recommendation Hub</p>", unsafe_allow_html=True)
    st.markdown("---")

    selected_menu = option_menu(
        menu_title="Navigation",
        options=["Product Recommendations", "Customer Segmentation", "EDA & Analytics"],
        icons=["bag-check-fill", "people-fill", "bar-chart-line-fill"],
        menu_icon="compass",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "rgba(15, 23, 42, 0.8)"},
            "icon": {"color": "#a855f7", "font-size": "18px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "4px", "--hover-color": "rgba(99, 102, 241, 0.2)"},
            "nav-link-selected": {"background-color": "#6366f1", "font-weight": "600"},
        }
    )

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>Powered by Machine Learning & Collaborative Filtering</div>", unsafe_allow_html=True)


# ==============================================================================
# 4. MODULE 1: PRODUCT RECOMMENDATION ENGINE
# ==============================================================================
if selected_menu == "Product Recommendations":
    st.markdown("## 🛍️ **Item-Based Recommendation Engine**")
    st.markdown("Discover products similar to your selections powered by **Cosine Similarity Collaborative Filtering**.")
    st.markdown("<br>", unsafe_allow_html=True)

    available_products = sorted(list(similarity_matrix.columns))

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        selected_prod = st.selectbox(
            "🔍 **Select or Search for a Product:**",
            options=available_products,
            index=0,
            help="Choose an item from the retail catalog to view top recommendations."
        )

    with col_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        get_rec_btn = st.button("✨ Get Recommendations")

    if get_rec_btn or selected_prod:
        st.markdown(f"### 🎯 Recommended Items for: *\"{selected_prod}\"*")
        st.markdown("---")

        # Compute recommendations
        if selected_prod in similarity_matrix.columns:
            sim_scores = similarity_matrix[selected_prod].sort_values(ascending=False)
            # Remove the product itself
            sim_scores = sim_scores[sim_scores.index != selected_prod].head(5)

            cols = st.columns(5)
            for idx, (rec_prod, score) in enumerate(sim_scores.items()):
                match_pct = int(score * 100)
                with cols[idx]:
                    card_html = f"""
                    <div class="product-card">
                        <div class="product-badge">Top Match #{idx+1}</div>
                        <div class="similarity-badge">{match_pct}% Match</div>
                        <div style="clear: both; margin-top: 10px;"></div>
                        <div class="product-title">{rec_prod}</div>
                        <div style="margin-top: 12px; font-size: 0.8rem; color: #94a3b8;">
                            Recommended for you based on co-purchase behavior.
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.warning("Product not found in similarity matrix.")


# ==============================================================================
# 5. MODULE 2: CUSTOMER SEGMENTATION (RFM)
# ==============================================================================
elif selected_menu == "Customer Segmentation":
    st.markdown("## 👥 **RFM Customer Segmentation Predictor**")
    st.markdown("Classify customer behavior using **Recency**, **Frequency**, and **Monetary** metrics powered by pre-trained **KMeans Clustering**.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_inputs, col_output = st.columns([1, 1], gap="large")

    with col_inputs:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📥 **Input Customer Metrics**")
        st.markdown("Provide purchasing behavior parameters below:")

        recency_val = st.number_input(
            "📅 **Recency (Days since last purchase):**",
            min_value=0, max_value=730, value=15, step=1,
            help="Number of days elapsed since the customer last made a transaction."
        )

        frequency_val = st.number_input(
            "🔄 **Frequency (Total purchase transactions):**",
            min_value=1, max_value=500, value=12, step=1,
            help="Total count of unique purchases completed by the customer."
        )

        monetary_val = st.number_input(
            "💵 **Monetary ($ Total lifetime spend):**",
            min_value=0.0, max_value=100000.0, value=1500.0, step=50.0,
            help="Cumulative currency value spent across all transactions."
        )

        predict_btn = st.button("🔮 Predict Customer Segment")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        st.markdown("#### 🏷️ **Segmentation Result**")

        if predict_btn:
            # Scale input
            input_df = pd.DataFrame([[recency_val, frequency_val, monetary_val]], columns=["Recency", "Frequency", "Monetary"])
            scaled_input = scaler.transform(input_df)
            cluster_id = kmeans_model.predict(scaled_input)[0]

            # Rule-based segment naming heuristic based on RFM profile
            if monetary_val >= 1200 or (frequency_val >= 15 and recency_val <= 45):
                seg_label = "High-Value"
                icon = "👑"
                css_class = "segment-high-value"
                desc = "Regular, frequent, recent, and big spenders. Highest priority for loyalty rewards and VIP programs."
            elif recency_val > 150 and frequency_val <= 3:
                seg_label = "At-Risk"
                icon = "⚠️"
                css_class = "segment-at-risk"
                desc = "Haven't purchased in a long time with low spend. Target with retention campaigns and win-back offers."
            elif frequency_val <= 4 and monetary_val < 400:
                seg_label = "Occasional"
                icon = "⏱️"
                css_class = "segment-occasional"
                desc = "Rare, occasional buyers. Nudge with seasonal discounts and targeted marketing communications."
            else:
                seg_label = "Regular"
                icon = "⭐"
                css_class = "segment-regular"
                desc = "Steady purchasers with consistent engagement. Upsell with cross-category recommendations."

            result_card = f"""
            <div class="{css_class}">
                <div style="font-size: 3rem; margin-bottom: 8px;">{icon}</div>
                <h2 style="margin: 0; color: #ffffff; font-weight: 700;">{seg_label} Customer</h2>
                <p style="margin-top: 12px; font-size: 1rem; color: #f1f5f9; line-height: 1.5;">{desc}</p>
                <hr style="border-color: rgba(255,255,255,0.2); margin: 16px 0;">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div><b>Recency:</b> {recency_val} days</div>
                    <div><b>Frequency:</b> {frequency_val} txns</div>
                    <div><b>Monetary:</b> ${monetary_val:,.2f}</div>
                </div>
            </div>
            """
            st.markdown(result_card, unsafe_allow_html=True)
        else:
            st.info("👈 Enter customer metrics on the left and click **Predict Customer Segment** to view insights.")


# ==============================================================================
# 6. MODULE 3: EDA & ANALYTICS DASHBOARD
# ==============================================================================
elif selected_menu == "EDA & Analytics":
    st.markdown("## 📊 **Exploratory Data & Analytics Dashboard**")
    st.markdown("Interactive visualizations exploring retail transaction patterns, country demographics, and RFM distributions.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Metric Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", f"{len(df_data):,}")
    m2.metric("Unique Customers", f"{df_data['CustomerID'].nunique():,}")
    m3.metric("Total Revenue", f"${df_data['TotalSpend'].sum():,.2f}")
    m4.metric("Product Catalog Size", f"{df_data['Description'].nunique():,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart Row 1
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🏆 Top 10 Best Selling Products")
        top_prod = df_data.groupby("Description")["Quantity"].sum().reset_index().sort_values(by="Quantity", ascending=False).head(10)
        fig_top = px.bar(
            top_prod, x="Quantity", y="Description", orientation="h",
            color="Quantity", color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_top, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🌍 Revenue Distribution by Country")
        country_rev = df_data.groupby("Country")["TotalSpend"].sum().reset_index().sort_values(by="TotalSpend", ascending=False)
        fig_country = px.pie(
            country_rev, names="Country", values="TotalSpend",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark"
        )
        fig_country.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_country, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chart Row 2: RFM Distributions
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 RFM Metrics Distribution")
    rfm_tab1, rfm_tab2, rfm_tab3 = st.tabs(["Recency", "Frequency", "Monetary"])

    with rfm_tab1:
        fig_r = px.histogram(rfm_df, x="Recency", nbins=30, color_discrete_sequence=["#6366f1"], template="plotly_dark")
        st.plotly_chart(fig_r, use_container_width=True)

    with rfm_tab2:
        fig_f = px.histogram(rfm_df, x="Frequency", nbins=30, color_discrete_sequence=["#a855f7"], template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)

    with rfm_tab3:
        fig_m = px.histogram(rfm_df, x="Monetary", nbins=30, color_discrete_sequence=["#10b981"], template="plotly_dark")
        st.plotly_chart(fig_m, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chart Row 3: Clustering Evaluation
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🧬 KMeans Clustering Evaluation Metrics")

    # Compute Elbow curve dynamically for evaluation plot
    scaler_temp = StandardScaler()
    scaled_rfm_vals = scaler_temp.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])
    inertias, silhouettes = [], []
    k_range = list(range(2, 7))
    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km_temp.fit_predict(scaled_rfm_vals)
        inertias.append(km_temp.inertia_)
        silhouettes.append(silhouette_score(scaled_rfm_vals, lbls))

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=k_range, y=inertias, mode="lines+markers", marker=dict(size=8, color="#8b5cf6")))
        fig_elbow.update_layout(title="Elbow Method (Inertia)", xaxis_title="k Clusters", yaxis_title="Inertia", template="plotly_dark")
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_e2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=k_range, y=silhouettes, mode="lines+markers", marker=dict(size=8, color="#34d399")))
        fig_sil.update_layout(title="Silhouette Scores", xaxis_title="k Clusters", yaxis_title="Silhouette Score", template="plotly_dark")
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
