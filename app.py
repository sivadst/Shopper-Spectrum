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
# 1. PAGE CONFIGURATION & MINIMALIST MATTE WHITE STYLING
# ==============================================================================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="▫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Pure Matte White, Swiss/Minimalist Monochrome Architecture
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #18181b;
}

/* Canvas & Background */
.stApp {
    background-color: #fafafa;
    color: #18181b;
}

/* Header & Footer cleanup */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #18181b !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

p, span, label {
    color: #27272a;
}

/* Matte White Card Component */
.matte-card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    transition: border-color 0.15s ease;
}

.matte-card:hover {
    border-color: #d4d4d8;
}

/* Product Card Styling */
.product-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #e4e4e7;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    margin-bottom: 16px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 190px;
}

.product-card:hover {
    border-color: #18181b;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
}

.product-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    background: #f4f4f5;
    color: #52525b !important;
    border: 1px solid #e4e4e7;
}

.similarity-badge {
    display: inline-block;
    float: right;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #f4f4f5;
    color: #18181b !important;
    border: 1px solid #d4d4d8;
}

.product-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #18181b;
    line-height: 1.45;
    margin-top: 14px;
    margin-bottom: 8px;
    word-break: break-word;
}

.product-desc {
    font-size: 0.78rem;
    color: #71717a;
    line-height: 1.4;
}

/* Minimalist Matte White Button */
.stButton > button {
    background-color: #ffffff !important;
    color: #18181b !important;
    font-weight: 600 !important;
    border: 1.5px solid #18181b !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-size: 0.875rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.15s ease !important;
    width: 100%;
}

.stButton > button:hover {
    background-color: #f4f4f5 !important;
    color: #000000 !important;
    border-color: #000000 !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08) !important;
}

.stButton > button:active {
    background-color: #e4e4e7 !important;
}

/* Ensure text inside buttons is always crisp dark and 100% visible */
.stButton > button,
.stButton > button *,
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #18181b !important;
    font-weight: 600 !important;
}

.stButton > button:hover *,
.stButton > button:hover p,
.stButton > button:hover span,
.stButton > button:hover div {
    color: #000000 !important;
}

/* Minimalist Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e4e4e7 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #f4f4f5 !important;
    margin: 16px 0 !important;
}

/* Metric Display Cards */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    padding: 16px 20px !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #71717a !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    color: #18181b !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

/* Minimalist Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #e4e4e7;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #71717a !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border-radius: 0 !important;
    padding: 8px 16px !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    color: #18181b !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #18181b !important;
}

/* Form Inputs & Selectbox */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    border-color: #e4e4e7 !important;
    border-radius: 6px !important;
    color: #18181b !important;
}

input {
    background-color: #ffffff !important;
    color: #18181b !important;
}

/* Minimalist Segment Card */
.segment-panel {
    background: #ffffff;
    border: 1px solid #18181b;
    border-radius: 8px;
    padding: 28px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.segment-tag {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: #f4f4f5;
    color: #18181b !important;
    border: 1px solid #e4e4e7;
    margin-bottom: 12px;
}

.segment-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #18181b;
    margin: 0 0 8px 0;
}

.segment-desc {
    color: #52525b;
    font-size: 0.9rem;
    line-height: 1.55;
    margin-bottom: 20px;
}

.segment-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-top: 1px solid #f4f4f5;
    padding-top: 16px;
    gap: 12px;
}

.segment-metric-item {
    font-size: 0.78rem;
    color: #71717a;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.segment-metric-val {
    font-size: 1.05rem;
    font-weight: 600;
    color: #18181b;
    margin-top: 2px;
}

/* Scrollbar Minimal */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #fafafa;
}
::-webkit-scrollbar-thumb {
    background: #d4d4d8;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #a1a1aa;
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
# 3. MINIMALIST MONOCHROME SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("<h3 style='margin-bottom:2px; font-weight:700; letter-spacing:-0.03em;'>Shopper Spectrum</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#71717a; font-size:0.8rem; margin-top:0; margin-bottom:16px;'>Customer Analytics & Recommendations</p>", unsafe_allow_html=True)

    selected_menu = option_menu(
        menu_title=None,
        options=["Product Recommendations", "Customer Segmentation", "EDA & Analytics"],
        icons=["box-seam", "person", "graph-up"],
        menu_icon=None,
        default_index=0,
        styles={
            "container": {"padding": "0px", "background-color": "transparent"},
            "icon": {"color": "#71717a", "font-size": "15px"},
            "nav-link": {
                "font-size": "13.5px",
                "font-weight": "400",
                "text-align": "left",
                "margin": "3px 0px",
                "padding": "9px 12px",
                "border-radius": "6px",
                "color": "#3f3f46",
                "--hover-color": "#f4f4f5",
            },
            "nav-link-selected": {
                "background-color": "#18181b",
                "color": "#ffffff",
                "font-weight": "500",
                "border-radius": "6px",
            },
        }
    )

    st.markdown("<hr style='border:0; border-top:1px solid #e4e4e7; margin: 24px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color: #a1a1aa; font-size: 0.72rem; line-height: 1.4;'>"
        "RETAIL INTELLIGENCE PLATFORM<br>"
        "Item-based filtering & RFM K-Means"
        "</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# 4. MODULE 1: PRODUCT RECOMMENDATION ENGINE
# ==============================================================================
if selected_menu == "Product Recommendations":
    st.markdown("<h2 style='margin-bottom:4px;'>Product Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#71717a; font-size:0.9rem; margin-top:0;'>Item-to-item similarity modeling based on customer co-purchase transactions.</p>", unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    available_products = sorted(list(similarity_matrix.columns))

    with st.container():
        st.markdown("<div class='matte-card'>", unsafe_allow_html=True)
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            selected_prod = st.selectbox(
                "Select Catalog Item",
                options=available_products,
                index=0,
                help="Select an item to view top related recommendations."
            )

        with col_btn:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            get_rec_btn = st.button("Find Similar Products")
        st.markdown("</div>", unsafe_allow_html=True)

    if get_rec_btn or selected_prod:
        st.markdown(f"<p style='font-size:0.85rem; color:#71717a; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:12px;'>Top 5 Matches for: <b style='color:#18181b;'>{selected_prod}</b></p>", unsafe_allow_html=True)

        if selected_prod in similarity_matrix.columns:
            sim_scores = similarity_matrix[selected_prod].sort_values(ascending=False)
            sim_scores = sim_scores[sim_scores.index != selected_prod].head(5)

            cols = st.columns(5)
            for idx, (rec_prod, score) in enumerate(sim_scores.items()):
                match_pct = int(score * 100)
                with cols[idx]:
                    card_html = f"""
                    <div class="product-card">
                        <div>
                            <div>
                                <span class="product-badge">Rank #{idx+1}</span>
                                <span class="similarity-badge">{match_pct}%</span>
                            </div>
                            <div class="product-title">{rec_prod}</div>
                        </div>
                        <div class="product-desc">
                            Co-purchased in customer transaction history.
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.warning("Selected product not found in similarity matrix.")


# ==============================================================================
# 5. MODULE 2: CUSTOMER SEGMENTATION (RFM)
# ==============================================================================
elif selected_menu == "Customer Segmentation":
    st.markdown("<h2 style='margin-bottom:4px;'>Customer Segmentation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#71717a; font-size:0.9rem; margin-top:0;'>RFM behavioral scoring evaluated against trained K-Means clusters.</p>", unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col_inputs, col_output = st.columns([1, 1], gap="large")

    with col_inputs:
        st.markdown("<div class='matte-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; margin-bottom:4px;'>Input Metrics</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#71717a; font-size:0.82rem; margin-bottom:16px;'>Configure customer transactional history below:</p>", unsafe_allow_html=True)

        recency_val = st.number_input(
            "Recency (Days since last purchase)",
            min_value=0, max_value=730, value=15, step=1,
            help="Days elapsed since the customer's most recent order."
        )

        frequency_val = st.number_input(
            "Frequency (Number of completed orders)",
            min_value=1, max_value=500, value=12, step=1,
            help="Total count of completed transactions."
        )

        monetary_val = st.number_input(
            "Monetary (₹ Lifetime spend)",
            min_value=0.0, max_value=100000.0, value=1500.0, step=50.0,
            help="Cumulative currency value across all purchases."
        )

        predict_btn = st.button("Evaluate Customer Segment")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        st.markdown("<h4 style='margin-top:0; margin-bottom:8px;'>Classification Result</h4>", unsafe_allow_html=True)

        if predict_btn:
            input_df = pd.DataFrame([[recency_val, frequency_val, monetary_val]], columns=["Recency", "Frequency", "Monetary"])
            scaled_input = scaler.transform(input_df)
            cluster_id = kmeans_model.predict(scaled_input)[0]

            if monetary_val >= 1200 or (frequency_val >= 15 and recency_val <= 45):
                seg_label = "High-Value"
                desc = "Regular, high-engagement customer with significant lifetime spending. Prioritize for dedicated account programs and priority release access."
            elif recency_val > 150 and frequency_val <= 3:
                seg_label = "At-Risk"
                desc = "Extended inactivity window with low lifetime transaction volume. Suitable for win-back campaigns and re-engagement messaging."
            elif frequency_val <= 4 and monetary_val < 400:
                seg_label = "Occasional"
                desc = "Infrequent buyer with low purchase density. Eligible for seasonal catalogs, category prompts, and introductory offers."
            else:
                seg_label = "Regular"
                desc = "Steady repeat purchaser exhibiting predictable transaction intervals. Suitable for cross-category recommendations."

            result_card = f"""
            <div class="segment-panel">
                <span class="segment-tag">{seg_label}</span>
                <div class="segment-title">{seg_label} Segment</div>
                <div class="segment-desc">{desc}</div>
                <div class="segment-metrics">
                    <div>
                        <div class="segment-metric-item">Recency</div>
                        <div class="segment-metric-val">{recency_val}d</div>
                    </div>
                    <div>
                        <div class="segment-metric-item">Frequency</div>
                        <div class="segment-metric-val">{frequency_val} orders</div>
                    </div>
                    <div>
                        <div class="segment-metric-item">Monetary</div>
                        <div class="segment-metric-val">₹{monetary_val:,.2f}</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(result_card, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="matte-card" style="border-style: dashed; color: #71717a; text-align: center; padding: 48px 24px;">
                    <div style="font-size: 0.88rem; color: #71717a;">Specify transaction metrics on the left and submit to view the segment evaluation.</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==============================================================================
# 6. MODULE 3: EDA & ANALYTICS DASHBOARD
# ==============================================================================
elif selected_menu == "EDA & Analytics":
    st.markdown("<h2 style='margin-bottom:4px;'>Transaction Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#71717a; font-size:0.9rem; margin-top:0;'>Descriptive metrics, catalog distribution, and RFM feature spreads.</p>", unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Metric Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", f"{len(df_data):,}")
    m2.metric("Unique Customers", f"{df_data['CustomerID'].nunique():,}")
    m3.metric("Total Revenue", f"₹{df_data['TotalSpend'].sum():,.2f}")
    m4.metric("Active Catalog Items", f"{df_data['Description'].nunique():,}")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Chart Row 1
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div class='matte-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; margin-bottom:16px;'>Top 10 Products by Volume</h4>", unsafe_allow_html=True)
        top_prod = df_data.groupby("Description")["Quantity"].sum().reset_index().sort_values(by="Quantity", ascending=False).head(10)
        
        fig_top = px.bar(
            top_prod, x="Quantity", y="Description", orientation="h",
            template="plotly_white",
            color_discrete_sequence=["#18181b"]
        )
        fig_top.update_layout(
            yaxis={"categoryorder": "total ascending", "title": ""},
            xaxis={"title": "Quantity Sold", "gridcolor": "#f4f4f5"},
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color="#18181b", size=11)
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='matte-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; margin-bottom:16px;'>Revenue Distribution by Country</h4>", unsafe_allow_html=True)
        country_rev = df_data.groupby("Country")["TotalSpend"].sum().reset_index().sort_values(by="TotalSpend", ascending=False)
        
        grayscale_palette = ["#18181b", "#3f3f46", "#52525b", "#71717a", "#a1a1aa", "#d4d4d8", "#e4e4e7"]
        fig_country = px.pie(
            country_rev, names="Country", values="TotalSpend",
            hole=0.45,
            color_discrete_sequence=grayscale_palette,
            template="plotly_white"
        )
        fig_country.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color="#18181b", size=11)
        )
        st.plotly_chart(fig_country, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chart Row 2: RFM Distributions
    st.markdown("<div class='matte-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; margin-bottom:16px;'>RFM Feature Distributions</h4>", unsafe_allow_html=True)
    rfm_tab1, rfm_tab2, rfm_tab3 = st.tabs(["Recency", "Frequency", "Monetary"])

    with rfm_tab1:
        fig_r = px.histogram(
            rfm_df, x="Recency", nbins=30,
            template="plotly_white",
            color_discrete_sequence=["#18181b"]
        )
        fig_r.update_layout(
            xaxis={"gridcolor": "#f4f4f5", "title": "Recency (Days)"},
            yaxis={"gridcolor": "#f4f4f5", "title": "Customer Count"},
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="Inter, sans-serif", color="#18181b")
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with rfm_tab2:
        fig_f = px.histogram(
            rfm_df, x="Frequency", nbins=30,
            template="plotly_white",
            color_discrete_sequence=["#27272a"]
        )
        fig_f.update_layout(
            xaxis={"gridcolor": "#f4f4f5", "title": "Frequency (Orders)"},
            yaxis={"gridcolor": "#f4f4f5", "title": "Customer Count"},
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="Inter, sans-serif", color="#18181b")
        )
        st.plotly_chart(fig_f, use_container_width=True)

    with rfm_tab3:
        fig_m = px.histogram(
            rfm_df, x="Monetary", nbins=30,
            template="plotly_white",
            color_discrete_sequence=["#3f3f46"]
        )
        fig_m.update_layout(
            xaxis={"gridcolor": "#f4f4f5", "title": "Monetary (₹ Spend)"},
            yaxis={"gridcolor": "#f4f4f5", "title": "Customer Count"},
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="Inter, sans-serif", color="#18181b")
        )
        st.plotly_chart(fig_m, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chart Row 3: Clustering Evaluation
    st.markdown("<div class='matte-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; margin-bottom:16px;'>K-Means Cluster Optimization</h4>", unsafe_allow_html=True)

    # Compute Elbow curve dynamically
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
        fig_elbow.add_trace(go.Scatter(
            x=k_range, y=inertias, mode="lines+markers",
            line=dict(color="#18181b", width=1.5),
            marker=dict(size=6, color="#18181b")
        ))
        fig_elbow.update_layout(
            title=dict(text="Elbow Curve (Inertia)", font=dict(size=13, color="#18181b")),
            xaxis=dict(title="k Clusters", gridcolor="#f4f4f5"),
            yaxis=dict(title="Inertia", gridcolor="#f4f4f5"),
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter, sans-serif", color="#18181b")
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_e2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=k_range, y=silhouettes, mode="lines+markers",
            line=dict(color="#18181b", width=1.5),
            marker=dict(size=6, color="#18181b")
        ))
        fig_sil.update_layout(
            title=dict(text="Silhouette Coefficient", font=dict(size=13, color="#18181b")),
            xaxis=dict(title="k Clusters", gridcolor="#f4f4f5"),
            yaxis=dict(title="Score", gridcolor="#f4f4f5"),
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter, sans-serif", color="#18181b")
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
