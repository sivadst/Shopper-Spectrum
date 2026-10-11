import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import nbformat as nbf

def generate_pipeline():
    print("Generating synthetic retail dataset...")
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
        ("22382", "LUNCH BAG PINK POLKADOT", 1.65),
        ("22384", "LUNCH BAG PINK RETROSPOT", 1.65),
        ("22666", "RECIPE BOX PANTRY TEATIME", 2.95),
        ("22630", "DOLLY GIRL LUNCH BOX", 1.95),
        ("22629", "SPACEBOY LUNCH BOX", 1.95),
        ("22726", "ALARM CLOCK BAKELIKE GREEN", 3.75),
        ("22727", "ALARM CLOCK BAKELIKE RED", 3.75),
        ("22728", "ALARM CLOCK BAKELIKE PINK", 3.75),
        ("22960", "JAM MAKING SET WITH JARS", 4.25),
        ("23209", "LUNCH BAG VINTAGE DOLLY PARK", 1.65),
        ("22457", "NATURAL SLATE HEART CHALKBOARD", 2.95)
    ]

    countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland"]
    country_weights = [0.75, 0.08, 0.06, 0.04, 0.02, 0.02, 0.015, 0.015]

    num_customers = 250
    customer_ids = [10000 + i for i in range(num_customers)]

    # Assign customer archetypes to ensure distinct RFM clusters
    # Archetype 1: High Value (15%): Recent, High Freq, High Spend
    # Archetype 2: Regular (35%): Fairly recent, Moderate Freq, Moderate Spend
    # Archetype 3: Occasional (30%): Older recency, Low Freq, Low Spend
    # Archetype 4: At Risk (20%): Very old recency, Low Freq, Moderate/Low Spend

    records = []
    base_date = pd.Timestamp("2023-12-31")

    for cid in customer_ids:
        r_val = np.random.rand()
        if r_val < 0.15:
            # High-value
            num_txns = np.random.randint(15, 35)
            recency_days_max = 30
        elif r_val < 0.50:
            # Regular
            num_txns = np.random.randint(5, 15)
            recency_days_max = 90
        elif r_val < 0.80:
            # Occasional
            num_txns = np.random.randint(2, 6)
            recency_days_max = 180
        else:
            # At-Risk
            num_txns = np.random.randint(1, 4)
            recency_days_max = 365

        country = np.random.choice(countries, p=country_weights)

        for txn_idx in range(num_txns):
            inv_no = f"58{np.random.randint(10000, 99999)}"
            days_ago = np.random.randint(0 if recency_days_max <= 30 else (31 if recency_days_max <= 90 else (91 if recency_days_max <= 180 else 181)), recency_days_max + 1)
            inv_date = base_date - pd.Timedelta(days=days_ago, hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))

            # Select 1 to 5 products per invoice
            items_in_inv = np.random.randint(1, 6)
            chosen_items = np.random.choice(len(products), size=items_in_inv, replace=False)

            for prod_idx in chosen_items:
                stock_code, desc, base_price = products[prod_idx]
                qty = np.random.randint(1, 24 if r_val < 0.15 else 12)
                price = round(base_price * np.random.uniform(0.9, 1.1), 2)
                records.append({
                    "InvoiceNo": inv_no,
                    "StockCode": stock_code,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": inv_date,
                    "UnitPrice": price,
                    "Country": country,
                    "CustomerID": cid
                })

    # Add a few cancelled/negative/missing records to mimic raw data preprocessing steps
    records.append({"InvoiceNo": "C581234", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "Quantity": -1, "InvoiceDate": base_date, "UnitPrice": 12.75, "Country": "United Kingdom", "CustomerID": 10001})
    records.append({"InvoiceNo": "581235", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "Quantity": 2, "InvoiceDate": base_date, "UnitPrice": 12.75, "Country": "United Kingdom", "CustomerID": np.nan})
    records.append({"InvoiceNo": "581236", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "Quantity": 0, "InvoiceDate": base_date, "UnitPrice": 0.0, "Country": "United Kingdom", "CustomerID": 10002})

    df_raw = pd.DataFrame(records)
    df_raw.to_csv("sample_data.csv", index=False)
    print(f"Saved sample_data.csv with {len(df_raw)} records.")

    # Clean Data
    df = df_raw.dropna(subset=["CustomerID"]).copy()
    df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

    # Calculate RFM
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalSpend": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # Scale RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Label Clusters based on mean values
    cluster_means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    print("Cluster Means:\n", cluster_means)

    # Assign human-readable labels to clusters based on mean characteristics
    # High-Value: high M, high F, low R
    # At-Risk: high R, low F, low M
    # Regular: mid F, mid M, lower R
    # Occasional: lower F, lower M, mid/high R

    # Sort cluster IDs by Monetary desc
    sorted_by_m = cluster_means.sort_values(by="Monetary", ascending=False).index.tolist()

    # Label mapping dynamically based on rank
    # highest spend -> High-Value
    # second highest spend -> Regular
    # lowest spend + highest recency -> At-Risk
    # remaining -> Occasional

    label_mapping = {}
    label_mapping[sorted_by_m[0]] = "High-Value"

    # Find worst recency among remaining for At-Risk
    remaining = sorted_by_m[1:]
    highest_recency_cluster = cluster_means.loc[remaining, "Recency"].idxmax()
    label_mapping[highest_recency_cluster] = "At-Risk"

    remaining.remove(highest_recency_cluster)
    if len(remaining) == 2:
        # higher monetary -> Regular, lower -> Occasional
        if cluster_means.loc[remaining[0], "Monetary"] >= cluster_means.loc[remaining[1], "Monetary"]:
            label_mapping[remaining[0]] = "Regular"
            label_mapping[remaining[1]] = "Occasional"
        else:
            label_mapping[remaining[1]] = "Regular"
            label_mapping[remaining[0]] = "Occasional"

    rfm["Segment"] = rfm["Cluster"].map(label_mapping)
    print("Assigned Segment Mapping:", label_mapping)

    # Save Scaler & KMeans model
    joblib.dump(scaler, "rfm_scaler.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")

    rfm_summary = {
        "cluster_means": cluster_means,
        "label_mapping": label_mapping
    }
    joblib.dump(rfm_summary, "rfm_summary.pkl")

    # Collaborative Filtering Matrix
    user_item_matrix = df.pivot_table(index="CustomerID", columns="Description", values="Quantity", aggfunc="sum", fill_value=0)
    # Binarize or item-item similarity based on customer purchase vectors
    item_item_matrix = pd.DataFrame(
        cosine_similarity(user_item_matrix.T),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    joblib.dump(item_item_matrix, "similarity_matrix.pkl")
    print("Saved similarity_matrix.pkl, kmeans_model.pkl, and rfm_scaler.pkl successfully!")

    # Build Jupyter Notebook
    nb = nbf.v4.new_notebook()

    cells = []

    # Markdown Title
    cells.append(nbf.v4.new_markdown_cell("""# 🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce
### *An End-to-End Unsupervised Machine Learning and Collaborative Filtering Pipeline*

---

## 📌 Project Overview
This notebook demonstrates a complete e-commerce data analytics and machine learning pipeline:
1. **Data Cleaning & Preprocessing**: Handling missing values, returns, and improper transactions.
2. **Exploratory Data Analysis (EDA)**: Insights into sales volume, top products, country distributions, and purchasing patterns.
3. **RFM Feature Engineering**: Calculating **Recency**, **Frequency**, and **Monetary** value for customer behavior modeling.
4. **Customer Segmentation via KMeans Clustering**: Scaled feature normalization, Elbow & Silhouette evaluation, and cluster profiling.
5. **Item-Based Collaborative Filtering**: Generating product-product similarity matrix using cosine distance.
6. **Model Persistence**: Exporting models and artifacts for live deployment in Streamlit.
"""))

    # Imports Cell
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
warnings.filterwarnings("ignore")

print("All required libraries successfully imported.")
"""))

    # Step 1: Data Collection & Understanding
    cells.append(nbf.v4.new_markdown_cell("""## 📥 Step 1: Dataset Collection & Preprocessing
We load the retail transaction dataset containing `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `Country`, and `CustomerID`.
"""))

    cells.append(nbf.v4.new_code_cell("""# Load dataset
df_raw = pd.read_csv("sample_data.csv")
print(f"Raw Dataset Shape: {df_raw.shape}")
df_raw.head()
"""))

    cells.append(nbf.v4.new_code_cell("""# Data Preprocessing
print("Initial Missing Values:")
print(df_raw.isnull().sum())

# 1. Remove missing CustomerID
df_clean = df_raw.dropna(subset=['CustomerID']).copy()

# 2. Exclude cancelled invoices (starting with 'C')
df_clean = df_clean[~df_clean['InvoiceNo'].str.startswith('C', na=False)]

# 3. Remove non-positive quantities and prices
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# Format data types
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['TotalSpend'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"\\nCleaned Dataset Shape: {df_clean.shape}")
df_clean.describe()
"""))

    # Step 2: EDA
    cells.append(nbf.v4.new_markdown_cell("""## 📊 Step 2: Exploratory Data Analysis (EDA)
We visualize sales trends, top-selling items, and geographical distribution using Plotly.
"""))

    cells.append(nbf.v4.new_code_cell("""# Top 10 Selling Products
top_products = df_clean.groupby('Description')['Quantity'].sum().reset_index()
top_products = top_products.sort_values(by='Quantity', ascending=False).head(10)

fig = px.bar(
    top_products,
    x='Quantity',
    y='Description',
    orientation='h',
    title='<b>Top 10 Selling Products by Quantity</b>',
    color='Quantity',
    color_continuous_scale='Blues'
)
fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
fig.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Revenue by Country
country_revenue = df_clean.groupby('Country')['TotalSpend'].sum().reset_index().sort_values(by='TotalSpend', ascending=False)

fig_country = px.bar(
    country_revenue,
    x='Country',
    y='TotalSpend',
    title='<b>Total Spend by Country</b>',
    color='TotalSpend',
    color_continuous_scale='Tealgrn'
)
fig_country.update_layout(template='plotly_dark')
fig_country.show()
"""))

    # Step 3: RFM Feature Engineering & Clustering
    cells.append(nbf.v4.new_markdown_cell("""## 🧠 Step 3: RFM Feature Engineering & Customer Segmentation
- **Recency**: Days since last purchase relative to snapshot date.
- **Frequency**: Total unique transaction count per customer.
- **Monetary**: Total spend per customer.
"""))

    cells.append(nbf.v4.new_code_cell("""# Calculate snapshot date
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

# Group by CustomerID
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSpend': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm.head()
"""))

    cells.append(nbf.v4.new_code_cell("""# Standardize RFM Values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Determine optimal K using Elbow Method & Silhouette Score
inertia = []
silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(rfm_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, cluster_labels))

# Plot Elbow and Silhouette
fig_eval = go.Figure()
fig_eval.add_trace(go.Scatter(x=list(K_range), y=inertia, name='Inertia (Elbow)', mode='lines+markers'))
fig_eval.update_layout(title='<b>KMeans Elbow Curve</b>', xaxis_title='Number of Clusters (k)', yaxis_title='Inertia', template='plotly_dark')
fig_eval.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Train final KMeans Model with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Segment Interpretation
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("Cluster Summary (Means):")
print(cluster_summary)
"""))

    cells.append(nbf.v4.new_code_cell("""# 3D Scatter Plot of RFM Clusters
fig_3d = px.scatter_3d(
    rfm,
    x='Recency',
    y='Frequency',
    z='Monetary',
    color='Cluster',
    title='<b>3D Visualization of Customer Segments</b>',
    opacity=0.8
)
fig_3d.update_layout(template='plotly_dark')
fig_3d.show()
"""))

    # Step 4: Product Recommendation System
    cells.append(nbf.v4.new_markdown_cell("""## 🛍️ Step 4: Item-Based Collaborative Filtering Recommendation Engine
We construct a **CustomerID x Product Description** matrix and compute product-product cosine similarity scores.
"""))

    cells.append(nbf.v4.new_code_cell("""# Pivot Table
user_item = df_clean.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)

# Cosine Similarity Matrix
item_similarity = cosine_similarity(user_item.T)
similarity_df = pd.DataFrame(item_similarity, index=user_item.columns, columns=user_item.columns)

# Recommendation function test
def recommend_products(product_name, top_n=5):
    if product_name not in similarity_df.columns:
        return []
    scores = similarity_df[product_name].sort_values(ascending=False)[1:top_n+1]
    return list(zip(scores.index, scores.values))

test_prod = similarity_df.columns[0]
print(f"Sample Recommendations for '{test_prod}':")
for p, s in recommend_products(test_prod):
    print(f" - {p} (Similarity: {s:.2%})")
"""))

    # Step 5: Save Models & Artifacts
    cells.append(nbf.v4.new_markdown_cell("""## 💾 Step 5: Model Artifact Exporting
Save trained models and matrices for deployment into Streamlit app.
"""))

    cells.append(nbf.v4.new_code_cell("""# Save artifacts using joblib
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "rfm_scaler.pkl")
joblib.dump(similarity_df, "similarity_matrix.pkl")

print("Artifacts successfully saved: kmeans_model.pkl, rfm_scaler.pkl, similarity_matrix.pkl")
"""))

    nb['cells'] = cells
    with open("Shopper_Spectrum.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook Shopper_Spectrum.ipynb written successfully!")

if __name__ == "__main__":
    generate_pipeline()
