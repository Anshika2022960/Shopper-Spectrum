import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_files():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    similarity_df = pickle.load(open("similarity_df.pkl", "rb"))
    cluster_mapping = pickle.load(open("cluster_mapping.pkl", "rb"))
    product_names = pickle.load(open("product_names.pkl", "rb"))
    return scaler, kmeans, similarity_df, cluster_mapping, product_names

scaler, kmeans, similarity_df, cluster_mapping, product_names = load_files()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fdf2f8 0%, #eff6ff 45%, #ecfdf5 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e3a8a 55%, #7c3aed 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(120deg, #2563eb, #7c3aed, #db2777);
    color: white;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.18);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 21px;
}

.module-card {
    padding: 25px;
    border-radius: 22px;
    background: white;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.10);
    border-top: 7px solid #2563eb;
    margin-bottom: 18px;
}

.kpi-card-blue {
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    text-align: center;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
}

.kpi-card-green {
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    text-align: center;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
}

.kpi-card-orange {
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #ffedd5, #fed7aa);
    text-align: center;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
}

.recommend-card {
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #ffffff, #eef2ff);
    border-left: 8px solid #7c3aed;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.10);
    margin-bottom: 14px;
}

.segment-result {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(135deg, #22c55e, #14b8a6);
    color: white;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.18);
}

.action-box {
    padding: 18px;
    border-radius: 16px;
    background: #fff7ed;
    border-left: 7px solid #f97316;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.markdown("### 📌 Dashboard Menu")

menu = st.sidebar.radio(
    "Choose Module",
    [
        "🏠 Home",
        "🛍️ Product Recommendation",
        "👥 Customer Segmentation",
        "📊 Business Insights",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("ML + E-Commerce Analytics")
st.sidebar.info("RFM | K-Means | Cosine Similarity")

st.markdown("""
<div class="hero">
    <h1>🛒 Shopper Spectrum</h1>
    <p>Customer Segmentation and Product Recommendation Dashboard</p>
</div>
""", unsafe_allow_html=True)

if menu == "🏠 Home":
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.header("📌 Project Overview")
    st.write(
        "This dashboard helps e-commerce businesses understand customer behavior, "
        "segment customers using RFM-based K-Means clustering, and recommend similar products "
        "using item-based collaborative filtering."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="kpi-card-blue">
            <h2>👥</h2>
            <h3>Customer Segmentation</h3>
            <p>RFM + K-Means</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="kpi-card-green">
            <h2>🛍️</h2>
            <h3>Product Recommendation</h3>
            <p>Collaborative Filtering</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="kpi-card-orange">
            <h2>📐</h2>
            <h3>Similarity Engine</h3>
            <p>Cosine Similarity</p>
        </div>
        """, unsafe_allow_html=True)

elif menu == "🛍️ Product Recommendation":
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.header("🛍️ Product Recommendation Module")
    st.write("Enter or select a product name to get the top 5 similar products.")
    st.markdown("</div>", unsafe_allow_html=True)

    search_mode = st.radio(
        "Choose Input Method",
        ["Select from Dropdown", "Type Product Name"],
        horizontal=True
    )

    if search_mode == "Select from Dropdown":
        product_name = st.selectbox("🛒 Select Product", sorted(product_names))
    else:
        product_name = st.text_input(
            "⌨️ Enter Product Name",
            placeholder="Example: DOLLY GIRL BEAKER"
        )

    if st.button("🔍 Get Recommendations", use_container_width=True):
        if product_name.strip() == "":
            st.warning("Please enter or select a product name.")
        else:
            product_name_clean = product_name.strip().upper()
            products_dict = {p.upper(): p for p in product_names}

            if product_name_clean not in products_dict:
                st.error("Product not found. Please check the product name.")
                with st.expander("View sample product names"):
                    st.write(product_names[:40])
            else:
                original_product = products_dict[product_name_clean]
                recommendations = similarity_df[original_product].sort_values(ascending=False)[1:6]

                st.success(f"Top 5 recommendations for: {original_product}")

                for i, (product, score) in enumerate(recommendations.items(), start=1):
                    st.markdown(f"""
                    <div class="recommend-card">
                        <h3>#{i} 🛍️ {product}</h3>
                        <p><b>Similarity Score:</b> {score:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

elif menu == "👥 Customer Segmentation":
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.header("👥 Customer Segmentation Module")
    st.write("Enter customer RFM values to predict the customer segment.")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("📅 Recency (days)", min_value=0, value=30)

    with col2:
        frequency = st.number_input("🔁 Frequency", min_value=1, value=5)

    with col3:
        monetary = st.number_input("💰 Monetary", min_value=0.0, value=1000.0)

    if st.button("🎯 Predict Cluster", use_container_width=True):
        input_data = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        scaled_input = scaler.transform(input_data)
        cluster = kmeans.predict(scaled_input)[0]
        segment = cluster_mapping[cluster]

        st.markdown(f"""
        <div class="segment-result">
            Customer Segment: {segment}
        </div>
        """, unsafe_allow_html=True)

        st.info(f"Predicted Cluster Number: {cluster}")

        if segment == "High-Value":
            action = "Offer loyalty rewards, premium discounts, and early access deals."
        elif segment == "Regular":
            action = "Encourage repeat purchases using personalized offers."
        elif segment == "Occasional":
            action = "Send promotional campaigns to increase engagement."
        else:
            action = "Use win-back offers and retention campaigns."

        st.markdown(f"""
        <div class="action-box">
            <h4>📢 Recommended Business Action</h4>
            <p>{action}</p>
        </div>
        """, unsafe_allow_html=True)

elif menu == "📊 Business Insights":
    st.header("📊 Business Insights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Products", len(product_names))

    with col2:
        st.metric("Customer Segments", 4)

    with col3:
        st.metric("ML Model", "K-Means")

    with col4:
        st.metric("Recommendation", "Cosine Similarity")

    st.markdown("---")

    st.subheader("💡 Key Business Insights")

    st.markdown("""
    <div class="module-card">
    <ul>
        <li><b>High-Value Customers:</b> Recently purchased, buy frequently, and spend more.</li>
        <li><b>Regular Customers:</b> Stable customers with moderate purchase behavior.</li>
        <li><b>Occasional Customers:</b> Purchase rarely and need engagement campaigns.</li>
        <li><b>At-Risk Customers:</b> Have not purchased recently and need retention offers.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📌 Recommended Business Actions")

    st.markdown("""
    <div class="recommend-card">
        <h4>🟢 High-Value Customers</h4>
        <p>Provide loyalty rewards, premium offers, and early access to new products.</p>
    </div>

    <div class="recommend-card">
        <h4>🔵 Regular Customers</h4>
        <p>Send personalized discounts and bundle offers to increase purchase frequency.</p>
    </div>

    <div class="recommend-card">
        <h4>🟠 Occasional Customers</h4>
        <p>Use festival offers, reminder emails, and limited-time discounts.</p>
    </div>

    <div class="recommend-card">
        <h4>🔴 At-Risk Customers</h4>
        <p>Use win-back campaigns, coupons, and personalized reactivation offers.</p>
    </div>
    """, unsafe_allow_html=True)

elif menu == "ℹ️ About Project":
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.header("ℹ️ About Project")
    st.markdown("""
    ### Project Title
    **Shopper Spectrum: Customer Segmentation and Product Recommendation in E-Commerce**

    ### Techniques Used
    - RFM Analysis
    - K-Means Clustering
    - Elbow Method
    - Silhouette Score
    - Item-Based Collaborative Filtering
    - Cosine Similarity
    - Streamlit Dashboard

    ### Tools
    Python, Pandas, Scikit-learn, Streamlit, Pickle
    """)
    st.markdown("</div>", unsafe_allow_html=True)