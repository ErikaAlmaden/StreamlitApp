import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx

# --- Page Title ---
st.title("Retail Insights Dashboard")
st.sidebar.header("Upload Your Dataset")
st.sidebar.write("The data set must have a Transaction_ID and Product or Products")

# --- File Upload ---
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())

    # Clean Transaction_ID: Remove commas
    if 'Transaction_ID' in data.columns:
        data['Transaction_ID'] = data['Transaction_ID'].astype(str).str.replace(',', '')
    else:
        st.error("Dataset must contain a column named 'Transaction_ID'.")
        st.stop()

    # Check for the product column name
    product_column = None
    if 'Products' in data.columns:
        product_column = 'Products'
    elif 'Product' in data.columns:
        product_column = 'Product'
    else:
        st.error("Dataset must contain a column named 'Products' or 'Product'.")
        st.stop()

    # Ensure the required columns are present
    if 'Transaction_ID' in data.columns and product_column:
        # Preprocess Data
        st.write("### Data Preprocessing")

        # Keep only Transaction_ID and Product columns
        transaction_data = data[['Transaction_ID', product_column]]

        # Ensure the product column is interpreted as a list
        transaction_data[product_column] = transaction_data[product_column].apply(eval)

        # Explode the product column to create one row per product per transaction
        exploded_data = transaction_data.explode(product_column)

        # Create one-hot encoded format with Transaction_ID as rows and Products as columns
        basket = exploded_data.pivot_table(
            index='Transaction_ID',
            columns=product_column,
            aggfunc=lambda x: 1,
            fill_value=0
        )

        st.write("### One-Hot Encoded Basket Format")
        st.write(basket.head())

        # Frequent Itemsets Mining
        min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.03)
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            st.warning("No frequent itemsets found. Try lowering the minimum support threshold.")
        else:
            st.write("### Frequent Itemsets")
            st.dataframe(frequent_itemsets, use_container_width=True)

            # Association Rules
            metric = st.sidebar.selectbox("Metric", ["lift", "confidence", "support"])
            threshold = st.sidebar.slider(f"Minimum {metric.capitalize()}", 0.1, 2.0, 1.0)
            rules = association_rules(frequent_itemsets, metric=metric, min_threshold=threshold)

            if rules.empty:
                st.warning("No association rules found. Try adjusting the metric or threshold.")
            else:
                st.write("### Association Rules")
                st.write(rules)

                # Visualizations
                st.write("### Visualizations")
                
                # Bar Chart for Frequent Itemsets
                st.write("#### Bar Chart of Frequent Itemsets")
                fig, ax = plt.subplots()
                frequent_itemsets.sort_values("support", ascending=False).head(10).plot(
                    x="itemsets", y="support", kind="bar", ax=ax, legend=False
                )
                ax.set_ylabel("Support")
                st.pyplot(fig)

                # Network Graph for Rules
                st.write("#### Association Rules Network")
                G = nx.DiGraph()
                for _, row in rules.iterrows():
                    G.add_edge(', '.join(list(row['antecedents'])), ', '.join(list(row['consequents'])), weight=row['lift'])
                pos = nx.spring_layout(G, seed=42)
                plt.figure(figsize=(10, 8))
                nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
                st.pyplot(plt)

                # Recommendations
                st.write("### Recommendations")
                for _, row in rules.iterrows():
                    st.write(f"If a customer buys **{', '.join(row['antecedents'])}**, consider recommending **{', '.join(row['consequents'])}**.")
    else:
        st.error("Dataset must contain 'Transaction_ID' and a product column ('Products' or 'Product').")
else:
    st.info("Awaiting CSV file upload.")
