
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import os

st.set_page_config(page_title="Portfolio Decision Tree Simulator", layout="wide")
st.title("🧬 Portfolio Decision Tree Simulator")

# ---- Load Data ----
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data_path = "portfolio.csv"
df = load_data(data_path)

# ---- Sidebar Adjustments ----
st.sidebar.header("Adjust Assumptions")
discount_rate = st.sidebar.slider("Discount rate (%)", 5.0, 15.0, 10.0) / 100

for idx, row in df.iterrows():
    new_pos = st.sidebar.slider(
        f"{row['Program']} - {row['Indication']} PoS",
        0.0, 1.0, float(row['PoS']), key=f"pos_{idx}", step=0.05
    )
    df.at[idx, 'PoS'] = new_pos

# ---- NPV Calculation ----
def calc_npv(r):
    revenue = r['PeakSales'] * 0.6
    return (revenue / ((1+discount_rate)**(r['LaunchYear']-2025))) * r['PoS'] - r['DevCost']

df['NPV'] = df.apply(calc_npv, axis=1).round(2)

# ---- Decision Tree Visual ----
st.subheader("🌳 Decision Tree View")
dot = graphviz.Digraph("Portfolio", format="png")
dot.attr(rankdir='LR')

dot.node("root", "Portfolio Decision")

programs = df['Program'].unique()
for p in programs:
    dot.node(p, p)
    dot.edge("root", p)
    sub = df[df['Program'] == p]
    for _, r in sub.iterrows():
        leaf_label = f"{r['Indication']}\nNPV: ${r['NPV']:.1f}M\nPoS: {r['PoS']:.2f}"
        leaf_name = f"{p}_{r['Indication']}"
        dot.node(leaf_name, leaf_label, shape="box")
        dot.edge(p, leaf_name)

st.graphviz_chart(dot, use_container_width=True)

# ---- Table & Summary ----
st.subheader("📊 Detailed Metrics")
st.dataframe(df)

total_npv = df['NPV'].sum()
st.markdown(f"### 💰 Total Portfolio NPV: **${total_npv:.1f}M**")

# ---- GPT Summary ----
if "OPENAI_API_KEY" in os.environ:
    try:
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
        prompt = (
            "You are a biotech portfolio strategist. Summarize the implications of the following asset-level "
            "NPVs and probabilities of success. Provide key recommendations in 3–4 sentences.\n\n"
            f"{df.to_markdown(index=False)}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        st.subheader("🤖 AI Summary")
        st.write(summary)
    except Exception as e:
        st.warning(f"GPT summary unavailable: {e}")
else:
    st.info("Set the environment variable OPENAI_API_KEY to enable GPT summaries.")
