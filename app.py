import os
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from segment_studio import (
    load_data,
    fill_missing_values,
    scaling,
    encoding,
    wcss_calc,
    clustering,
    create_cluster_table,
    add_numeric_means,
    add_categorical_modes,
    create_final_csv,
)

OLLAMA_URL = "https://ollama.com/api/chat"
OLLAMA_MODEL = "gpt-oss:120b"
EXPECTED_COLUMNS = ["cluster_id", "count", "name", "description"]


def draw_elbow_graph(wcss_table):
    """Draw the elbow graph inside Streamlit (elbow_graph() uses plt.show()).

    :param wcss_table: A DataFrame with K and WCSS columns.
    :return: None.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wcss_table["K"], wcss_table["WCSS"], "bx-", linewidth=2, markersize=8)
    ax.set_title("The Elbow Method for Optimal K", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
    ax.set_xticks(wcss_table["K"])
    st.pyplot(fig)
    plt.close(fig)


def ask_llm_for_names(cluster_summary, api_key):
    """Send the cluster summary to Ollama and return the answer text.

    :param cluster_summary: The detailed cluster summary table.
    :param api_key: The Ollama API key.
    :return: The text returned by the model.
    """
    prompt = f"""
Analyze the following cluster summary table:

{cluster_summary.to_string(index=False)}

Use the numeric averages and most common categorical values
to create a short meaningful name and a one-line description for every cluster.

Return ONLY CSV with exactly these columns:
cluster_id,count,name,description

Rules:
- One row for every cluster_id in the table.
- Keep the same cluster_id and count values.
- Put the name and description in double quotes.
- No Markdown, no code block.
- Do not add any explanation before or after the CSV.
"""
    response = requests.post(
        OLLAMA_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7},
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def parse_llm_csv(text, cluster_table):
    """Convert the model's CSV text into a DataFrame and check it.

    :param text: The CSV text returned by the model.
    :param cluster_table: The raw cluster table (to check the cluster IDs).
    :return: A DataFrame with cluster_id, count, name and description.
    """
    # Remove a code block if the model added one anyway.
    text = text.strip().replace("```csv", "").replace("```", "").strip()

    llm_table = pd.read_csv(StringIO(text))
    llm_table.columns = [column.strip() for column in llm_table.columns]

    if list(llm_table.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected columns {EXPECTED_COLUMNS}, got {list(llm_table.columns)}")

    llm_table["cluster_id"] = llm_table["cluster_id"].astype(int)

    if sorted(llm_table["cluster_id"]) != sorted(cluster_table["cluster_id"]):
        raise ValueError("The cluster IDs in the response do not match the clusters.")

    return llm_table


def reset_results(from_step):
    """Remove saved results from a step onwards, so old results are not shown.

    :param from_step: The first result key to remove.
    :return: None.
    """
    keys = ["wcss_table", "df_clustered", "cluster_table", "cluster_summary", "llm_table"]
    for key in keys[keys.index(from_step):]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# STEP 1 — Upload CSV
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Segment Studio")
st.title("Segment Studio")
st.write(
    "Segment Studio uses **K-Means clustering** to split the rows of a CSV file "
    "into groups of similar records. An AI model then gives every group a name "
    "and a short description."
)

st.header("Step 1 — Upload CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file to start.")
    st.stop()

# When a new file is uploaded, forget the results of the previous file.
file_id = (uploaded_file.name, uploaded_file.size)
if st.session_state.get("file_id") != file_id:
    st.session_state.clear()
    st.session_state["file_id"] = file_id

try:
    uploaded_file.seek(0)
    df_original = load_data(uploaded_file)
except pd.errors.EmptyDataError:
    st.error("The uploaded CSV file is empty.")
    st.stop()
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if df_original.empty:
    st.error("The uploaded CSV file has no data rows.")
    st.stop()

st.dataframe(df_original)
col1, col2 = st.columns(2)
col1.metric("Rows", df_original.shape[0])
col2.metric("Columns", df_original.shape[1])

n_rows = df_original.shape[0]

# ---------------------------------------------------------------------------
# STEP 2 — Prepare the data
# ---------------------------------------------------------------------------
st.header("Step 2 — Prepare the data")

try:
    df_clean = fill_missing_values(df_original)
    df_scaled = scaling(df_clean)
    df_encoded = encoding(df_scaled)
except Exception as error:
    st.error(f"Could not prepare the data: {error}")
    st.stop()

if df_encoded.isnull().values.any():
    st.error("The data still has missing values after cleaning (for example, a column that is completely empty).")
    st.stop()

st.write(
    "Missing values were filled (median for numbers, most common value for text), "
    "numeric columns were scaled, and categorical columns were encoded."
)
st.write(f"Prepared data for clustering: **{df_encoded.shape[0]} rows × {df_encoded.shape[1]} columns**")

# ---------------------------------------------------------------------------
# STEP 3 — WCSS and Elbow Method
# ---------------------------------------------------------------------------
st.header("Step 3 — WCSS and Elbow Method")

col1, col2 = st.columns(2)
min_k = col1.number_input("Minimum K", min_value=1, value=min(2, n_rows), step=1)
max_k = col2.number_input("Maximum K", min_value=1, value=min(10, n_rows), step=1)

if st.button("Calculate WCSS"):
    if min_k > max_k:
        st.error("Minimum K must be smaller than or equal to Maximum K.")
    elif max_k > n_rows:
        st.error(f"Maximum K cannot be larger than the number of rows ({n_rows}).")
    else:
        reset_results("wcss_table")
        with st.spinner("Calculating WCSS..."):
            try:
                st.session_state["wcss_table"] = wcss_calc(df_encoded, int(min_k), int(max_k))
            except Exception as error:
                st.error(f"Could not calculate WCSS: {error}")

if "wcss_table" not in st.session_state:
    st.stop()

wcss_table = st.session_state["wcss_table"]
st.dataframe(wcss_table, hide_index=True)
draw_elbow_graph(wcss_table)

# ---------------------------------------------------------------------------
# STEP 4 — Create clusters
# ---------------------------------------------------------------------------
st.header("Step 4 — Choose K and create clusters")

k = st.selectbox("Number of clusters (K)", wcss_table["K"].tolist())

if st.button("Create Clusters"):
    reset_results("df_clustered")
    with st.spinner("Creating clusters..."):
        try:
            df_clustered = clustering(int(k), df_encoded)
            cluster_table = create_cluster_table(df_clustered)

            # STEP 5 uses the cleaned (not scaled/encoded) data,
            # so the means and the most common values are easy to understand.
            cluster_summary = add_numeric_means(cluster_table.copy(), df_clean, df_clustered)
            cluster_summary = add_categorical_modes(cluster_summary, df_clean, df_clustered)

            st.session_state["df_clustered"] = df_clustered
            st.session_state["cluster_table"] = cluster_table
            st.session_state["cluster_summary"] = cluster_summary
        except Exception as error:
            st.error(f"Could not create the clusters: {error}")

if "cluster_table" not in st.session_state:
    st.stop()

df_clustered = st.session_state["df_clustered"]
cluster_table = st.session_state["cluster_table"]
cluster_summary = st.session_state["cluster_summary"]

st.subheader("Raw cluster table")
st.dataframe(cluster_table[EXPECTED_COLUMNS], hide_index=True)

# ---------------------------------------------------------------------------
# STEP 5 — Cluster summaries
# ---------------------------------------------------------------------------
st.header("Step 5 — Cluster summary")
st.write("Average of every numeric column and most common value of every categorical column, per cluster.")
st.dataframe(cluster_summary.drop(columns=["name", "description"]), hide_index=True)

# ---------------------------------------------------------------------------
# STEP 6 — LLM names and descriptions
# ---------------------------------------------------------------------------
st.header("Step 6 — Generate names and descriptions")

if st.button("Generate Names and Descriptions"):
    reset_results("llm_table")
    api_key = os.getenv("OLLAMA_API_KEY")

    if not api_key:
        st.error("The OLLAMA_API_KEY environment variable is not set. Set it and restart the app.")
    else:
        llm_text = None
        with st.spinner("Asking the AI model..."):
            try:
                llm_text = ask_llm_for_names(cluster_summary, api_key)
            except requests.exceptions.RequestException as error:
                st.error(f"The Ollama API request failed: {error}")
            except (KeyError, ValueError):
                st.error("The Ollama API returned an unexpected response.")

        if llm_text is not None:
            try:
                st.session_state["llm_table"] = parse_llm_csv(llm_text, cluster_table)
            except Exception as error:
                st.error(f"The AI response could not be converted to a cluster table: {error}")
                with st.expander("Show the AI response"):
                    st.text(llm_text)

if "llm_table" not in st.session_state:
    st.stop()

llm_table = st.session_state["llm_table"]
st.subheader("Final cluster table")
st.dataframe(llm_table, hide_index=True)

# ---------------------------------------------------------------------------
# STEP 7 — Create final CSV
# ---------------------------------------------------------------------------
st.header("Step 7 — Download the final CSV")

df_final = create_final_csv(df_original, df_clustered, llm_table)

# create_final_csv() drops cluster_id; put it back if it was an original column.
if "cluster_id" in df_original.columns:
    position = df_original.columns.get_loc("cluster_id")
    df_final.insert(position, "cluster_id", df_original["cluster_id"].values)

st.dataframe(df_final)

base_name = os.path.splitext(uploaded_file.name)[0]
st.download_button(
    label="Download CSV",
    data=df_final.to_csv(index=False).encode("utf-8"),
    file_name=f"{base_name}_clustered.csv",
    mime="text/csv",
)
