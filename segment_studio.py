import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.cluster import KMeans
from io import StringIO

def load_data(path):
    """Load a CSV file.

    :param path: Path to the CSV file.
    :return: A DataFrame containing the file data.
    """
    df = pd.read_csv(path)
    return df


def count_nan (df):
    """Count missing values in each column.

    :param df: The DataFrame to check.
    :return: A Series with the number of missing values per column.
    """
    num_nan=df.isnull().sum()
    return num_nan

def fill_missing_values(df):
    """Fill missing numeric and categorical values.

    :param df: The DataFrame with possible missing values.
    :return: A copied DataFrame with missing values filled.
    """
    df = df.copy()

    # Separate numeric columns from categorical columns.
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    categorical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df



def scaling(df):
    """Standardize the numeric columns in a DataFrame.

    :param df: The DataFrame to scale.
    :return: A copied DataFrame with standardized numeric columns.
    """
    df_scaled = df.copy()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df_scaled

def encoding(df):
    """Convert categorical columns into dummy variables.

    :param df: The DataFrame to encode.
    :return: A DataFrame containing numeric dummy variables.
    """
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded

def wcss_calc(df,min_k, max_k):
    """Calculate WCSS for a range of cluster counts.

    :param df: The prepared data used for clustering.
    :param min_k: The smallest number of clusters to test.
    :param max_k: The largest number of clusters to test.
    :return: A DataFrame containing each K value and its WCSS.
    """

    wcss=[]
    for k in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df)
        # .inertia_ gives the WCSS for that model
        wcss.append(kmeans.inertia_)
    wcss_table = pd.DataFrame({
        'K': range(min_k, max_k + 1),
        'WCSS': wcss
    })
    return wcss_table

def elbow_graph(wcss_table):
    """Display an elbow graph of K values and WCSS scores.

    :param wcss_table: A DataFrame with K and WCSS columns.
    :return: None.
    """

    # Create a line graph that shows how WCSS changes.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(wcss_table['K'] , wcss_table['WCSS'], 'bx-', linewidth=2, markersize=8)
    plt.title('The Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
    plt.xticks(wcss_table['K'])
    plt.show()
def clustering(k, df):
    """Assign each row to a KMeans cluster.

    :param k: The number of clusters to create.
    :param df: The prepared data used for clustering.
    :return: A copied DataFrame with a cluster_id column.
    """
    df_clustered = df.copy()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_clustered['cluster_id'] = kmeans.fit_predict(df_clustered)

    return df_clustered

def create_cluster_table(df_clustered):
    """Create a summary table with the size of each cluster.

    :param df_clustered: A DataFrame containing cluster IDs.
    :return: A cluster summary DataFrame with empty name and description columns.
    """
    cluster_table = df_clustered['cluster_id'].value_counts().reset_index()
    cluster_table.columns = ['cluster_id', 'count']

    cluster_table['name'] = ''
    cluster_table['description'] = ''

    return cluster_table

def add_numeric_means(cluster_table, df, df_clustered):
    """Add the mean of each numeric column for every cluster.

    :param cluster_table: The cluster summary table to update.
    :param df: The original DataFrame containing numeric values.
    :param df_clustered: The DataFrame containing cluster IDs.
    :return: The cluster table with numeric mean columns added.
    """
    df = df.copy()
    df['cluster_id'] = df_clustered['cluster_id'].values

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for column in numeric_columns:
        if column != 'cluster_id':
            means = df.groupby('cluster_id')[column].mean()
            cluster_table[f'mean_{column}'] = cluster_table['cluster_id'].map(means)

    return cluster_table

def add_categorical_modes(cluster_table, df, df_clustered):
    """Add the most common categorical value for every cluster.

    :param cluster_table: The cluster summary table to update.
    :param df: The original DataFrame containing categorical values.
    :param df_clustered: The DataFrame containing cluster IDs.
    :return: The cluster table with most-common-value columns added.
    """
    df = df.copy()

    categorical_columns = df.select_dtypes(exclude='number').columns

    df['cluster_id'] = df_clustered['cluster_id'].values

    for column in categorical_columns:
        modes = df.groupby('cluster_id')[column].agg(lambda x: x.mode()[0])
        cluster_table[f'most_common_{column}'] = cluster_table['cluster_id'].map(modes)

    return cluster_table

def create_final_csv(df, df_clustered, llm_table):
    """Add cluster names to the original data.

    :param df: The original DataFrame.
    :param df_clustered: The DataFrame containing cluster IDs.
    :param llm_table: A table that maps cluster IDs to cluster names.
    :return: A DataFrame containing a name_cluster column.
    """
    df_final = df.copy()

    df_final['cluster_id'] = df_clustered['cluster_id'].values

    name_map = llm_table.set_index('cluster_id')['name']

    df_final['name_cluster'] = df_final['cluster_id'].map(name_map)

    df_final = df_final.drop(columns='cluster_id')

    return df_final


import requests

def save_to_csv(df, path):
    """Save a DataFrame as a CSV file without row indexes.

    :param df: The DataFrame to save.
    :param path: The path of the output CSV file.
    :return: None.
    """
    df.to_csv(path, index=False)
# Load and prepare the data for clustering.
if __name__ == "__main__":
    df_original = load_data('C:/Users/User/train.csv')
    df = fill_missing_values(df_original)
    df_scaled = scaling(df)
    df_encoded = encoding(df_scaled)
    wcss_table=wcss_calc(df_encoded, 2, 10)

    elbow_graph(wcss_table)
    df_clustered=clustering(5, df_encoded)
    cluster_table=create_cluster_table(df_clustered)


    # Add useful values that describe each cluster.
    cluster_table = add_numeric_means(cluster_table, df, df_clustered)
    cluster_table = add_categorical_modes(cluster_table, df, df_clustered)

    print(cluster_table)
    print(cluster_table.columns)


    print(df.dtypes)
    print(df.select_dtypes(exclude=['float64', 'int64']).columns)

    API_KEY = "cbf37916ef84f6da011dc5340.SLa1xTamf3ecLz"  # put your key here instead of ollama_your_key_here

    # Ask the model to create a name and description for each cluster.
    response = requests.post(
        "https://ollama.com/api/chat",
        headers={
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "model": "gpt-oss:120b",
            "messages": [
                {
                    "role": "user",
                    "content": f"""
                    Analyze the following cluster summary table:

                    {cluster_table.to_string()}

                    Use the numeric averages and most common categorical values
                    to fill the name and description for each cluster.

                    Return only CSV with these columns:
                    cluster_id,count,name,description

                    Do not add any explanation before or after the CSV.
                    """

                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }
    )

    data = response.json()

    llm_result = data["message"]["content"]


    llm_table = pd.read_csv(StringIO(llm_result))
    print(llm_table)

    # Add the generated cluster names and save the final data.
    df_final = create_final_csv(df_original, df_clustered, llm_table)

    print(df_final.head())

    save_to_csv(df_final, 'C:/Users/User/train_clustered.csv')
