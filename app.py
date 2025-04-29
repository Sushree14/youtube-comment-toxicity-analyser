import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Set the environment to use polling for file watching (Fixing issue with PyTorch)
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Function to extract video ID from URL
def extract_video_id(url):
    video_id = url.split("v=")[-1]
    if "&" in video_id:
        video_id = video_id.split("&")[0]
    return video_id

# Function to get comments from YouTube
def get_video_comments(video_url, api_key):
    video_id = extract_video_id(video_url)
    base_url = "https://youtube.googleapis.com/youtube/v3/commentThreads"
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': 50,
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    comments = []
    if response.status_code == 200:
        data = response.json()
        for item in data.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
    return comments

# Function to plot Toxic vs Non-Toxic Distribution
def plot_toxicity_distribution(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='label', data=df, palette='Blues', edgecolor='black')
    ax.set_title('Toxic vs Non-Toxic Comments', fontsize=16)
    ax.set_xlabel('Toxicity Label', fontsize=12)
    ax.set_ylabel('Number of Comments', fontsize=12)
    ax.set_xticklabels(['Non-Toxic', 'Toxic'], fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot Toxicity Scores Distribution using a Line Graph
def plot_toxicity_scores_line_graph(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['score'], marker='o', linestyle='-', color='purple', label='Toxicity Score')
    plt.title('Toxicity Scores Distribution (Line Graph)', fontsize=16)
    plt.xlabel('Comment Index', fontsize=12)
    plt.ylabel('Toxicity Score', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit App UI
def main():
    st.title('AI-powered YouTube Comment Toxicity Analyzer')
    st.sidebar.header('YouTube Video Toxicity Analysis')

    api_key = st.sidebar.text_input("Enter Your YouTube API Key", type="password")

    youtube_url = st.text_input("Enter YouTube Video URL")
    if st.button("Analyze Comments"):
        if api_key:
            # Fetch comments
            comments = get_video_comments(youtube_url, api_key)
            
            # Here you should add your logic to classify comments as toxic or non-toxic using the pre-trained model
            toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")
            toxicity_results = []
            for comment in comments:
                result = toxicity_model(comment)
                label = result[0]['label']
                score = result[0]['score']  # Add the score
                toxicity_results.append({'label': label, 'score': score})  # Append both label and score

            # Create DataFrame with both label and score
            df = pd.DataFrame(toxicity_results)
            df['comment'] = comments  # Add the comment column to the DataFrame

            # Display results
            st.write("Comments and Toxicity Analysis:")
            st.dataframe(df)

            # Plot graphs
            plot_toxicity_distribution(df)
            plot_toxicity_scores_line_graph(df)

        else:
            st.error("Please provide a valid API key!")

if __name__ == '__main__':
    main()
