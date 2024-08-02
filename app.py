import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# アプリの設定
st.set_page_config(page_title="Review Analysis App", page_icon="📊")

# スタイル設定
st.markdown("""
<style>
body {
    font-family: 'Helvetica Neue', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Streamlitアプリのタイトル
st.title('Review Analysis App📊')

# ファイルアップロード
uploaded_file = st.file_uploader("ファイルをアップロードしてください", type=["csv", "xlsx"])

if uploaded_file:
    # ファイルをデータフレームとして読み込む
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("データプレビュー：", df.head())
    
    # 口コミが含まれている列を選択
    review_column = st.selectbox("口コミが含まれている列を選択してください", df.columns)

    # 分析開始ボタン
    if st.button('分析開始'):
        try:
            # 口コミをベクトル化
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode(df[review_column].tolist())
            
            # クラスタリングを実行
            num_clusters = st.slider("クラスタ数を選択してください", 2, 10, 5)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(embeddings)
            
            # PCAを使用して3次元に可視化
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(embeddings)
            df['pca_one'] = pca_result[:, 0]
            df['pca_two'] = pca_result[:, 1]
            df['pca_three'] = pca_result[:, 2]
            
            # クラスタの色を指定
            color_sequence = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            fig = px.scatter_3d(
                df, x='pca_one', y='pca_two', z='pca_three',
                color='cluster', hover_data=[review_column],
                color_discrete_sequence=color_sequence[:num_clusters]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment Analysis
            analyzer = SentimentIntensityAnalyzer()
            df['sentiment_score'] = df[review_column].apply(lambda x: analyzer.polarity_scores(x)['compound'] * 5)
            df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else 'negative')
            
            st.write("Sentiment Analysis結果：")
            st.write(df[[review_column, 'sentiment', 'sentiment_score']])
            
            # ベクトル数値を列として追加
            for i in range(embeddings.shape[1]):
                df[f'vector_{i}'] = embeddings[:, i]
            
            # 頻出単語ランキング
            word_list = ' '.join(df[review_column].tolist()).split()
            word_freq = Counter(word_list)
            most_common_words = word_freq.most_common(20)
            words, counts = zip(*most_common_words)
            
            fig = px.bar(x=words, y=counts, labels={'x': '単語', 'y': '出現回数'}, title="頻出単語ランキング")
            st.plotly_chart(fig, use_container_width=True)
            
            # データをダウンロードするためのリンクを作成
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(df)
            st.download_button(
                label="データをCSVとしてダウンロード",
                data=csv,
                file_name='review_analysis_result.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error("選択いただいた列は分析不可です。")
            st.error(str(e))
