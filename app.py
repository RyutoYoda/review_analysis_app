import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import plotly.express as px
from snownlp import SnowNLP
import re

# アプリの設定
st.set_page_config(page_title="Review Analysis App", page_icon="📈")

# スタイル設定はそのまま

# 前処理関数やその他の関数はそのまま

# 必要なセッション状態の初期化
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'df' not in st.session_state:
    st.session_state.df = None

if 'num_clusters' not in st.session_state:
    st.session_state.num_clusters = 5

if 'fig' not in st.session_state:
    st.session_state.fig = None

if 'lof_fig' not in st.session_state:
    st.session_state.lof_fig = None

# ファイルのアップロード処理もそのまま

# クラスタリングと3次元プロット
if st.session_state.embeddings is not None:
    if st.button('クラスタリングと3次元プロットを実行'):
        # 既存のクラスタリングと3次元プロット処理

# 感情分析の実行もそのまま

# 新しい可視化: 外れ値検出と2次元可視化
if st.session_state.embeddings is not None:
    if st.button('外れ値検出と2次元可視化を実行'):
        try:
            sentence_embeddings = st.session_state.embeddings
            
            num_clusters = st.session_state.num_clusters  # クラスタ数はスライダーで選んだ値
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(sentence_embeddings)
            
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
            outliers = lof.fit_predict(sentence_embeddings)
            outlier_scores = lof.negative_outlier_factor_
            
            # PCAで2次元に縮約
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(sentence_embeddings)
            
            marker_size = np.clip(np.abs(outlier_scores) * 5, 5, 20)
            hover_text = st.session_state.df[review_column].tolist()
            
            fig = px.scatter(
                x=reduced_embeddings[:, 0], 
                y=reduced_embeddings[:, 1], 
                color=labels.astype(str),
                size=marker_size,
                title=f"クラスタリング結果と外れ値の可視化（クラスタ数: {num_clusters}）",
                labels={'color': 'Cluster'},
                hover_name=hover_text
            )
            
            outlier_points = reduced_embeddings[outliers == -1]
            fig.add_scatter(x=outlier_points[:, 0], y=outlier_points[:, 1], mode='markers', marker=dict(color='red', size=10), name='Outliers')
            st.session_state.lof_fig = fig  # 新しいプロットをセッション状態に保存
            st.plotly_chart(st.session_state.lof_fig, use_container_width=True)
            
            # 結果をCSVに保存
            outliers_indices = np.where(outliers == -1)[0]
            outliers_keywords = [st.session_state.df[review_column].iloc[i] for i in outliers_indices]
            outliers_df = pd.DataFrame(outliers_keywords, columns=['Outlier Keywords'])
            st.write("外れ値のキーワード：", outliers_df)
            
            csv_outliers = outliers_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="外れ値データをCSVとしてダウンロード",
                data=csv_outliers,
                file_name='outliers_keywords.csv',
                mime='text/csv'
            )
        
        except Exception as e:
            st.error("外れ値検出と可視化中にエラーが発生しました。")
            st.error(str(e))
