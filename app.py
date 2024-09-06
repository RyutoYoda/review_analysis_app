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

# スタイル設定
st.markdown("""
<style>
body {
    font-family: 'Helvetica Neue', sans-serif;
    background-color: #1e1e1e;
    color: #ffffff;
}
header, footer {
    visibility: hidden;
}
.main {
    background-color: #1e1e1e;
}
.big-font {
    font-size: 36px !important;
    font-weight: bold;
    color: #61dafb;
}
.label-font {
    font-size: 20px !important;
    color: #61dafb;
}
.stButton>button {
    background-color: #61dafb !important;
    color: #ffffff !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}
.stTextInput>div>div>input {
    background-color: #333333 !important;
    color: #ffffff !important;
    border: 1px solid #61dafb !important;
}
.stSelectbox>div>div>div {
    background-color: #333333 !important;
    color: #ffffff !important;
    border: 1px solid #61dafb !important;
}
.stSlider>div>div>div>div {
    background-color: #61dafb !重要です;
}
</style>
""", unsafe_allow_html=True)

# Streamlitアプリのタイトル
st.markdown('<div class="big-font">Review Analysis App📈</div>', unsafe_allow_html=True)

# トグルで説明と使い方を表示
with st.expander("アプリケーションの説明と使い方"):
    st.markdown("""
    ### アプリケーションの説明
    このアプリケーションは、口コミデータを分析し、クラスタリングと感情分析を行うツールです。以下の機能を提供します：
    - 口コミデータの埋め込みベクトル生成
    - クラスタリング
    - 3次元プロットによる可視化
    - 感情分析
    - 外れ値検出とクラスタリングの2次元可視化

    ### 使い方
    1. 口コミデータが含まれるCSVまたはExcelファイルをアップロードします。
    2. 口コミが含まれている列を選択します。
    3. 「埋め込みベクトルを生成」ボタンをクリックします。
    4. クラスタ数を選択します。
    5. 「クラスタリングと3次元プロットを実行」ボタンをクリックします。
    6. 「感情分析を実行」ボタンをクリックします。
    7. 「外れ値検出と2次元可視化を実行」ボタンをクリックします。
    8. 必要に応じて、「データをCSVとしてダウンロード」ボタンをクリックして、結果をダウンロードします。
    """)

# 前処理関数や必要な変数はそのまま残します

# 埋め込みベクトル生成やクラスタリングなどの処理は既存コード

# 新しい可視化の追加
if st.session_state.embeddings is not None:
    if st.button('外れ値検出と2次元可視化を実行'):
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
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
            st.plotly_chart(fig, use_container_width=True)
            
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

