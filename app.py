import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

# セッション状態を初期化
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    # ファイルをデータフレームとして読み込む
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("データプレビュー：", df.head())
    
    # 口コミが含まれている列を選択
    review_column = st.selectbox("口コミが含まれている列を選択してください", df.columns)
    
    # レビュー列以外を切り落とし、レビューIDを追加
    st.session_state.df = df[[review_column]].dropna()
    st.session_state.df['review_id'] = st.session_state.df.index

    # 埋め込みベクトル生成ボタン
    if st.button('埋め込みベクトルを生成'):
        try:
            with st.spinner('埋め込みベクトルを生成中...'):
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.embeddings = model.encode(st.session_state.df[review_column].astype(str).tolist())
            
            st.success('埋め込みベクトルの生成が完了しました！')
            st.session_state.num_clusters = st.slider("クラスタ数を選択してください", 2, 10, 5)
        
        except Exception as e:
            st.error("埋め込みベクトルの生成に失敗しました。")
            st.error(str(e))
    
    # クラスタリングと3次元プロットボタン
    if st.session_state.embeddings is not None:
        if st.button('クラスタリングと3次元プロットを実行'):
            try:
                # クラスタリングを実行
                kmeans = KMeans(n_clusters=st.session_state.num_clusters, random_state=42)
                st.session_state.df['cluster'] = kmeans.fit_predict(st.session_state.embeddings)
                
                # PCAを使用して3次元に可視化
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(st.session_state.embeddings)
                st.session_state.df['pca_one'] = pca_result[:, 0]
                st.session_state.df['pca_two'] = pca_result[:, 1]
                st.session_state.df['pca_three'] = pca_result[:, 2]
                
                # クラスタの色を指定
                color_sequence = px.colors.qualitative.T10
                fig = px.scatter_3d(
                    st.session_state.df, x='pca_one', y='pca_two', z='pca_three',
                    color='cluster', hover_data=[review_column],
                    color_discrete_sequence=color_sequence[:st.session_state.num_clusters]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error("クラスタリングとプロットに失敗しました。")
                st.error(str(e))
    
    # 感情分析ボタン
    if st.session_state.embeddings is not None and st.button('感情分析を実行'):
        try:
            analyzer = SentimentIntensityAnalyzer()
            st.session_state.df['sentiment_score'] = st.session_state.df[review_column].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'] * 5)
            st.session_state.df['sentiment'] = st.session_state.df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else 'negative')
            
            st.write("Sentiment Analysis結果：")
            st.write(st.session_state.df[[review_column, 'sentiment', 'sentiment_score']])
            
            # ベクトル数値を列として追加
            for i in range(st.session_state.embeddings.shape[1]):
                st.session_state.df[f'vector_{i}'] = st.session_state.embeddings[:, i]
            
            # 追加した列を表示
            st.write("更新されたデータフレーム：")
            st.write(st.session_state.df)
        
        except Exception as e:
            st.error("感情分析中にエラーが発生しました。")
            st.error(str(e))

    # データをダウンロードするためのリンクを作成
    if st.session_state.embeddings is not None and st.button('データをCSVとしてダウンロード'):
        try:
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(st.session_state.df)
            st.download_button(
                label="データをCSVとしてダウンロード",
                data=csv,
                file_name='review_analysis_result.csv',
                mime='text/csv',
            )
        
        except Exception as e:
            st.error("データのダウンロード中にエラーが発生しました。")
            st.error(str(e))
