import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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

    ### 使い方
    1. 口コミデータが含まれるCSVまたはExcelファイルをアップロードします。
    2. 口コミが含まれている列を選択します。
    3. 「埋め込みベクトルを生成」ボタンをクリックします。
    4. クラスタ数を選択します。
    5. 「クラスタリングと3次元プロットを実行」ボタンをクリックします。
    6. 「感情分析を実行」ボタンをクリックします。
    7. 必要に応じて、「データをCSVとしてダウンロード」ボタンをクリックして、結果をダウンロードします。
    """)

# テキストの前処理関数
def preprocess_text(text):
    text = text.lower()  # 小文字に変換
    text = re.sub(r'\d+', '', text)  # 数字を削除
    text = re.sub(r'\s+', ' ', text)  # 不要な空白を削除
    text = re.sub(r'[^\w\sぁ-んァ-ン一-龥]', '', text)  # 特殊文字を削除、漢字、ひらがな、カタカナを含む
    return text

# ネガティブワードとポジティブワードのリストを定義
NEGATIVE_WORDS = ["最悪", "ひどい", "不満", "失敗", "問題","できません","悪く","つらい","悲しい","悪い"]
POSITIVE_WORDS = ["最高", "素晴らしい", "満足", "成功", "良い","入","長持ち","便利","勧めたい","嬉しい","楽しい","問題ありません","問題ない","友人にも"]

# 単語ベースで感情分析を強化する関数
def enhanced_sentiment_analysis(text):
    score = 0
    for word in NEGATIVE_WORDS:
        if word in text:
            score -= 1
    for word in POSITIVE_WORDS:
        if word in text:
            score += 1
    snow_score = SnowNLP(text).sentiments
    combined_score = (snow_score * 2) - 1 + (score / max(len(NEGATIVE_WORDS), len(POSITIVE_WORDS)))  # 正規化とスケーリング
    scaled_score = np.clip(combined_score, -1, 1)  # スコアを-1から1の範囲にクリッピング
    return 'positive' if scaled_score > 0 else 'negative', scaled_score

# ファイルアップロード
uploaded_file = st.file_uploader("ファイルをアップロードしてください", type=["csv", "xlsx"], label_visibility='visible', key="fileUploader")

# セッション状態を初期化
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'df' not in st.session_state:
    st.session_state.df = None

if 'num_clusters' not in st.session_state:
    st.session_state.num_clusters = 5

if 'fig' not in st.session_state:
    st.session_state.fig = None

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

    # テキストの前処理
    st.session_state.df[review_column] = st.session_state.df[review_column].apply(preprocess_text)

    # 埋め込みベクトル生成ボタン
    if st.button('埋め込みベクトルを生成'):
        try:
            with st.spinner('埋め込みベクトルを生成中...'):
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.embeddings = model.encode(st.session_state.df[review_column].astype(str).tolist())
            
            st.success('埋め込みベクトルの生成が完了しました！')
        
        except Exception as e:
            st.error("埋め込みベクトルの生成に失敗しました。")
            st.error(str(e))

    # クラスタリング数の選択
    st.session_state.num_clusters = st.slider("クラスタ数を選択してください", 2, 10, 5, label_visibility='visible')
    
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
                st.session_state.fig = px.scatter_3d(
                    st.session_state.df, x='pca_one', y='pca_two', z='pca_three',
                    color='cluster', hover_data=[review_column],
                    color_discrete_sequence=color_sequence[:st.session_state.num_clusters]
                )
                st.plotly_chart(st.session_state.fig, use_container_width=True)
            
            except Exception as e:
                st.error("クラスタリングとプロットに失敗しました。")
                st.error(str(e))
    
    # 感情分析ボタン
    if st.session_state.embeddings is not None:
        if st.button('感情分析を実行'):
            try:
                st.session_state.df['sentiment'], st.session_state.df['sentiment_score'] = zip(*st.session_state.df[review_column].astype(str).apply(enhanced_sentiment_analysis))
                
                st.write("Sentiment Analysis結果：")
                st.write(st.session_state.df[[review_column, 'sentiment', 'sentiment_score']])
                
                # ベクトル数値を列として追加
                for i in range(st.session_state.embeddings.shape[1]):
                    st.session_state.df[f'vector_{i}'] = st.session_state.embeddings[:, i]
                
                # PCAを使用して3次元に可視化（再計算）
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(st.session_state.embeddings)
                st.session_state.df['pca_one'] = pca_result[:, 0]
                st.session_state.df['pca_two'] = pca_result[:, 1]
                st.session_state.df['pca_three'] = pca_result[:, 2]
                
                # プロットを再作成（ポジティブを赤、ネガティブを青）
                color_map = {'positive': 'red', 'negative': 'blue'}
                st.session_state.fig = px.scatter_3d(
                    st.session_state.df, x='pca_one', y='pca_two', z='pca_three',
                    color='sentiment', hover_data=[review_column],
                    color_discrete_map=color_map
                )
                st.plotly_chart(st.session_state.fig, use_container_width=True)
            
            except Exception as e:
                st.error("感情分析中にエラーが発生しました。")
                st.error(str(e))
    
    # データをダウンロードするためのリンクを作成
    if st.session_state.embeddings is not None and st.session_state.df is not None:
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

