# 変分法とDFT教育アプリ - セットアップガイド

## 🚀 クイックスタート (既存環境)

現在の環境では既にファイルが存在しているので、すぐに起動できます:

```bash
cd /home/ubuntu/repos/machine-learning
streamlit run variational_dft_app.py
```

## 📦 新しい環境でのセットアップ

### ステップ1: リポジトリのクローン

```bash
# ホームディレクトリに移動
cd ~

# GitHubからリポジトリをクローン
git clone https://github.com/minimumtone/machine-learning.git

# クローンしたディレクトリに移動
cd machine-learning
```

### ステップ2: Python環境の確認

```bash
# Pythonのバージョン確認 (3.8以上が必要)
python3 --version
```

Python 3.8未満の場合は、アップデートが必要です。

### ステップ3: 依存パッケージのインストール

```bash
# requirements.txtから一括インストール
pip install -r requirements.txt
```

**または**個別にインストール:

```bash
pip install streamlit numpy matplotlib scipy pandas plotly
```

### ステップ4: アプリの起動

```bash
# Streamlitアプリを起動
streamlit run variational_dft_app.py
```

起動に成功すると、以下のようなメッセージが表示されます:

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### ステップ5: ブラウザでアクセス

自動的にブラウザが開きますが、開かない場合は手動で以下のURLにアクセス:

```
http://localhost:8501
```

## 🔧 詳細なオプション

### カスタムポートで起動

```bash
streamlit run variational_dft_app.py --server.port 8502
```

### ヘッドレスモード (サーバー環境)

```bash
streamlit run variational_dft_app.py --server.headless true
```

### デバッグモード

```bash
streamlit run variational_dft_app.py --logger.level=debug
```

## 🐛 トラブルシューティング

### 問題1: `streamlit: command not found`

**原因**: Streamlitがインストールされていない

**解決策**:
```bash
pip install streamlit
```

### 問題2: `ModuleNotFoundError: No module named 'numpy'`

**原因**: 必要なパッケージがインストールされていない

**解決策**:
```bash
pip install -r requirements.txt
```

### 問題3: ポートが既に使用されている

**原因**: 8501ポートが他のアプリで使用中

**解決策**: 別のポートを指定
```bash
streamlit run variational_dft_app.py --server.port 8502
```

### 問題4: 日本語が文字化けする

**原因**: 日本語フォントがシステムにインストールされていない

**解決策** (Ubuntu/Debian):
```bash
sudo apt-get install fonts-noto-cjk
```

**解決策** (macOS):
通常は標準でインストールされています。

### 問題5: Matplotlibの警告が出る

**原因**: バックエンドの設定問題（通常は無視して問題ありません）

**解決策**: 環境変数を設定
```bash
export MPLBACKEND=Agg
streamlit run variational_dft_app.py
```

## 📚 ファイル構成

```
machine-learning/
├── variational_dft_app.py      # メインアプリケーション
├── README_variational_dft.md   # アプリの詳細説明
├── SETUP_GUIDE.md              # このファイル
├── requirements.txt            # 依存パッケージリスト
└── ...                         # その他のファイル
```

## 🎓 使い方

アプリが起動したら:

1. **タブを選択**: 画面上部の5つのタブから学習したいトピックを選択
2. **パラメータ調整**: スライダーやドロップダウンでパラメータを変更
3. **結果を確認**: グラフと計算結果がリアルタイムで更新されます

推奨学習順序:
1. 📚 変分法の基礎
2. 🎯 変分原理の実演
3. ⚛️ DFT基礎
4. 🔬 簡易DFT計算
5. 📊 エネルギー最小化

## 💻 動作環境

- **OS**: Windows, macOS, Linux
- **Python**: 3.8以上
- **ブラウザ**: Chrome, Firefox, Safari, Edge (最新版推奨)
- **メモリ**: 最低2GB推奨

## 📝 アップデート方法

リポジトリの最新版を取得:

```bash
cd ~/machine-learning
git pull origin main
```

依存パッケージの更新:

```bash
pip install -r requirements.txt --upgrade
```

## 🆘 サポート

問題が解決しない場合は、以下の情報を添えてお問い合わせください:

- Pythonのバージョン (`python3 --version`)
- OSの種類とバージョン
- エラーメッセージの全文
- 実行したコマンド

---

**開発**: 機械学習・材料科学教育プロジェクト
