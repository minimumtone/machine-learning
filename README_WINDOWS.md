# Windows環境での実行ガイド

このガイドは、`diffusion_couple_pinns.py` をWindows環境で実行するための手順を説明します。

## 必要な環境

- Windows 10 / 11
- Python 3.8以上
- pip（Pythonパッケージマネージャー）

## セットアップ手順

### 1. Pythonのインストール

まだPythonがインストールされていない場合は、以下からダウンロードしてインストールしてください：

https://www.python.org/downloads/

**重要**: インストール時に "Add Python to PATH" にチェックを入れてください。

### 2. インストール確認

コマンドプロンプトまたはPowerShellを開き、以下のコマンドでPythonが正しくインストールされているか確認します：

```cmd
python --version
```

または

```cmd
python3 --version
```

### 3. 必要なパッケージのインストール

#### 方法A: requirements.txtを使用（推奨）

プロジェクトディレクトリに移動し、以下を実行：

```cmd
cd path\to\machine-learning
pip install -r requirements_diffusion_pinns.txt
```

#### 方法B: 個別にインストール

```cmd
pip install numpy torch matplotlib tqdm
```

Streamlitを使用する場合は追加でインストール：

```cmd
pip install streamlit pandas plotly
```

### 4. プログラムの実行

#### 基本実行（デモンストレーション）

```cmd
python diffusion_couple_pinns.py
```

実行すると：
- FDMによる拡散対データが生成されます
- 純物質境界条件の概念図が表示されます
- カレントディレクトリに `diffusion_couple_demo.png` が保存されます

#### Streamlitモード（未実装）

```cmd
streamlit run darken_pinns_app.py
```

## トラブルシューティング

### matplotlibの表示エラー

もしmatplotlibの表示でエラーが発生する場合は、以下を試してください：

1. スクリプトの先頭で別のバックエンドを使用：

```python
import matplotlib
matplotlib.use('Agg')  # GUIなし
```

2. または、Anaconda環境を使用する場合：

```cmd
conda install matplotlib
```

### PyTorchのインストールエラー

PyTorchのインストールに問題がある場合は、公式サイトから適切なバージョンを選択してください：

https://pytorch.org/get-started/locally/

Windows + CPU版の場合：

```cmd
pip install torch torchvision torchaudio
```

### パス関連のエラー

Windows環境ではファイルパスに注意が必要です。スクリプトは自動的に現在のディレクトリに出力ファイルを保存します。

特定のディレクトリに保存したい場合は、スクリプト内の保存パスを修正してください：

```python
save_path = r"C:\Users\YourName\Documents\diffusion_couple_demo.png"
```

## プログラムの出力

実行が成功すると、以下が生成されます：

1. **コンソール出力**: FDMデータ生成の情報と純物質境界条件の説明
2. **グラフ表示**: 2つのサブプロット
   - (a) FDMによる拡散対の濃度プロファイル
   - (b) 純物質境界条件の概念図
3. **画像ファイル**: `diffusion_couple_demo.png` がカレントディレクトリに保存

## 詳細な技術情報

プログラムの詳細な説明については、`DIFFUSION_COUPLE_README.md` を参照してください。

## GPUの使用

PyTorchがCUDA対応のGPUを検出すると、自動的にGPUを使用します。GPUの状態は実行時にコンソールに表示されます：

```
GPU is available. Using device: cuda (NVIDIA GeForce RTX 3080)
```

CPUのみで実行する場合：

```
GPU not available. Using device: CPU
```

## サポート

問題が発生した場合は、GitHubのissueを作成してください。
