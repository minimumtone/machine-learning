#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
深層学習モデルモジュール
=============================================================================

【学習目標】
    - ニューラルネットワークの基本構造を理解する
    - PyTorchを用いた深層学習モデルの実装方法を習得する
    - 材料物性予測への深層学習の応用を学ぶ

【前提知識】
    - 機械学習の基本概念
    - 線形代数の基礎
    - Python/NumPyの基本操作

【対象】
    材料工学部 3回生

【深層学習とは】
    深層学習（Deep Learning）は、多層のニューラルネットワークを用いた
    機械学習手法です。特徴量の自動抽出が可能で、複雑なパターンを学習できます。

【材料工学での応用例】
    - 結晶構造からの物性予測
    - 材料画像からの欠陥検出
    - 分子構造からの特性予測

=============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# PyTorchのインポート（インストールされている場合）
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorchがインストールされていません。一部の機能が制限されます。")


# =============================================================================
# ニューラルネットワークモデル定義
# =============================================================================

if TORCH_AVAILABLE:
    class SimpleNN(nn.Module):
        """
        シンプルな全結合ニューラルネットワーク。

        材料物性予測などの回帰・分類タスクに使用できる
        基本的なフィードフォワードネットワークです。

        【構造】
        入力層 → 隠れ層1 → ReLU → 隠れ層2 → ReLU → 出力層

        Attributes:
            layers: ネットワーク層のリスト
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            dropout_rate: float = 0.2
        ):
            """
            Args:
                input_dim: 入力次元数
                hidden_dims: 各隠れ層のユニット数のリスト
                output_dim: 出力次元数
                dropout_rate: ドロップアウト率
            """
            super(SimpleNN, self).__init__()

            layers = []
            prev_dim = input_dim

            # 隠れ層の構築
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim

            # 出力層
            layers.append(nn.Linear(prev_dim, output_dim))

            self.layers = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """順伝播"""
            return self.layers(x)

    class MaterialPropertyPredictor(nn.Module):
        """
        材料物性予測用のニューラルネットワーク。

        バッチ正規化を含む、より高度なアーキテクチャです。
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [128, 64, 32],
            output_dim: int = 1
        ):
            super(MaterialPropertyPredictor, self).__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


# =============================================================================
# 学習・評価関数
# =============================================================================

def train_neural_network(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    hidden_dims: List[int] = [64, 32],
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    task: str = 'regression'
) -> Dict[str, Any]:
    """
    ニューラルネットワークを学習する。

    【学習の流れ】
    1. データをミニバッチに分割
    2. 順伝播で予測を計算
    3. 損失関数で誤差を計算
    4. 逆伝播で勾配を計算
    5. オプティマイザでパラメータを更新
    6. 1-5をエポック数だけ繰り返し

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        x_val: 検証データの特徴量（オプション）
        y_val: 検証データの目的変数（オプション）
        hidden_dims: 隠れ層のユニット数リスト
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        task: タスク種類 ('regression', 'classification')

    Returns:
        学習結果を含む辞書
    """
    if not TORCH_AVAILABLE:
        return {'error': 'PyTorchがインストールされていません'}

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データの準備
    x_tensor = torch.FloatTensor(x_train).to(device)
    if task == 'regression':
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    else:
        y_tensor = torch.LongTensor(y_train).to(device)

    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデルの構築
    input_dim = x_train.shape[1]
    if task == 'regression':
        output_dim = 1
        model = SimpleNN(input_dim, hidden_dims, output_dim).to(device)
        criterion = nn.MSELoss()
    else:
        output_dim = len(np.unique(y_train))
        model = SimpleNN(input_dim, hidden_dims, output_dim).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習履歴
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # 学習ループ
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)

            if task == 'regression':
                loss = criterion(outputs, batch_y)
            else:
                loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        history['train_loss'].append(avg_loss)

        # 検証
        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                x_val_tensor = torch.FloatTensor(x_val).to(device)
                if task == 'regression':
                    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
                else:
                    y_val_tensor = torch.LongTensor(y_val).to(device)

                val_outputs = model(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                history['val_loss'].append(val_loss)

    return {
        'model': model,
        'history': history,
        'device': device,
        'task': task
    }


def predict_neural_network(
    model: Any,
    x_data: np.ndarray,
    device: Any = None,
    task: str = 'regression'
) -> np.ndarray:
    """
    学習済みニューラルネットワークで予測する。

    Args:
        model: 学習済みモデル
        x_data: 入力データ
        device: 計算デバイス
        task: タスク種類

    Returns:
        予測結果
    """
    if not TORCH_AVAILABLE:
        return np.array([])

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_data).to(device)
        outputs = model(x_tensor)

        if task == 'regression':
            predictions = outputs.cpu().numpy().flatten()
        else:
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    return predictions


def evaluate_neural_network(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: Any = None,
    task: str = 'regression'
) -> Dict[str, float]:
    """
    ニューラルネットワークを評価する。

    Args:
        model: 学習済みモデル
        x_test: テストデータの特徴量
        y_test: テストデータの目的変数
        device: 計算デバイス
        task: タスク種類

    Returns:
        評価指標を含む辞書
    """
    predictions = predict_neural_network(model, x_test, device, task)

    if task == 'regression':
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }
    else:
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        return {
            'accuracy': accuracy,
            'f1': f1
        }


# =============================================================================
# 可視化関数
# =============================================================================

def plot_training_history(
    history: Dict[str, List],
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    学習履歴を可視化する。

    Args:
        history: 学習履歴
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')

    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    予測値と実測値を比較する散布図を作成する。

    Args:
        y_true: 実測値
        y_pred: 予測値
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='white', linewidth=0.5)

    # 理想線
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

    # R²スコアを表示
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(
        0.05, 0.95, f'R² = {r2:.4f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# scikit-learnベースの代替実装
# =============================================================================

def train_mlp_sklearn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    hidden_layer_sizes: Tuple[int, ...] = (100, 50),
    max_iter: int = 500,
    task: str = 'regression'
) -> Dict[str, Any]:
    """
    scikit-learnのMLPを使用してニューラルネットワークを学習する。

    PyTorchがインストールされていない環境でも使用可能な代替実装です。

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）
        hidden_layer_sizes: 隠れ層の構造
        max_iter: 最大イテレーション数
        task: タスク種類

    Returns:
        学習結果を含む辞書
    """
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if task == 'regression':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])

    model.fit(x_train, y_train)

    results = {
        'model': model,
        'train_predictions': model.predict(x_train)
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = model.predict(x_test)

        if task == 'regression':
            from sklearn.metrics import r2_score, mean_squared_error
            results['test_r2'] = r2_score(y_test, results['test_predictions'])
            results['test_rmse'] = np.sqrt(
                mean_squared_error(y_test, results['test_predictions'])
            )
        else:
            from sklearn.metrics import accuracy_score
            results['test_accuracy'] = accuracy_score(
                y_test, results['test_predictions']
            )

    return results


# =============================================================================
# メイン実行部（デモンストレーション）
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("深層学習モデルモジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータの作成
    print("\n【1. サンプルデータの作成】")
    print("-" * 50)

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    x_data, y_data = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    print(f"訓練データ: {x_train.shape[0]}サンプル")
    print(f"テストデータ: {x_test.shape[0]}サンプル")

    # scikit-learn MLPでの学習
    print("\n【2. scikit-learn MLPでの学習】")
    print("-" * 50)

    sklearn_result = train_mlp_sklearn(
        x_train, y_train, x_test, y_test,
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        task='regression'
    )

    print(f"テストR²: {sklearn_result['test_r2']:.4f}")
    print(f"テストRMSE: {sklearn_result['test_rmse']:.4f}")

    # PyTorchでの学習（利用可能な場合）
    if TORCH_AVAILABLE:
        print("\n【3. PyTorchでの学習】")
        print("-" * 50)

        x_train_sub, x_val, y_train_sub, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        pytorch_result = train_neural_network(
            x_train_sub, y_train_sub,
            x_val, y_val,
            hidden_dims=[64, 32],
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            task='regression'
        )

        eval_result = evaluate_neural_network(
            pytorch_result['model'],
            x_test, y_test,
            pytorch_result['device'],
            task='regression'
        )

        print(f"テストR²: {eval_result['r2']:.4f}")
        print(f"テストRMSE: {eval_result['rmse']:.4f}")

        # 可視化
        print("\n【4. 可視化】")
        print("-" * 50)

        fig1 = plot_training_history(
            pytorch_result['history'],
            title="Neural Network Training History"
        )
        plt.close(fig1)
        print("学習履歴プロット: 作成完了")

        predictions = predict_neural_network(
            pytorch_result['model'],
            x_test,
            pytorch_result['device'],
            task='regression'
        )

        fig2 = plot_predictions_vs_actual(
            y_test, predictions,
            title="Neural Network Predictions"
        )
        plt.close(fig2)
        print("予測結果プロット: 作成完了")
    else:
        print("\n※ PyTorchがインストールされていないため、PyTorchデモはスキップ")

    print("\n処理完了!")
