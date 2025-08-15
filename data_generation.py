"""
データ生成スクリプト
シンボリック回帰用の物理法則データを生成し、CSVファイルとして保存
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_kinetic_energy_data(n_samples=100):
    """運動エネルギーのデータを生成"""
    m = np.random.uniform(1, 10, n_samples)
    v = np.random.uniform(1, 20, n_samples)
    K = 0.5 * m * v**2 + np.random.normal(0, 0.1, n_samples)  # ノイズを加える
    return pd.DataFrame({'m': m, 'v': v, 'K': K})

def generate_pendulum_data(n_samples=100):
    """単振り子の周期データを生成"""
    L = np.random.uniform(0.5, 5, n_samples)
    m = np.random.uniform(0.1, 2, n_samples)  # 無関係な変数
    g = np.random.uniform(9.8, 10.2, n_samples)
    T = 2 * np.pi * np.sqrt(L / g) + np.random.normal(0, 0.01, n_samples)
    return pd.DataFrame({'L': L, 'm': m, 'g': g, 'T': T})

def generate_gravity_data(n_samples=100):
    """万有引力のデータを生成"""
    G = 6.674e-11
    m1 = 1e10 * np.random.uniform(1, 10, n_samples)
    m2 = 1e10 * np.random.uniform(1, 10, n_samples)
    r = np.random.uniform(100, 1000, n_samples)
    F = G * (m1 * m2) / r**2 + np.random.normal(0, 1e-5, n_samples)
    return pd.DataFrame({'m1': m1, 'm2': m2, 'r': r, 'F': F})

def main():
    """メイン関数：全データを生成してCSVファイルに保存"""
    print("物理法則データを生成中...")
    
    kinetic_data = generate_kinetic_energy_data()
    pendulum_data = generate_pendulum_data()
    gravity_data = generate_gravity_data()
    
    kinetic_data.to_csv('kinetic_energy.csv', index=False)
    pendulum_data.to_csv('pendulum.csv', index=False)
    gravity_data.to_csv('gravity.csv', index=False)
    
    print("✅ データ生成完了！")
    print("生成されたファイル:")
    print("- kinetic_energy.csv (運動エネルギー)")
    print("- pendulum.csv (単振り子の周期)")
    print("- gravity.csv (万有引力)")
    
    print("\n📊 データ概要:")
    print("\n1. 運動エネルギーデータ:")
    print(kinetic_data.describe())
    
    print("\n2. 単振り子の周期データ:")
    print(pendulum_data.describe())
    
    print("\n3. 万有引力データ:")
    print(gravity_data.describe())

if __name__ == "__main__":
    main()
