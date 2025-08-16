"""
材料工学データ生成スクリプト
材料科学の物理法則に基づくデータセットを生成
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_thermal_conductivity_data(n_samples=100):
    """
    熱伝導率データを生成 (Wiedemann-Franz法則)
    κ = L₀ × σ × T
    """
    sigma = np.random.uniform(1e6, 1e8, n_samples)  # 電気伝導率 [S/m]
    T = np.random.uniform(200, 400, n_samples)      # 温度 [K]
    L0 = 2.44e-8  # ローレンツ数 [W·Ω/K²]
    
    kappa = L0 * sigma * T + np.random.normal(0, 0.1, n_samples)
    
    return pd.DataFrame({
        'sigma': sigma,
        'T': T,
        'kappa': kappa
    })

def generate_hall_effect_data(n_samples=100):
    """
    ホール効果データを生成
    R_H = 1/(n × e)
    """
    n = np.random.uniform(1e20, 1e24, n_samples)  # キャリア密度 [m⁻³]
    e = 1.602e-19  # 電子電荷 [C]
    
    R_H = 1 / (n * e) + np.random.normal(0, 1e-8, n_samples)
    
    return pd.DataFrame({
        'n': n,
        'R_H': R_H
    })

def generate_hall_petch_data(n_samples=100):
    """
    Hall-Petch関係データを生成
    σ_y = σ₀ + k/√d
    """
    d = np.random.uniform(1e-6, 1e-4, n_samples)  # 結晶粒径 [m]
    sigma_0 = 50e6  # 基準応力 [Pa]
    k = 0.5e-3      # Hall-Petch定数
    
    sigma_y = sigma_0 + k / np.sqrt(d) + np.random.normal(0, 1e6, n_samples)
    
    return pd.DataFrame({
        'd': d,
        'sigma_y': sigma_y
    })

def generate_diffusion_data(n_samples=100):
    """
    拡散係数データを生成 (Arrhenius式)
    D = D₀ × exp(-Q/(R×T))
    """
    T = np.random.uniform(800, 1200, n_samples)  # 温度 [K]
    D0 = 1e-4  # 頻度因子 [m²/s]
    Q = 200000  # 活性化エネルギー [J/mol]
    R = 8.314   # 気体定数 [J/(mol·K)]
    
    D = D0 * np.exp(-Q / (R * T)) + np.random.normal(0, 1e-12, n_samples)
    
    return pd.DataFrame({
        'T': T,
        'D': D
    })

def generate_elastic_modulus_data(n_samples=100):
    """
    弾性率データを生成 (温度依存性)
    E = E₀ × (1 - α × (T - T₀))
    """
    T = np.random.uniform(200, 600, n_samples)  # 温度 [K]
    E0 = 200e9    # 基準弾性率 [Pa]
    alpha = 5e-4  # 温度係数 [K⁻¹]
    T0 = 300      # 基準温度 [K]
    
    E = E0 * (1 - alpha * (T - T0)) + np.random.normal(0, 1e9, n_samples)
    
    return pd.DataFrame({
        'T': T,
        'E': E
    })

def main():
    """メイン関数：全材料工学データを生成してCSVファイルに保存"""
    print("材料工学データを生成中...")
    
    thermal_data = generate_thermal_conductivity_data()
    hall_data = generate_hall_effect_data()
    hall_petch_data = generate_hall_petch_data()
    diffusion_data = generate_diffusion_data()
    elastic_data = generate_elastic_modulus_data()
    
    thermal_data.to_csv('thermal_conductivity.csv', index=False)
    hall_data.to_csv('hall_effect.csv', index=False)
    hall_petch_data.to_csv('hall_petch.csv', index=False)
    diffusion_data.to_csv('diffusion.csv', index=False)
    elastic_data.to_csv('elastic_modulus.csv', index=False)
    
    print("✅ 材料工学データ生成完了！")
    print("生成されたファイル:")
    print("- thermal_conductivity.csv (熱伝導率 - Wiedemann-Franz法則)")
    print("- hall_effect.csv (ホール効果)")
    print("- hall_petch.csv (機械的強度 - Hall-Petch関係)")
    print("- diffusion.csv (拡散係数 - Arrhenius式)")
    print("- elastic_modulus.csv (弾性率 - 温度依存性)")
    
    print("\n📊 データ概要:")
    print("\n1. 熱伝導率データ (Wiedemann-Franz法則):")
    print(thermal_data.describe())
    
    print("\n2. ホール効果データ:")
    print(hall_data.describe())
    
    print("\n3. Hall-Petch関係データ:")
    print(hall_petch_data.describe())
    
    print("\n4. 拡散係数データ (Arrhenius式):")
    print(diffusion_data.describe())
    
    print("\n5. 弾性率データ (温度依存性):")
    print(elastic_data.describe())

if __name__ == "__main__":
    main()
