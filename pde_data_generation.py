"""
PDE発見用データ生成スクリプト
FDMによる熱伝導方程式の数値解を生成し、CSVファイルとして保存
"""

import numpy as np
import pandas as pd
from pde_discovery import HeatConductionFDM, NumericalDerivatives

def generate_heat_conduction_data(alpha: float = 0.01, nx: int = 50, nt: int = 100, 
                                 T_final: float = 1.0, noise_level: float = 0.001):
    """
    熱伝導方程式のFDMデータを生成
    
    Parameters:
    alpha: 熱拡散係数
    nx: 空間格子点数
    nt: 時間格子点数
    T_final: 最終時刻
    noise_level: ノイズレベル
    """
    
    fdm = HeatConductionFDM(nx=nx, nt=nt, alpha=alpha, T_final=T_final)
    u_numerical = fdm.solve()
    
    u_noisy = u_numerical + np.random.normal(0, noise_level, u_numerical.shape)
    
    derivatives = NumericalDerivatives()
    dudt = derivatives.compute_dt(u_noisy, fdm.dt)
    dudx = derivatives.compute_dx(u_noisy, fdm.dx)
    d2udx2 = derivatives.compute_d2x(u_noisy, fdm.dx)
    
    data_list = []
    
    for i in range(nt):
        for j in range(nx):
            data_list.append({
                't': fdm.t[i],
                'x': fdm.x[j],
                'u': u_noisy[i, j],
                'dudt': dudt[i, j],
                'dudx': dudx[i, j],
                'd2udx2': d2udx2[i, j],
                'alpha_true': alpha
            })
    
    return pd.DataFrame(data_list)

def generate_multiple_scenarios():
    """複数のシナリオでデータを生成"""
    
    scenarios = [
        {"alpha": 0.01, "name": "low_diffusion"},
        {"alpha": 0.05, "name": "medium_diffusion"},
        {"alpha": 0.1, "name": "high_diffusion"}
    ]
    
    all_data = []
    
    for scenario in scenarios:
        print(f"Generating data for {scenario['name']} (α = {scenario['alpha']})...")
        
        data = generate_heat_conduction_data(
            alpha=scenario['alpha'],
            nx=40,
            nt=80,
            T_final=0.8,
            noise_level=0.001
        )
        
        data['scenario'] = scenario['name']
        all_data.append(data)
    
    return pd.concat(all_data, ignore_index=True)

def main():
    """メイン関数：PDE発見用データを生成してCSVファイルに保存"""
    print("PDE発見用データを生成中...")
    
    single_data = generate_heat_conduction_data()
    single_data.to_csv('heat_conduction_single.csv', index=False)
    
    multi_data = generate_multiple_scenarios()
    multi_data.to_csv('heat_conduction_multi.csv', index=False)
    
    print("✅ PDE発見用データ生成完了！")
    print("生成されたファイル:")
    print("- heat_conduction_single.csv (単一シナリオ)")
    print("- heat_conduction_multi.csv (複数シナリオ)")
    
    print("\n📊 データ概要:")
    print("\n1. 単一シナリオデータ:")
    print(single_data.describe())
    
    print("\n2. 複数シナリオデータ:")
    print(multi_data.groupby('scenario').describe())

if __name__ == "__main__":
    main()
