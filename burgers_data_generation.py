"""
Burgers方程式発見用データ生成スクリプト
FDMによるBurgers方程式の数値解を生成し、CSVファイルとして保存
"""

import numpy as np
import pandas as pd
from pde_discovery import BurgersFDM, NumericalDerivatives

def generate_burgers_data(nu: float = 0.01, nx: int = 50, nt: int = 100, 
                         T_final: float = 0.5, noise_level: float = 0.001):
    """
    Burgers方程式のFDMデータを生成
    
    Parameters:
    nu: 粘性係数
    nx: 空間格子点数
    nt: 時間格子点数
    T_final: 最終時刻
    noise_level: ノイズレベル
    """
    
    fdm = BurgersFDM(nx=nx, nt=nt, nu=nu, T_final=T_final)
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
                'u_dudx': u_noisy[i, j] * dudx[i, j],
                'nu_true': nu
            })
    
    return pd.DataFrame(data_list)

def generate_multiple_burgers_scenarios():
    """複数のシナリオでBurgersデータを生成"""
    
    scenarios = [
        {"nu": 0.001, "name": "low_viscosity"},
        {"nu": 0.01, "name": "medium_viscosity"},
        {"nu": 0.05, "name": "high_viscosity"}
    ]
    
    all_data = []
    
    for scenario in scenarios:
        print(f"Generating Burgers data for {scenario['name']} (ν = {scenario['nu']})...")
        
        data = generate_burgers_data(
            nu=scenario['nu'],
            nx=40,
            nt=80,
            T_final=0.4,
            noise_level=0.001
        )
        
        data['scenario'] = scenario['name']
        all_data.append(data)
    
    return pd.concat(all_data, ignore_index=True)

def main():
    """メイン関数：Burgers方程式発見用データを生成してCSVファイルに保存"""
    print("Burgers方程式発見用データを生成中...")
    
    single_data = generate_burgers_data()
    single_data.to_csv('burgers_single.csv', index=False)
    
    multi_data = generate_multiple_burgers_scenarios()
    multi_data.to_csv('burgers_multi.csv', index=False)
    
    print("✅ Burgers方程式発見用データ生成完了！")
    print("生成されたファイル:")
    print("- burgers_single.csv (単一シナリオ)")
    print("- burgers_multi.csv (複数シナリオ)")
    
    print("\n📊 データ概要:")
    print("\n1. 単一シナリオデータ:")
    print(single_data.describe())
    
    print("\n2. 複数シナリオデータ:")
    print(multi_data.groupby('scenario').describe())

if __name__ == "__main__":
    main()
