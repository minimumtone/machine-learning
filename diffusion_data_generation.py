"""
拡散方程式発見用データ生成スクリプト
FDMによる拡散方程式の数値解を生成し、CSVファイルとして保存
"""

import numpy as np
import pandas as pd
from pde_discovery import DiffusionFDM

def generate_diffusion_data():
    """単一シナリオの拡散データを生成"""
    
    fdm = DiffusionFDM(L=0.02, T_final=3600, nx=50, nt=100, D=1e-11)
    u_numerical = fdm.solve()
    
    data = []
    for i, t in enumerate(fdm.t):
        for j, x in enumerate(fdm.x):
            data.append({
                't': t,
                'x': x,
                'c': u_numerical[i, j]
            })
    
    return pd.DataFrame(data)

def generate_multiple_diffusion_scenarios():
    """複数のシナリオでデータを生成"""
    
    scenarios = [
        {"D": 5e-12, "name": "low_diffusion"},
        {"D": 1e-11, "name": "medium_diffusion"},
        {"D": 5e-11, "name": "high_diffusion"}
    ]
    
    all_data = []
    
    for scenario in scenarios:
        fdm = DiffusionFDM(L=0.02, T_final=3600, nx=30, nt=50, D=scenario["D"])
        u_numerical = fdm.solve()
        
        for i, t in enumerate(fdm.t):
            for j, x in enumerate(fdm.x):
                all_data.append({
                    't': t,
                    'x': x,
                    'c': u_numerical[i, j],
                    'D_true': scenario["D"],
                    'scenario': scenario["name"]
                })
    
    return pd.DataFrame(all_data)

def main():
    """メイン関数：拡散方程式発見用データを生成してCSVファイルに保存"""
    print("拡散方程式発見用データを生成中...")
    
    single_data = generate_diffusion_data()
    single_data.to_csv('diffusion_single.csv', index=False)
    
    multi_data = generate_multiple_diffusion_scenarios()
    multi_data.to_csv('diffusion_multi.csv', index=False)
    
    print("✅ 拡散方程式発見用データ生成完了！")
    print("生成されたファイル:")
    print("- diffusion_single.csv (単一シナリオ)")
    print("- diffusion_multi.csv (複数シナリオ)")
    
    print("\n📊 データ概要:")
    print("\n1. 単一シナリオデータ:")
    print(f"   データ点数: {len(single_data)}")
    print(f"   時間範囲: {single_data['t'].min():.0f} - {single_data['t'].max():.0f} s")
    print(f"   空間範囲: {single_data['x'].min():.4f} - {single_data['x'].max():.4f} m")
    print(f"   濃度範囲: {single_data['c'].min():.3f} - {single_data['c'].max():.3f}")
    
    print("\n2. 複数シナリオデータ:")
    print(f"   データ点数: {len(multi_data)}")
    print(f"   シナリオ数: {len(multi_data['scenario'].unique())}")
    print(f"   拡散係数範囲: {multi_data['D_true'].min():.2e} - {multi_data['D_true'].max():.2e} m²/s")

if __name__ == "__main__":
    main()
