"""
状態空間モデル・データ同化システムのテストスクリプト
"""

import sys
import numpy as np
from state_space_model_app import (
    KalmanFilter, ParticleFilter, EnsembleKalmanFilter,
    generate_lorenz_data, generate_simple_model_data
)

def test_basic_functionality():
    """基本機能のテスト"""
    print("🧪 基本機能テスト開始...")
    
    try:
        print("  📊 データ生成テスト...")
        t, true_states, obs = generate_simple_model_data('random_walk', 50)
        assert len(t) == 50
        assert len(true_states) == 50
        assert len(obs) == 50
        print("    ✅ シンプルモデルデータ生成成功")
        
        time_grid, true_states, observations = generate_lorenz_data(100)
        assert len(time_grid) == 100
        assert true_states.shape == (100, 3)
        print("    ✅ ローレンツデータ生成成功")
        
        print("  🎯 カルマンフィルタテスト...")
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[0.2]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        
        kf = KalmanFilter(F, H, Q, R, x0, P0)
        test_obs = np.random.randn(20, 1)
        results = kf.assimilate(test_obs)
        
        assert 'states' in results
        assert 'log_likelihood' in results
        assert results['states'].shape == (20, 1)
        print("    ✅ カルマンフィルタ動作成功")
        
        print("  🎲 パーティクルフィルタテスト...")
        def transition_func(x):
            return x + 0.1 * np.random.randn()
        
        def observation_func(x):
            return np.array([x])
        
        def process_noise_func():
            return np.random.randn() * 0.1
        
        pf = ParticleFilter(100, 1, 1, transition_func, observation_func, process_noise_func, 0.2)
        test_obs = np.random.randn(10, 1)
        pf_results = pf.assimilate(test_obs)
        
        assert 'states' in pf_results
        assert pf_results['states'].shape == (10, 1)
        print("    ✅ パーティクルフィルタ動作成功")
        
        print("  🌊 アンサンブルカルマンフィルタテスト...")
        def simple_transition(x):
            return x * 0.9
        
        def simple_observation(x):
            return np.array([x])
        
        enkf = EnsembleKalmanFilter(50, 1, 1, simple_transition, simple_observation, 0.1, 0.2)
        enkf_results = enkf.assimilate(test_obs)
        
        assert 'states' in enkf_results
        assert enkf_results['states'].shape == (10, 1)
        print("    ✅ アンサンブルカルマンフィルタ動作成功")
        
        print("✅ 全ての基本機能テストが成功しました！")
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lorenz_assimilation():
    """ローレンツ方程式でのデータ同化テスト"""
    print("🌪️ ローレンツ方程式データ同化テスト開始...")
    
    try:
        time_grid, true_states, observations = generate_lorenz_data(200, obs_interval=5)
        
        F = np.eye(3)
        H = np.eye(3)
        Q = np.eye(3) * 0.1
        R = np.eye(3) * 0.25
        x0 = observations[0]
        P0 = np.eye(3) * 10.0
        
        kf = KalmanFilter(F, H, Q, R, x0, P0)
        results = kf.assimilate(observations)
        
        assert results['states'].shape == (len(observations), 3)
        assert len(results['innovations']) == len(observations)
        
        obs_times = np.arange(0, len(true_states), 5)
        true_at_obs = true_states[obs_times]
        rmse = np.sqrt(np.mean((results['states'] - true_at_obs)**2))
        
        print(f"    📊 RMSE: {rmse:.4f}")
        print(f"    📈 対数尤度: {results['log_likelihood']:.4f}")
        
        assert rmse < 5.0  # 合理的な範囲内
        assert not np.isnan(results['log_likelihood'])
        
        print("✅ ローレンツ方程式データ同化テスト成功！")
        return True
        
    except Exception as e:
        print(f"❌ ローレンツテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 状態空間モデル・データ同化システムテスト開始")
    print("=" * 60)
    
    basic_success = test_basic_functionality()
    print()
    
    lorenz_success = test_lorenz_assimilation()
    print()
    
    print("=" * 60)
    if basic_success and lorenz_success:
        print("🎉 全てのテストが成功しました！")
        print("✅ 状態空間モデル・データ同化システムは正常に動作します")
    else:
        print("❌ 一部のテストが失敗しました")
        sys.exit(1)
