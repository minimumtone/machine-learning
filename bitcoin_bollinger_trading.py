import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Bitcoin ボリンジャーバンド取引システム",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Bitcoin ボリンジャーバンド取引システム")
st.markdown("**2日後の売買で利益最大化するパラメータ最適化システム**")

@st.cache_data
def load_bitcoin_data(start_date=None, end_date=None, period="2y"):
    """ビットコイン価格データを取得（期間指定またはデフォルト2年間）"""
    try:
        btc = yf.Ticker("BTC-USD")
        if start_date and end_date:
            data = btc.history(start=start_date, end=end_date)
        else:
            data = btc.history(period=period)
        return data
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None

def calculate_bollinger_bands(data, window=20, num_std=2.0):
    """ボリンジャーバンドを計算"""
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return rolling_mean, upper_band, lower_band

def generate_trading_signals(data, window=20, num_std=2.0):
    """取引シグナルを生成"""
    middle_band, upper_band, lower_band = calculate_bollinger_bands(data, window, num_std)
    
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['middle_band'] = middle_band
    signals['upper_band'] = upper_band
    signals['lower_band'] = lower_band
    
    signals['position'] = 0
    signals['signal'] = 0
    
    signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1  # 買い
    signals.loc[signals['price'] > signals['upper_band'], 'signal'] = -1  # 売り
    
    signals['position'] = signals['signal'].shift(1).fillna(0)
    
    return signals

def backtest_strategy(data, window=20, num_std=2.0, holding_days=2):
    """バックテスト実行"""
    signals = generate_trading_signals(data, window, num_std)
    
    signals['future_price'] = signals['price'].shift(-holding_days)
    signals['returns'] = (signals['future_price'] - signals['price']) / signals['price']
    
    trading_returns = signals[signals['signal'] != 0].copy()
    trading_returns['strategy_returns'] = trading_returns['returns'] * trading_returns['signal']
    
    if len(trading_returns) > 0:
        total_return = trading_returns['strategy_returns'].sum()
        win_rate = (trading_returns['strategy_returns'] > 0).mean()
        num_trades = len(trading_returns)
        avg_return = trading_returns['strategy_returns'].mean()
    else:
        total_return = 0
        win_rate = 0
        num_trades = 0
        avg_return = 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'avg_return': avg_return,
        'signals': signals,
        'trading_returns': trading_returns
    }

def optimize_parameters(data):
    """パラメータ最適化（1-30日の保有期間を含む）"""
    best_result = None
    best_params = None
    best_return = -999
    
    holding_days_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 30]
    
    for window in range(10, 31, 5):  # 10, 15, 20, 25, 30
        for num_std in np.arange(1.0, 3.1, 0.5):  # 1.0, 1.5, 2.0, 2.5, 3.0
            for holding_days in holding_days_range:
                result = backtest_strategy(data, window, float(num_std), holding_days)
                if result['total_return'] > best_return:
                    best_return = result['total_return']
                    best_params = (window, num_std, holding_days)
                    best_result = result
    
    return best_params, best_result

def plot_bollinger_bands(data, signals, window=20, num_std=2.0):
    """ボリンジャーバンドと取引シグナルをプロット"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Bitcoin価格とボリンジャーバンド', '取引シグナル'),
        row_width=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=signals['price'], name='Bitcoin価格', line=dict(color='black')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=signals['upper_band'], name='上限バンド', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=signals['middle_band'], name='中央線', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=signals['lower_band'], name='下限バンド', line=dict(color='green', dash='dash')),
        row=1, col=1
    )
    
    buy_signals = signals[signals['signal'] == 1]
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=buy_signals['price'], 
                      mode='markers', name='買いシグナル', 
                      marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )
    
    sell_signals = signals[signals['signal'] == -1]
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=sell_signals['price'], 
                      mode='markers', name='売りシグナル', 
                      marker=dict(color='red', size=10, symbol='triangle-down')),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=signals['signal'], name='シグナル', line=dict(color='purple')),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'Bitcoin ボリンジャーバンド取引システム (期間: {window}, 標準偏差: {num_std})',
        xaxis_title='日付',
        yaxis_title='価格 (USD)',
        height=800
    )
    
    return fig

def plot_portfolio_performance(signals, initial_capital=100000):
    """資産推移グラフを作成（ホールド戦略との比較付き）"""
    trading_data = signals[signals['signal'] != 0].copy()
    if len(trading_data) == 0:
        return go.Figure()
    
    trading_data = trading_data.sort_index()
    
    portfolio_value = [initial_capital]
    dates = [trading_data.index[0]]
    cash = initial_capital
    position = 0
    
    for i, (date, row) in enumerate(trading_data.iterrows()):
        if row['signal'] == 1 and cash > 0:  # 買いシグナル
            position = cash / row['price']
            cash = 0
        elif row['signal'] == -1 and position > 0:  # 売りシグナル
            cash = position * row['price']
            position = 0
        
        if position > 0:
            current_value = position * row['price']
        else:
            current_value = cash
            
        portfolio_value.append(current_value)
        dates.append(date)
    
    initial_price = signals['price'].iloc[0]
    bitcoin_amount = initial_capital / initial_price
    hold_values = signals['price'] * bitcoin_amount
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_value,
        mode='lines',
        name='ボリンジャーバンド戦略',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=signals.index,
        y=hold_values,
        mode='lines',
        name='ホールド戦略（放置）',
        line=dict(color='orange', width=2, dash='dot')
    ))
    
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="初期資本"
    )
    
    final_value_strategy = portfolio_value[-1]
    final_value_hold = hold_values.iloc[-1]
    strategy_return = (final_value_strategy - initial_capital) / initial_capital
    hold_return = (final_value_hold - initial_capital) / initial_capital
    outperformance = strategy_return - hold_return
    
    fig.update_layout(
        title=f'📈 資産推移グラフ比較<br>戦略: ${final_value_strategy:,.0f} ({strategy_return:.1%}) vs ホールド: ${final_value_hold:,.0f} ({hold_return:.1%})<br>アウトパフォーマンス: {outperformance:.1%}',
        xaxis_title='日付',
        yaxis_title='ポートフォリオ価値 (USD)',
        hovermode='x unified',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_cumulative_returns(trading_returns):
    """累積リターンの推移グラフを作成"""
    if len(trading_returns) == 0:
        return go.Figure()
    
    cumulative_returns = (1 + trading_returns['strategy_returns']).cumprod() - 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns * 100,
        mode='lines',
        name='累積リターン',
        line=dict(color='green', width=3),
        fill='tonexty'
    ))
    
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="ブレークイーブン"
    )
    
    fig.update_layout(
        title='📊 累積リターンの推移',
        xaxis_title='日付',
        yaxis_title='累積リターン (%)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def main():
    st.sidebar.header("パラメータ設定")
    
    st.sidebar.subheader("📅 バックテスト期間選択")
    use_historical = st.sidebar.checkbox("10年前からの期間を選択", value=False)
    
    if use_historical:
        from datetime import datetime, timedelta
        
        current_year = datetime.now().year
        start_year = st.sidebar.selectbox(
            "開始年を選択",
            range(current_year - 10, current_year),
            index=5  # デフォルトは5年前
        )
        
        start_month = st.sidebar.selectbox(
            "開始月を選択",
            range(1, 13),
            index=0  # デフォルトは1月
        )
        
        start_date = datetime(start_year, start_month, 1)
        end_date = start_date + timedelta(days=365)
        
        st.sidebar.info(f"選択期間: {start_date.strftime('%Y-%m-%d')} から {end_date.strftime('%Y-%m-%d')}")
        
        with st.spinner("選択された期間のBitcoinデータを読み込み中..."):
            data = load_bitcoin_data(start_date=start_date, end_date=end_date)
    else:
        with st.spinner("Bitcoinデータを読み込み中..."):
            data = load_bitcoin_data()
    
    if data is None:
        st.error("データの読み込みに失敗しました")
        return
    
    st.success(f"データ読み込み完了: {len(data)} 日分のデータ")
    st.write(f"期間: {data.index[0].strftime('%Y-%m-%d')} から {data.index[-1].strftime('%Y-%m-%d')}")
    
    if st.sidebar.button("パラメータ最適化実行"):
        with st.spinner("最適パラメータを計算中（1-30日の保有期間を含む）..."):
            best_params, best_result = optimize_parameters(data)
            
            st.session_state['best_params'] = best_params
            st.session_state['best_result'] = best_result
    
    if 'best_params' in st.session_state:
        default_window, default_std, default_holding = st.session_state['best_params']
        st.sidebar.success(f"最適化完了! 最適パラメータ: 期間={default_window}, 標準偏差={default_std:.1f}, 保有日数={default_holding}")
    else:
        default_window, default_std, default_holding = 20, 2.0, 2
    
    window = st.sidebar.slider("ボリンジャーバンド期間", 5, 50, default_window)
    num_std = st.sidebar.slider("標準偏差倍数", 0.5, 4.0, default_std, 0.1)
    holding_days = st.sidebar.slider("保有日数", 1, 30, default_holding)
    
    with st.spinner("バックテスト実行中..."):
        result = backtest_strategy(data, window, num_std, holding_days)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総収益率", f"{result['total_return']:.2%}")
    
    with col2:
        st.metric("勝率", f"{result['win_rate']:.1%}")
    
    with col3:
        st.metric("取引回数", f"{result['num_trades']}")
    
    with col4:
        st.metric("平均収益率", f"{result['avg_return']:.2%}")
    
    fig = plot_bollinger_bands(data, result['signals'], window, num_std)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📈 資産推移グラフ")
    portfolio_fig = plot_portfolio_performance(result['signals'], initial_capital=100000)
    st.plotly_chart(portfolio_fig, use_container_width=True)
    
    if len(result['trading_returns']) > 0:
        st.subheader("📊 累積リターンの推移")
        cumulative_fig = plot_cumulative_returns(result['trading_returns'])
        st.plotly_chart(cumulative_fig, use_container_width=True)
    
    if len(result['trading_returns']) > 0:
        st.subheader("取引詳細")
        
        trading_details = result['trading_returns'][['price', 'future_price', 'signal', 'returns', 'strategy_returns']].copy()
        trading_details['signal_type'] = trading_details['signal'].map({1: '買い', -1: '売り'})
        trading_details['profit_loss'] = trading_details['strategy_returns'].apply(lambda x: '利益' if x > 0 else '損失')
        
        st.dataframe(
            trading_details[['signal_type', 'price', 'future_price', 'strategy_returns', 'profit_loss']].round(4),
            column_config={
                'signal_type': '取引タイプ',
                'price': '取引価格',
                'future_price': f'{holding_days}日後価格',
                'strategy_returns': '収益率',
                'profit_loss': '結果'
            }
        )
        
        st.subheader("収益分布")
        fig_hist = px.histogram(
            trading_details, 
            x='strategy_returns', 
            nbins=20,
            title='取引収益率の分布',
            labels={'strategy_returns': '収益率', 'count': '頻度'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    if 'best_result' in st.session_state:
        st.subheader("最適化結果")
        best_result = st.session_state['best_result']
        best_params = st.session_state['best_params']
        
        st.write(f"**最適パラメータ:** 期間={best_params[0]}, 標準偏差={best_params[1]:.1f}, 保有日数={best_params[2]}")
        st.write(f"**最大総収益率:** {best_result['total_return']:.2%}")
        st.write(f"**勝率:** {best_result['win_rate']:.1%}")
        st.write(f"**取引回数:** {best_result['num_trades']}")

if __name__ == "__main__":
    main()
