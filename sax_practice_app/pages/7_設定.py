"""
Settings page for SAX practice app.
Allows users to configure app preferences, audio settings, and practice goals.
"""

import streamlit as st
from pathlib import Path
import json

st.set_page_config(
    page_title="設定 - SAX Practice",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ 設定")
st.markdown("---")

if 'settings' not in st.session_state:
    st.session_state.settings = {
        'user_name': 'Player',
        'target_bpm': 120,
        'daily_minutes': 30,
        'swing_ratio': 0.67,
        'metronome_sound': 'click',
        'metronome_volume': 0.7,
        'use_sharps': False,
        'theme': 'light',
        'language': 'ja',
        'show_hints': True,
        'auto_advance': False,
        'srs_strength': 'medium',
    }

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 プロフィール",
    "🎯 練習目標",
    "🔊 オーディオ",
    "🎨 表示",
    "📚 学習設定"
])

with tab1:
    st.header("プロフィール設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_name = st.text_input(
            "ユーザー名",
            value=st.session_state.settings['user_name'],
            help="練習記録に表示される名前"
        )
        st.session_state.settings['user_name'] = user_name
        
        st.markdown("### 楽器情報")
        instrument = st.selectbox(
            "楽器",
            ["Alto Sax (E♭)", "Tenor Sax (B♭)", "Soprano Sax (B♭)", "Baritone Sax (E♭)"],
            help="移調楽器の設定"
        )
        
        experience_level = st.select_slider(
            "経験レベル",
            options=["初心者", "初級", "中級", "上級", "プロ"],
            value="初級"
        )
    
    with col2:
        st.markdown("### 統計情報")
        st.metric("総練習時間", f"{st.session_state.get('total_practice_time', 0)} 分")
        st.metric("連続日数", f"{st.session_state.get('current_streak', 0)} 日")
        st.metric("達成率", f"{st.session_state.get('achievement_rate', 0)}%")
        
        if st.button("🗑️ 全データをリセット", type="secondary"):
            if st.checkbox("本当にリセットしますか？"):
                st.warning("この操作は取り消せません！")
                if st.button("確認：全データを削除"):
                    st.session_state.clear()
                    st.success("全データをリセットしました")
                    st.rerun()

with tab2:
    st.header("練習目標設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 日次目標")
        
        daily_minutes = st.slider(
            "1日の練習時間（分）",
            min_value=10,
            max_value=180,
            value=st.session_state.settings['daily_minutes'],
            step=5,
            help="1日の目標練習時間"
        )
        st.session_state.settings['daily_minutes'] = daily_minutes
        
        target_bpm = st.slider(
            "目標BPM",
            min_value=40,
            max_value=300,
            value=st.session_state.settings['target_bpm'],
            step=5,
            help="スケール練習の目標テンポ"
        )
        st.session_state.settings['target_bpm'] = target_bpm
        
        st.markdown("### 週次目標")
        weekly_sessions = st.number_input(
            "週の練習回数",
            min_value=1,
            max_value=7,
            value=5,
            help="週に何回練習するか"
        )
        
        focus_areas = st.multiselect(
            "重点練習項目",
            ["スケール", "コード進行", "耳トレ", "楽典", "アドリブ", "タンギング"],
            default=["スケール", "コード進行"]
        )
    
    with col2:
        st.markdown("### 長期目標")
        
        st.text_area(
            "3ヶ月目標",
            placeholder="例：メジャースケール全キー120BPMで演奏できる",
            height=100
        )
        
        st.text_area(
            "6ヶ月目標",
            placeholder="例：ii-V-Iを使った簡単なアドリブができる",
            height=100
        )
        
        st.text_area(
            "1年目標",
            placeholder="例：ジャズスタンダード10曲をアドリブで演奏できる",
            height=100
        )
        
        st.markdown("### 目標達成予測")
        if target_bpm > 0 and daily_minutes > 0:
            current_bpm = st.session_state.get('current_bpm', 60)
            days_to_goal = max(1, int((target_bpm - current_bpm) / 2))
            st.info(f"📅 目標達成まで約 **{days_to_goal}日** （現在BPM: {current_bpm}）")

with tab3:
    st.header("オーディオ設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### メトロノーム設定")
        
        metronome_sound = st.selectbox(
            "クリック音",
            ["click", "stick", "beep", "cowbell"],
            index=["click", "stick", "beep", "cowbell"].index(
                st.session_state.settings['metronome_sound']
            ),
            help="メトロノームのクリック音の種類"
        )
        st.session_state.settings['metronome_sound'] = metronome_sound
        
        metronome_volume = st.slider(
            "メトロノーム音量",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings['metronome_volume'],
            step=0.1,
            help="メトロノームの音量（0.0〜1.0）"
        )
        st.session_state.settings['metronome_volume'] = metronome_volume
        
        swing_ratio = st.slider(
            "デフォルトスイング比",
            min_value=0.50,
            max_value=0.75,
            value=st.session_state.settings['swing_ratio'],
            step=0.01,
            format="%.2f",
            help="スイングのデフォルト比率（50:50〜75:25）"
        )
        st.session_state.settings['swing_ratio'] = swing_ratio
        
        st.markdown(f"**スイング比:** {int(swing_ratio*100)}:{int((1-swing_ratio)*100)}")
        
        count_in_bars = st.number_input(
            "カウントイン小節数",
            min_value=0,
            max_value=4,
            value=1,
            help="練習開始前のカウントイン小節数"
        )
    
    with col2:
        st.markdown("### バッキングトラック設定")
        
        backing_style = st.selectbox(
            "デフォルトスタイル",
            ["Swing", "Latin", "Bossa", "Ballad", "Funk"],
            help="バッキングトラックのデフォルトスタイル"
        )
        
        backing_volume = st.slider(
            "バッキング音量",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        st.markdown("### オーディオ品質")
        
        sample_rate = st.selectbox(
            "サンプルレート",
            [22050, 44100, 48000],
            index=1,
            help="オーディオのサンプルレート（Hz）"
        )
        
        buffer_size = st.selectbox(
            "バッファサイズ",
            [256, 512, 1024, 2048],
            index=1,
            help="オーディオバッファサイズ（小さいほど低レイテンシ）"
        )
        
        st.info("💡 低レイテンシが必要な場合は、バッファサイズを小さくしてください")

with tab4:
    st.header("表示設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 外観")
        
        theme = st.selectbox(
            "テーマ",
            ["light", "dark", "auto"],
            index=["light", "dark", "auto"].index(st.session_state.settings['theme']),
            help="アプリのカラーテーマ"
        )
        st.session_state.settings['theme'] = theme
        
        language = st.selectbox(
            "言語",
            ["ja", "en"],
            index=["ja", "en"].index(st.session_state.settings['language']),
            format_func=lambda x: "日本語" if x == "ja" else "English",
            help="アプリの表示言語"
        )
        st.session_state.settings['language'] = language
        
        font_size = st.select_slider(
            "フォントサイズ",
            options=["小", "中", "大", "特大"],
            value="中"
        )
        
        use_sharps = st.checkbox(
            "シャープ表記を使用",
            value=st.session_state.settings['use_sharps'],
            help="音名をシャープ（#）で表記（オフの場合はフラット♭）"
        )
        st.session_state.settings['use_sharps'] = use_sharps
    
    with col2:
        st.markdown("### 表示オプション")
        
        show_hints = st.checkbox(
            "ヒントを表示",
            value=st.session_state.settings['show_hints'],
            help="練習のヒントやアドバイスを表示"
        )
        st.session_state.settings['show_hints'] = show_hints
        
        show_staff = st.checkbox(
            "五線譜を表示",
            value=True,
            help="音符を五線譜で表示"
        )
        
        show_fingering = st.checkbox(
            "運指を表示",
            value=False,
            help="サックスの運指図を表示"
        )
        
        show_keyboard = st.checkbox(
            "鍵盤を表示",
            value=True,
            help="ピアノ鍵盤で音名を表示"
        )
        
        st.markdown("### アニメーション")
        
        enable_animations = st.checkbox(
            "アニメーションを有効化",
            value=True,
            help="画面遷移やエフェクトのアニメーション"
        )
        
        animation_speed = st.select_slider(
            "アニメーション速度",
            options=["遅い", "普通", "速い"],
            value="普通",
            disabled=not enable_animations
        )

with tab5:
    st.header("学習設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### SRS（間隔反復）設定")
        
        srs_strength = st.select_slider(
            "SRS強度",
            options=["弱", "中", "強"],
            value=st.session_state.settings['srs_strength'],
            help="復習間隔の長さ（強いほど間隔が長い）"
        )
        st.session_state.settings['srs_strength'] = srs_strength
        
        auto_advance = st.checkbox(
            "自動進行",
            value=st.session_state.settings['auto_advance'],
            help="練習項目を自動的に次に進める"
        )
        st.session_state.settings['auto_advance'] = auto_advance
        
        review_before_new = st.checkbox(
            "新規項目の前に復習",
            value=True,
            help="新しい項目を学ぶ前に復習項目を優先"
        )
        
        daily_new_items = st.slider(
            "1日の新規項目数",
            min_value=1,
            max_value=20,
            value=5,
            help="1日に追加する新しい練習項目の数"
        )
        
        st.markdown("### 難易度調整")
        
        difficulty_level = st.select_slider(
            "全体難易度",
            options=["初心者", "初級", "中級", "上級"],
            value="初級"
        )
        
        adaptive_difficulty = st.checkbox(
            "適応的難易度調整",
            value=True,
            help="パフォーマンスに応じて自動的に難易度を調整"
        )
    
    with col2:
        st.markdown("### 練習モード")
        
        practice_mode = st.radio(
            "デフォルト練習モード",
            ["フリー", "ガイド付き", "チャレンジ"],
            help="練習セッションのデフォルトモード"
        )
        
        st.markdown("**モード説明:**")
        st.markdown("- **フリー**: 自由に練習項目を選択")
        st.markdown("- **ガイド付き**: AIが最適な練習順序を提案")
        st.markdown("- **チャレンジ**: 時間制限付きの課題に挑戦")
        
        st.markdown("### フィードバック設定")
        
        immediate_feedback = st.checkbox(
            "即時フィードバック",
            value=True,
            help="練習中にリアルタイムでフィードバック"
        )
        
        detailed_stats = st.checkbox(
            "詳細統計を表示",
            value=True,
            help="練習後に詳細な統計情報を表示"
        )
        
        encourage_messages = st.checkbox(
            "励ましメッセージ",
            value=True,
            help="練習中に励ましのメッセージを表示"
        )

st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("💾 設定を保存", type="primary", use_container_width=True):
        settings_file = Path(__file__).parent.parent / 'data' / 'settings.json'
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.settings, f, indent=2, ensure_ascii=False)
        
        st.success("✅ 設定を保存しました！")

with col2:
    if st.button("🔄 デフォルトに戻す", use_container_width=True):
        st.session_state.settings = {
            'user_name': 'Player',
            'target_bpm': 120,
            'daily_minutes': 30,
            'swing_ratio': 0.67,
            'metronome_sound': 'click',
            'metronome_volume': 0.7,
            'use_sharps': False,
            'theme': 'light',
            'language': 'ja',
            'show_hints': True,
            'auto_advance': False,
            'srs_strength': 'medium',
        }
        st.success("✅ デフォルト設定に戻しました")
        st.rerun()

with col3:
    if st.button("📤 設定をエクスポート", use_container_width=True):
        settings_json = json.dumps(st.session_state.settings, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 JSONをダウンロード",
            data=settings_json,
            file_name="sax_practice_settings.json",
            mime="application/json",
            use_container_width=True
        )

st.markdown("---")
st.markdown("### 📥 設定をインポート")
uploaded_file = st.file_uploader(
    "設定ファイル（JSON）をアップロード",
    type=['json'],
    help="以前エクスポートした設定ファイルをインポート"
)

if uploaded_file is not None:
    try:
        imported_settings = json.load(uploaded_file)
        st.session_state.settings.update(imported_settings)
        st.success("✅ 設定をインポートしました！")
        st.rerun()
    except Exception as e:
        st.error(f"❌ 設定のインポートに失敗しました: {e}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    SAX Practice App v0.1 | 設定は自動的に保存されます
    </div>
    """,
    unsafe_allow_html=True
)
