"""
SAX Practice App - Main Entry Point
Streamlit-based web application for saxophone jazz practice
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime, date
import json

sys.path.append(str(Path(__file__).parent))

from utils import SRSManager, get_all_keys
from utils.theory import SCALE_INTERVALS


st.set_page_config(
    page_title="SAX Practice App",
    page_icon="🎷",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "Player"
    
    if 'target_bpm' not in st.session_state:
        st.session_state.target_bpm = 120
    
    if 'daily_minutes' not in st.session_state:
        st.session_state.daily_minutes = 30
    
    if 'swing_ratio' not in st.session_state:
        st.session_state.swing_ratio = 0.67
    
    if 'practice_history' not in st.session_state:
        st.session_state.practice_history = []
    
    if 'current_streak' not in st.session_state:
        st.session_state.current_streak = 0
    
    if 'total_practice_time' not in st.session_state:
        st.session_state.total_practice_time = 0
    
    if 'srs_manager' not in st.session_state:
        data_dir = Path(__file__).parent / 'data'
        data_dir.mkdir(exist_ok=True)
        st.session_state.srs_manager = SRSManager(data_dir / 'srs_items.json')


def load_user_data():
    data_file = Path(__file__).parent / 'data' / 'user_data.json'
    if data_file.exists():
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                st.session_state.user_name = data.get('user_name', 'Player')
                st.session_state.target_bpm = data.get('target_bpm', 120)
                st.session_state.daily_minutes = data.get('daily_minutes', 30)
                st.session_state.swing_ratio = data.get('swing_ratio', 0.67)
                st.session_state.practice_history = data.get('practice_history', [])
                st.session_state.current_streak = data.get('current_streak', 0)
                st.session_state.total_practice_time = data.get('total_practice_time', 0)
        except Exception as e:
            st.error(f"Error loading user data: {e}")


def save_user_data():
    data_file = Path(__file__).parent / 'data' / 'user_data.json'
    data_file.parent.mkdir(exist_ok=True)
    
    data = {
        'user_name': st.session_state.user_name,
        'target_bpm': st.session_state.target_bpm,
        'daily_minutes': st.session_state.daily_minutes,
        'swing_ratio': st.session_state.swing_ratio,
        'practice_history': st.session_state.practice_history,
        'current_streak': st.session_state.current_streak,
        'total_practice_time': st.session_state.total_practice_time,
    }
    
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sidebar():
    with st.sidebar:
        st.title("🎷 SAX Practice")
        st.markdown("---")
        
        st.subheader("👤 Profile")
        user_name = st.text_input("Name", value=st.session_state.user_name, key="sidebar_name")
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name
            save_user_data()
        
        st.markdown("---")
        
        st.subheader("🎯 Goals")
        target_bpm = st.number_input("Target BPM", min_value=40, max_value=300, 
                                     value=st.session_state.target_bpm, step=5)
        if target_bpm != st.session_state.target_bpm:
            st.session_state.target_bpm = target_bpm
            save_user_data()
        
        daily_minutes = st.number_input("Daily Practice (min)", min_value=5, max_value=180,
                                       value=st.session_state.daily_minutes, step=5)
        if daily_minutes != st.session_state.daily_minutes:
            st.session_state.daily_minutes = daily_minutes
            save_user_data()
        
        st.markdown("---")
        
        st.subheader("📊 Today's Progress")
        today_practice = sum(
            record['duration_min'] for record in st.session_state.practice_history
            if record.get('date') == str(date.today())
        )
        progress = min(today_practice / st.session_state.daily_minutes, 1.0)
        st.progress(progress)
        st.write(f"{today_practice:.1f} / {st.session_state.daily_minutes} min")
        
        st.markdown("---")
        
        st.subheader("🔥 Streak")
        st.metric("Current Streak", f"{st.session_state.current_streak} days")
        
        st.markdown("---")
        
        st.subheader("📚 SRS Items")
        srs_stats = st.session_state.srs_manager.get_stats()
        st.write(f"Due: {srs_stats['due']}")
        st.write(f"Total: {srs_stats['total']}")


def home_page():
    st.title("🎷 Welcome to SAX Practice App")
    st.markdown("### Your comprehensive jazz saxophone learning companion")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Practice Time", f"{st.session_state.total_practice_time:.1f} min")
    
    with col2:
        st.metric("Current Streak", f"{st.session_state.current_streak} days")
    
    with col3:
        today_practice = sum(
            record['duration_min'] for record in st.session_state.practice_history
            if record.get('date') == str(date.today())
        )
        st.metric("Today's Practice", f"{today_practice:.1f} min")
    
    st.markdown("---")
    
    st.subheader("📅 Today's Practice Menu")
    
    srs_manager = st.session_state.srs_manager
    due_items = srs_manager.get_due_items(limit=5)
    
    if due_items:
        st.write("**Due for Review:**")
        for item in due_items:
            if item.item_type == 'scale_pattern':
                content = item.content
                st.write(f"- Scale: {content['key']} {content['scale_type']} ({content['pattern']})")
            elif item.item_type == 'chord_progression':
                content = item.content
                st.write(f"- Progression: {content['progression']} in {content['key']}")
            elif item.item_type == 'theory_quiz':
                st.write(f"- Theory Quiz: {item.content.get('question', 'Quiz')[:50]}...")
    else:
        st.info("No items due for review today. Great job staying on track!")
    
    st.markdown("---")
    
    st.subheader("🎯 Recommended Practice Routine")
    
    routine = [
        ("🎵", "Warm-up", "Long tones and breathing exercises", "5 min"),
        ("🎼", "Scales", "Practice major scales in all 12 keys", "10 min"),
        ("🎹", "Chord Progressions", "ii-V-I in different keys", "10 min"),
        ("👂", "Ear Training", "Interval recognition and call & response", "5 min"),
        ("🎺", "Improvisation", "Practice over backing tracks", "10 min"),
    ]
    
    for emoji, title, description, duration in routine:
        with st.expander(f"{emoji} {title} ({duration})"):
            st.write(description)
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎵 Start Metronome", use_container_width=True):
            st.switch_page("pages/1_メトロノーム.py")
    
    with col2:
        if st.button("🎼 Practice Scales", use_container_width=True):
            st.switch_page("pages/2_スケール練習.py")
    
    with col3:
        if st.button("🎹 Chord Progressions", use_container_width=True):
            st.switch_page("pages/3_コード進行.py")
    
    st.markdown("---")
    
    st.subheader("📖 Getting Started with Jazz")
    
    with st.expander("🎓 What is Jazz?"):
        st.write("""
        Jazz is a music genre that originated in African-American communities in the late 19th and early 20th centuries.
        It's characterized by:
        - **Swing feel**: Uneven eighth notes that create a bouncing rhythm
        - **Improvisation**: Creating melodies spontaneously over chord progressions
        - **Blues influence**: Use of blue notes and blues scales
        - **Complex harmony**: Extended chords (7ths, 9ths, 11ths, 13ths)
        """)
    
    with st.expander("🎷 Why Saxophone?"):
        st.write("""
        The saxophone is one of the most expressive instruments in jazz:
        - Wide dynamic range from soft whispers to powerful screams
        - Ability to bend notes and use vibrato for emotional expression
        - Central role in jazz history (Charlie Parker, John Coltrane, Sonny Rollins)
        - Versatile across many jazz styles (bebop, cool jazz, free jazz, fusion)
        """)
    
    with st.expander("📚 Learning Path"):
        st.write("""
        1. **Fundamentals** (Weeks 1-4)
           - Proper embouchure and breathing
           - Major scales in all keys
           - Basic rhythm and swing feel
        
        2. **Jazz Basics** (Weeks 5-12)
           - Blues scales and blues form
           - ii-V-I progressions
           - Guide tones and voice leading
           - Simple improvisation
        
        3. **Intermediate** (Months 4-12)
           - All modes and altered scales
           - Bebop scales and patterns
           - Transcribing solos
           - Playing over standards
        
        4. **Advanced** (Year 2+)
           - Complex harmony (Coltrane changes)
           - Advanced improvisation concepts
           - Developing your own voice
           - Playing in ensembles
        """)


def main():
    init_session_state()
    load_user_data()
    sidebar()
    home_page()


if __name__ == "__main__":
    main()
