"""
Metronome Page - High-precision metronome with Tone.js
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


st.set_page_config(
    page_title="Metronome - SAX Practice",
    page_icon="🎵",
    layout="wide"
)


def init_metronome_state():
    if 'metro_bpm' not in st.session_state:
        st.session_state.metro_bpm = 120
    if 'metro_beats' not in st.session_state:
        st.session_state.metro_beats = 4
    if 'metro_subdivision' not in st.session_state:
        st.session_state.metro_subdivision = 1
    if 'metro_accent_beats' not in st.session_state:
        st.session_state.metro_accent_beats = [1]
    if 'metro_swing_ratio' not in st.session_state:
        st.session_state.metro_swing_ratio = 67
    if 'metro_click_pattern' not in st.session_state:
        st.session_state.metro_click_pattern = 'all'


def main():
    init_metronome_state()
    
    st.title("🎵 Metronome")
    st.markdown("High-precision metronome for practice")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Settings")
        
        bpm = st.slider("BPM (Beats Per Minute)", 
                       min_value=20, max_value=300, 
                       value=st.session_state.metro_bpm, 
                       step=1,
                       key="bpm_slider")
        st.session_state.metro_bpm = bpm
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            time_sig_options = {
                "4/4": (4, 4),
                "3/4": (3, 4),
                "5/4": (5, 4),
                "6/8": (6, 8),
                "7/8": (7, 8),
                "2/4": (2, 4),
            }
            time_sig = st.selectbox("Time Signature", 
                                   options=list(time_sig_options.keys()),
                                   index=0)
            beats, beat_unit = time_sig_options[time_sig]
            st.session_state.metro_beats = beats
        
        with col_b:
            subdivision = st.selectbox("Subdivision",
                                      options=[1, 2, 3, 4, 6, 8],
                                      index=0,
                                      format_func=lambda x: {
                                          1: "Quarter notes",
                                          2: "8th notes",
                                          3: "Triplets",
                                          4: "16th notes",
                                          6: "Sextuplets",
                                          8: "32nd notes"
                                      }.get(x, f"{x}"))
            st.session_state.metro_subdivision = subdivision
        
        st.markdown("---")
        
        st.subheader("Accent Pattern")
        accent_beats = st.multiselect("Accent on beats:",
                                     options=list(range(1, beats + 1)),
                                     default=st.session_state.metro_accent_beats)
        st.session_state.metro_accent_beats = accent_beats if accent_beats else [1]
        
        st.markdown("---")
        
        st.subheader("Click Pattern")
        click_pattern = st.radio("Click on:",
                                ["All beats", "Beats 2 & 4 only", "Beat 1 only", "Custom"],
                                index=0)
        
        if click_pattern == "All beats":
            click_on_beats = list(range(1, beats + 1))
        elif click_pattern == "Beats 2 & 4 only":
            click_on_beats = [2, 4] if beats >= 4 else [2]
        elif click_pattern == "Beat 1 only":
            click_on_beats = [1]
        else:
            click_on_beats = st.multiselect("Click on beats:",
                                           options=list(range(1, beats + 1)),
                                           default=list(range(1, beats + 1)))
        
        st.markdown("---")
        
        st.subheader("Swing Feel")
        swing_ratio = st.slider("Swing Ratio (%)", 
                               min_value=50, max_value=75, 
                               value=st.session_state.metro_swing_ratio,
                               help="50% = straight, 67% = standard swing, 75% = heavy swing")
        st.session_state.metro_swing_ratio = swing_ratio
        
        if swing_ratio > 50:
            st.info(f"Swing: {swing_ratio}:{100-swing_ratio} ratio")
    
    with col2:
        st.subheader("Quick Presets")
        
        if st.button("🎺 Jazz Swing (120)", use_container_width=True):
            st.session_state.metro_bpm = 120
            st.session_state.metro_beats = 4
            st.session_state.metro_swing_ratio = 67
            st.rerun()
        
        if st.button("🎸 Medium Swing (140)", use_container_width=True):
            st.session_state.metro_bpm = 140
            st.session_state.metro_beats = 4
            st.session_state.metro_swing_ratio = 67
            st.rerun()
        
        if st.button("🚀 Fast Bebop (200)", use_container_width=True):
            st.session_state.metro_bpm = 200
            st.session_state.metro_beats = 4
            st.session_state.metro_swing_ratio = 60
            st.rerun()
        
        if st.button("🎹 Ballad (60)", use_container_width=True):
            st.session_state.metro_bpm = 60
            st.session_state.metro_beats = 4
            st.session_state.metro_swing_ratio = 67
            st.rerun()
        
        if st.button("🌴 Bossa Nova (120)", use_container_width=True):
            st.session_state.metro_bpm = 120
            st.session_state.metro_beats = 4
            st.session_state.metro_swing_ratio = 50
            st.rerun()
        
        if st.button("🎵 Waltz 3/4 (120)", use_container_width=True):
            st.session_state.metro_bpm = 120
            st.session_state.metro_beats = 3
            st.session_state.metro_swing_ratio = 50
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("Practice Tips")
        st.info("""
        **Metronome Practice:**
        - Start slow, focus on accuracy
        - Gradually increase tempo
        - Practice with 2&4 clicks only
        - Try playing ahead/behind the beat
        """)
    
    st.markdown("---")
    
    metronome_html = Path(__file__).parent.parent / 'components' / 'metronome.html'
    
    with open(metronome_html, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    settings = {
        'bpm': st.session_state.metro_bpm,
        'beats': st.session_state.metro_beats,
        'subdivision': st.session_state.metro_subdivision,
        'accentBeats': st.session_state.metro_accent_beats,
        'swingRatio': st.session_state.metro_swing_ratio / 100.0,
        'clickOnBeats': click_on_beats if 'click_on_beats' in locals() else list(range(1, beats + 1))
    }
    
    html_with_settings = html_content.replace(
        'let bpm = 120;',
        f'let bpm = {settings["bpm"]};'
    ).replace(
        'let beatsPerMeasure = 4;',
        f'let beatsPerMeasure = {settings["beats"]};'
    ).replace(
        'let accentBeats = [1];',
        f'let accentBeats = {settings["accentBeats"]};'
    ).replace(
        'let clickOnBeats = [1, 2, 3, 4];',
        f'let clickOnBeats = {settings["clickOnBeats"]};'
    )
    
    components.html(html_with_settings, height=400)
    
    st.markdown("---")
    
    st.subheader("📚 Metronome Practice Guide")
    
    with st.expander("🎯 Why Practice with a Metronome?"):
        st.write("""
        A metronome is essential for developing:
        - **Steady time**: Maintain consistent tempo
        - **Internal pulse**: Develop your own sense of time
        - **Rhythmic accuracy**: Play exactly on the beat
        - **Groove**: Feel the pocket and swing
        
        Great jazz musicians have impeccable time. The metronome is your tool to develop this skill.
        """)
    
    with st.expander("🎼 Advanced Metronome Techniques"):
        st.write("""
        1. **Click on 2 & 4**: Simulates hi-hat in jazz (more challenging!)
        2. **Click on 1 only**: Forces you to maintain time internally
        3. **Slow practice**: Start at 60-80 BPM for difficult passages
        4. **Gradual increase**: Increase by 5 BPM when comfortable
        5. **Subdivision practice**: Feel 8th notes while click plays quarters
        6. **Polyrhythm**: Play triplets against quarter note clicks
        """)
    
    with st.expander("🎺 Swing Feel Explained"):
        st.write("""
        **Swing ratio** determines how uneven eighth notes are:
        
        - **50% (Straight)**: Even eighth notes (Latin, Bossa, Funk)
        - **60%**: Light swing (fast tempos, 200+ BPM)
        - **67%**: Standard swing (most common, 120-180 BPM)
        - **75%**: Heavy swing (slow blues, ballads)
        
        At faster tempos, swing naturally becomes straighter. At slower tempos, it becomes more pronounced.
        
        The swing feel comes from the triplet subdivision: the first note is 2/3 of the triplet, 
        the second is 1/3. This creates the characteristic "long-short" feel of jazz.
        """)


if __name__ == "__main__":
    main()
