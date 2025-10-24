"""
Chord Progression Practice Page - Practice jazz chord progressions
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
from datetime import datetime, date

sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    get_all_keys, transpose_progression, get_ii_v_i_progression,
    get_guide_tones, roman_to_chord, get_chord_scale_options, CHORD_INTERVALS
)


st.set_page_config(
    page_title="Chord Progressions - SAX Practice",
    page_icon="🎹",
    layout="wide"
)


def load_progression_data():
    progressions_file = Path(__file__).parent.parent / 'data' / 'theory' / 'progressions.yml'
    with open(progressions_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_progression_state():
    if 'prog_key' not in st.session_state:
        st.session_state.prog_key = 'C'
    if 'prog_type' not in st.session_state:
        st.session_state.prog_type = 'ii_v_i'
    if 'prog_bpm' not in st.session_state:
        st.session_state.prog_bpm = 120
    if 'prog_show_guide_tones' not in st.session_state:
        st.session_state.prog_show_guide_tones = True
    if 'prog_show_scales' not in st.session_state:
        st.session_state.prog_show_scales = True


def record_progression_practice(key, progression_type, bpm, duration_min):
    if 'practice_history' not in st.session_state:
        st.session_state.practice_history = []
    
    record = {
        'date': str(date.today()),
        'timestamp': datetime.now().isoformat(),
        'category': 'chord_progression',
        'key': key,
        'progression': progression_type,
        'bpm': bpm,
        'duration_min': duration_min,
        'success': 1.0
    }
    
    st.session_state.practice_history.append(record)
    st.session_state.total_practice_time = st.session_state.get('total_practice_time', 0) + duration_min
    
    from main import save_user_data
    save_user_data()


def convert_roman_to_chords(roman_numerals, key):
    chords = []
    for roman in roman_numerals:
        chord = roman_to_chord(roman, key)
        chords.append(chord)
    return chords


def main():
    init_progression_state()
    progression_data = load_progression_data()
    
    st.title("🎹 Chord Progression Practice")
    st.markdown("Master jazz chord progressions and improvisation")
    
    tab1, tab2, tab3 = st.tabs(["📝 Practice", "🎯 ii-V-I Master", "📊 Progress"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Progression Settings")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                key = st.selectbox("Key",
                                  options=get_all_keys(),
                                  index=get_all_keys().index(st.session_state.prog_key))
                st.session_state.prog_key = key
            
            with col_b:
                prog_types = list(progression_data['progressions'].keys())
                prog_display = [
                    f"{progression_data['progressions'][p]['name']} ({progression_data['progressions'][p]['name_ja']})"
                    for p in prog_types
                ]
                
                selected_index = prog_types.index(st.session_state.prog_type) if st.session_state.prog_type in prog_types else 0
                prog_selected = st.selectbox("Progression",
                                            options=prog_display,
                                            index=selected_index)
                prog_type = prog_types[prog_display.index(prog_selected)]
                st.session_state.prog_type = prog_type
            
            prog_info = progression_data['progressions'][prog_type]
            st.info(f"**Description:** {prog_info['description']}\n\n{prog_info['description_ja']}")
            
            st.markdown("---")
            
            bpm = st.slider("Practice BPM",
                          min_value=40, max_value=300,
                          value=st.session_state.prog_bpm,
                          step=5)
            st.session_state.prog_bpm = bpm
            
            suggested_bpm = prog_info.get('tempo', 120)
            if abs(bpm - suggested_bpm) > 20:
                st.caption(f"💡 Suggested tempo: {suggested_bpm} BPM ({prog_info.get('style', 'swing')})")
            
            st.markdown("---")
            
            st.subheader("Chord Progression")
            
            roman_form = prog_info['form']
            actual_chords = convert_roman_to_chords(roman_form, key)
            
            bars = prog_info.get('bars', len(roman_form))
            
            st.write(f"**Form:** {bars} bars")
            st.write(f"**Style:** {prog_info.get('style', 'swing').title()}")
            
            st.markdown("### Chords")
            
            cols_per_row = 4
            for i in range(0, len(actual_chords), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(actual_chords):
                        chord = actual_chords[i + j]
                        roman = roman_form[i + j]
                        with col:
                            st.markdown(f"**Bar {i+j+1}**")
                            st.code(chord, language=None)
                            st.caption(f"({roman})")
            
            st.markdown("---")
            
            show_guide = st.checkbox("Show Guide Tones", 
                                    value=st.session_state.prog_show_guide_tones)
            st.session_state.prog_show_guide_tones = show_guide
            
            show_scales = st.checkbox("Show Recommended Scales",
                                     value=st.session_state.prog_show_scales)
            st.session_state.prog_show_scales = show_scales
            
            if show_guide or show_scales:
                st.markdown("---")
                st.subheader("Practice Guide")
                
                for i, (chord, roman) in enumerate(zip(actual_chords, roman_form)):
                    with st.expander(f"Bar {i+1}: {chord} ({roman})"):
                        root = chord.rstrip('0123456789abdefgilmnorstuvy#b+-')
                        chord_type = chord[len(root):]
                        
                        if show_guide:
                            try:
                                guide_tones = get_guide_tones(root, chord_type if chord_type else 'maj7')
                                if guide_tones:
                                    st.write(f"**Guide Tones:** {' - '.join(guide_tones)}")
                            except:
                                st.write("**Guide Tones:** N/A")
                        
                        if show_scales:
                            try:
                                scales = get_chord_scale_options(chord_type if chord_type else 'maj7')
                                st.write(f"**Recommended Scales:** {', '.join(scales)}")
                            except:
                                st.write("**Recommended Scales:** Major")
            
            st.markdown("---")
            
            col_e, col_f, col_g = st.columns(3)
            
            with col_e:
                if st.button("✅ Practiced", use_container_width=True):
                    duration = (bars * 4 * 60) / bpm
                    record_progression_practice(key, prog_type, bpm, duration / 60)
                    st.success("Practice recorded!")
                    st.rerun()
            
            with col_f:
                if st.button("⭐ Mastered!", use_container_width=True):
                    duration = (bars * 4 * 60) / bpm
                    record_progression_practice(key, prog_type, bpm, duration / 60)
                    st.success("🎉 Mastered! Excellent!")
                    st.rerun()
            
            with col_g:
                if st.button("🔄 New Key", use_container_width=True):
                    all_keys = get_all_keys()
                    current_index = all_keys.index(st.session_state.prog_key)
                    next_index = (current_index + 1) % len(all_keys)
                    st.session_state.prog_key = all_keys[next_index]
                    st.rerun()
        
        with col2:
            st.subheader("Quick Progressions")
            
            quick_progs = ['ii_v_i', 'blues_12bar', 'turnaround', 'rhythm_changes_a']
            
            for prog in quick_progs:
                if prog in progression_data['progressions']:
                    prog_name = progression_data['progressions'][prog]['name']
                    if st.button(prog_name, use_container_width=True, key=f"quick_{prog}"):
                        st.session_state.prog_type = prog
                        st.rerun()
            
            st.markdown("---")
            
            st.subheader("Practice Tips")
            st.info("""
            **Progression Practice:**
            1. Learn the chords first
            2. Practice guide tones
            3. Add approach notes
            4. Use recommended scales
            5. Develop melodic ideas
            6. Record yourself
            """)
            
            st.markdown("---")
            
            st.subheader("Difficulty")
            difficulty = prog_info.get('difficulty', 1)
            st.write("⭐" * difficulty + "☆" * (5 - difficulty))
    
    with tab2:
        st.subheader("🎯 ii-V-I Master Class")
        st.markdown("The most important progression in jazz - practice in all 12 keys")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### What is ii-V-I?")
            st.write("""
            The ii-V-I progression is the backbone of jazz harmony. It appears in almost every jazz standard.
            
            In the key of C major:
            - **ii** = Dm7 (D minor 7th)
            - **V** = G7 (G dominant 7th)
            - **I** = Cmaj7 (C major 7th)
            """)
            
            st.markdown("---")
            
            selected_key = st.selectbox("Select Key for ii-V-I",
                                       options=get_all_keys(),
                                       key="ii_v_i_key")
            
            ii_v_i = get_ii_v_i_progression(selected_key)
            
            st.markdown("### Progression")
            cols = st.columns(3)
            for i, (chord, roman) in enumerate(zip(ii_v_i, ['ii', 'V', 'I'])):
                with cols[i]:
                    st.markdown(f"**{roman}**")
                    st.code(chord, language=None)
            
            st.markdown("---")
            
            st.markdown("### Guide Tones")
            st.write("""
            Guide tones (3rd and 7th of each chord) create smooth voice leading:
            """)
            
            for chord, roman in zip(ii_v_i, ['ii', 'V', 'I']):
                root = chord.rstrip('0123456789abdefgilmnorstuvy#b+-')
                chord_type = chord[len(root):]
                try:
                    guide_tones = get_guide_tones(root, chord_type)
                    st.write(f"**{chord}:** {' → '.join(guide_tones)}")
                except:
                    st.write(f"**{chord}:** N/A")
            
            st.markdown("---")
            
            st.markdown("### Recommended Scales")
            scales_for_ii_v_i = [
                ("ii (minor 7th)", "Dorian mode"),
                ("V (dominant 7th)", "Mixolydian or Bebop Dominant"),
                ("I (major 7th)", "Ionian (Major) or Lydian")
            ]
            
            for chord_type, scale in scales_for_ii_v_i:
                st.write(f"**{chord_type}:** {scale}")
            
            st.markdown("---")
            
            if st.button("✅ Practiced ii-V-I", use_container_width=True):
                record_progression_practice(selected_key, 'ii_v_i', st.session_state.prog_bpm, 1.0)
                st.success("Practice recorded!")
                st.rerun()
        
        with col2:
            st.subheader("All 12 Keys")
            
            all_keys = get_all_keys()
            
            for key in all_keys:
                ii_v_i = get_ii_v_i_progression(key)
                with st.expander(f"Key of {key}"):
                    for chord, roman in zip(ii_v_i, ['ii', 'V', 'I']):
                        st.write(f"**{roman}:** {chord}")
    
    with tab3:
        st.subheader("📊 Chord Progression Progress")
        
        if 'practice_history' in st.session_state and st.session_state.practice_history:
            prog_records = [r for r in st.session_state.practice_history 
                          if r.get('category') == 'chord_progression']
            
            if prog_records:
                st.write(f"**Total progression practice sessions:** {len(prog_records)}")
                
                total_time = sum(r.get('duration_min', 0) for r in prog_records)
                st.write(f"**Total practice time:** {total_time:.1f} minutes")
                
                keys_practiced = set(r.get('key') for r in prog_records)
                st.write(f"**Keys practiced:** {len(keys_practiced)} / 12")
                
                progressions_practiced = set(r.get('progression') for r in prog_records)
                st.write(f"**Different progressions practiced:** {len(progressions_practiced)}")
                
                st.markdown("---")
                
                st.subheader("Keys Practiced")
                all_keys = get_all_keys()
                
                cols = st.columns(6)
                for i, key in enumerate(all_keys):
                    with cols[i % 6]:
                        if key in keys_practiced:
                            st.success(f"✅ {key}")
                        else:
                            st.info(f"⭕ {key}")
                
                st.markdown("---")
                
                st.subheader("Recent Practice")
                recent_records = sorted(prog_records, 
                                      key=lambda x: x.get('timestamp', ''), 
                                      reverse=True)[:10]
                
                for record in recent_records:
                    st.write(f"**{record.get('key')} - {record.get('progression')}** - "
                           f"{record.get('bpm')} BPM - "
                           f"{record.get('date')}")
            else:
                st.info("No progression practice recorded yet. Start practicing!")
        else:
            st.info("No practice history available.")
    
    st.markdown("---")
    
    st.subheader("📚 Chord Progression Guide")
    
    with st.expander("🎯 Why Practice Chord Progressions?"):
        st.write("""
        Chord progressions are the harmonic framework of jazz:
        
        - **Harmonic Understanding**: Learn how chords move and resolve
        - **Improvisation**: Know what scales/notes work over each chord
        - **Voice Leading**: Create smooth melodic lines through changes
        - **Standards**: Most jazz standards use common progressions
        - **Ear Training**: Recognize progressions by ear
        
        Practicing progressions bridges the gap between scales and real music.
        """)
    
    with st.expander("🎼 Guide Tone Practice"):
        st.write("""
        **Guide tones** are the 3rd and 7th of each chord. They:
        
        - Define the chord quality (major, minor, dominant)
        - Create smooth voice leading between chords
        - Provide strong target notes for improvisation
        
        **Practice Method:**
        1. Play only guide tones through the progression
        2. Notice how they move (often by half-step or whole-step)
        3. Add approach notes (chromatic or diatonic)
        4. Expand to full improvisation
        
        **Example (ii-V-I in C):**
        - Dm7: F (3rd) → C (7th)
        - G7: B (3rd) → F (7th)
        - Cmaj7: E (3rd) → B (7th)
        
        Notice: C→B→B and F→F→E (smooth motion!)
        """)
    
    with st.expander("🎺 Common Jazz Progressions"):
        st.write("""
        **Essential Progressions to Master:**
        
        1. **ii-V-I**: The foundation (appears everywhere)
        2. **12-Bar Blues**: Classic form with variations
        3. **Rhythm Changes**: Based on "I Got Rhythm" (bebop favorite)
        4. **I-VI-ii-V**: Turnaround (ends phrases)
        5. **iii-VI-ii-V**: Extended turnaround
        6. **Coltrane Changes**: Advanced (Giant Steps)
        
        **Practice Strategy:**
        - Master ii-V-I in all 12 keys first
        - Add blues in all keys
        - Learn one standard per week
        - Transpose standards to different keys
        - Practice with backing tracks
        """)
    
    with st.expander("🎹 Chord-Scale Theory"):
        st.write("""
        **Which scale to use over each chord?**
        
        **Major 7th chords:**
        - Ionian (Major scale)
        - Lydian (raised 4th, brighter sound)
        
        **Minor 7th chords:**
        - Dorian (most common in jazz)
        - Aeolian (natural minor)
        - Phrygian (Spanish flavor)
        
        **Dominant 7th chords:**
        - Mixolydian (basic choice)
        - Bebop Dominant (adds chromatic passing tone)
        - Whole Tone (for altered sound)
        - Diminished Half-Whole (for 7b9, 7#9)
        
        **Half-Diminished (m7b5):**
        - Locrian
        - Locrian #2
        
        **Diminished 7th:**
        - Diminished Whole-Half scale
        """)


if __name__ == "__main__":
    main()
