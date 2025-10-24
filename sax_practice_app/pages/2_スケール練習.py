"""
Scale Practice Page - Comprehensive scale practice with all jazz scales
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
from datetime import datetime, date
import json

sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    generate_scale, generate_scale_pattern, get_all_keys,
    SCALE_INTERVALS, SRSItem
)


st.set_page_config(
    page_title="Scale Practice - SAX Practice",
    page_icon="🎼",
    layout="wide"
)


def load_scale_data():
    scales_file = Path(__file__).parent.parent / 'data' / 'theory' / 'scales.yml'
    with open(scales_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_scale_state():
    if 'scale_key' not in st.session_state:
        st.session_state.scale_key = 'C'
    if 'scale_type' not in st.session_state:
        st.session_state.scale_type = 'major'
    if 'scale_pattern' not in st.session_state:
        st.session_state.scale_pattern = 'ascending_descending'
    if 'scale_octaves' not in st.session_state:
        st.session_state.scale_octaves = 1
    if 'scale_bpm' not in st.session_state:
        st.session_state.scale_bpm = 80
    if 'scale_practice_mode' not in st.session_state:
        st.session_state.scale_practice_mode = 'manual'
    if 'scale_current_key_index' not in st.session_state:
        st.session_state.scale_current_key_index = 0
    if 'scale_completed_keys' not in st.session_state:
        st.session_state.scale_completed_keys = []
    if 'scale_practice_start_time' not in st.session_state:
        st.session_state.scale_practice_start_time = None


def record_scale_practice(key, scale_type, pattern, bpm, duration_min):
    if 'practice_history' not in st.session_state:
        st.session_state.practice_history = []
    
    record = {
        'date': str(date.today()),
        'timestamp': datetime.now().isoformat(),
        'category': 'scale',
        'key': key,
        'scale_type': scale_type,
        'pattern': pattern,
        'bpm': bpm,
        'duration_min': duration_min,
        'success': 1.0
    }
    
    st.session_state.practice_history.append(record)
    st.session_state.total_practice_time = st.session_state.get('total_practice_time', 0) + duration_min
    
    from main import save_user_data
    save_user_data()


def main():
    init_scale_state()
    scale_data = load_scale_data()
    
    st.title("🎼 Scale Practice")
    st.markdown("Master all scales in all 12 keys")
    
    tab1, tab2, tab3 = st.tabs(["📝 Practice", "🔄 12-Key Rotation", "📊 Progress"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Scale Settings")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                key = st.selectbox("Key", 
                                  options=get_all_keys(),
                                  index=get_all_keys().index(st.session_state.scale_key))
                st.session_state.scale_key = key
            
            with col_b:
                scale_types = list(scale_data['scales'].keys())
                scale_type_display = [
                    f"{scale_data['scales'][s]['name']} ({scale_data['scales'][s]['name_ja']})"
                    for s in scale_types
                ]
                
                selected_index = scale_types.index(st.session_state.scale_type) if st.session_state.scale_type in scale_types else 0
                scale_type_selected = st.selectbox("Scale Type",
                                                  options=scale_type_display,
                                                  index=selected_index)
                scale_type = scale_types[scale_type_display.index(scale_type_selected)]
                st.session_state.scale_type = scale_type
            
            scale_info = scale_data['scales'][scale_type]
            st.info(f"**Description:** {scale_info['description']}\n\n{scale_info['description_ja']}")
            
            st.markdown("---")
            
            col_c, col_d = st.columns(2)
            
            with col_c:
                pattern_types = list(scale_data['patterns'].keys())
                pattern_display = [
                    f"{scale_data['patterns'][p]['name']} ({scale_data['patterns'][p]['name_ja']})"
                    for p in pattern_types
                ]
                
                pattern_index = pattern_types.index(st.session_state.scale_pattern) if st.session_state.scale_pattern in pattern_types else 0
                pattern_selected = st.selectbox("Pattern",
                                               options=pattern_display,
                                               index=pattern_index)
                pattern = pattern_types[pattern_display.index(pattern_selected)]
                st.session_state.scale_pattern = pattern
            
            with col_d:
                octaves = st.selectbox("Octaves",
                                      options=[1, 2],
                                      index=st.session_state.scale_octaves - 1)
                st.session_state.scale_octaves = octaves
            
            pattern_info = scale_data['patterns'][pattern]
            st.caption(f"{pattern_info['description']} / {pattern_info['description_ja']}")
            
            st.markdown("---")
            
            bpm = st.slider("Practice BPM",
                          min_value=40, max_value=300,
                          value=st.session_state.scale_bpm,
                          step=5)
            st.session_state.scale_bpm = bpm
            
            target_bpm = st.session_state.get('target_bpm', 120)
            if bpm < target_bpm:
                progress = bpm / target_bpm
                st.progress(progress)
                st.caption(f"Progress to target: {bpm}/{target_bpm} BPM ({progress*100:.0f}%)")
            else:
                st.success(f"🎉 Target achieved! ({bpm} BPM)")
            
            st.markdown("---")
            
            st.subheader("Scale Notes")
            
            try:
                scale_notes = generate_scale_pattern(key, scale_type, pattern, octaves)
                
                notes_display = " → ".join(scale_notes)
                st.code(notes_display, language=None)
                
                st.caption(f"Total notes: {len(scale_notes)}")
                
                duration_seconds = (len(scale_notes) * 60) / bpm
                st.caption(f"Estimated duration at {bpm} BPM: {duration_seconds:.1f} seconds")
                
            except Exception as e:
                st.error(f"Error generating scale: {e}")
            
            st.markdown("---")
            
            col_e, col_f, col_g = st.columns(3)
            
            with col_e:
                if st.button("✅ Mark as Practiced", use_container_width=True):
                    record_scale_practice(key, scale_type, pattern, bpm, duration_seconds / 60)
                    st.success("Practice recorded!")
                    
                    item_id = f"scale_{key}_{scale_type}_{pattern}"
                    srs_manager = st.session_state.get('srs_manager')
                    if srs_manager:
                        if item_id not in srs_manager.items:
                            content = {
                                'key': key,
                                'scale_type': scale_type,
                                'pattern': pattern,
                                'octaves': octaves
                            }
                            srs_item = SRSItem(item_id, 'scale_pattern', content)
                            srs_manager.add_item(srs_item)
                        
                        srs_manager.update_item(item_id, quality=4)
                    
                    st.rerun()
            
            with col_f:
                if st.button("⭐ Mastered!", use_container_width=True):
                    record_scale_practice(key, scale_type, pattern, bpm, duration_seconds / 60)
                    
                    item_id = f"scale_{key}_{scale_type}_{pattern}"
                    srs_manager = st.session_state.get('srs_manager')
                    if srs_manager:
                        if item_id not in srs_manager.items:
                            content = {
                                'key': key,
                                'scale_type': scale_type,
                                'pattern': pattern,
                                'octaves': octaves
                            }
                            srs_item = SRSItem(item_id, 'scale_pattern', content)
                            srs_manager.add_item(srs_item)
                        
                        srs_manager.update_item(item_id, quality=5)
                    
                    st.success("🎉 Mastered! Great job!")
                    st.rerun()
            
            with col_g:
                if st.button("❌ Need More Practice", use_container_width=True):
                    item_id = f"scale_{key}_{scale_type}_{pattern}"
                    srs_manager = st.session_state.get('srs_manager')
                    if srs_manager:
                        if item_id not in srs_manager.items:
                            content = {
                                'key': key,
                                'scale_type': scale_type,
                                'pattern': pattern,
                                'octaves': octaves
                            }
                            srs_item = SRSItem(item_id, 'scale_pattern', content)
                            srs_manager.add_item(srs_item)
                        
                        srs_manager.update_item(item_id, quality=2)
                    
                    st.info("Marked for review. Keep practicing!")
                    st.rerun()
        
        with col2:
            st.subheader("Quick Keys")
            
            all_keys = get_all_keys()
            
            for i in range(0, len(all_keys), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(all_keys):
                        k = all_keys[i + j]
                        with col:
                            if st.button(k, use_container_width=True, key=f"key_{k}"):
                                st.session_state.scale_key = k
                                st.rerun()
            
            st.markdown("---")
            
            st.subheader("Practice Tips")
            st.info("""
            **Scale Practice:**
            1. Start slow (60-80 BPM)
            2. Focus on tone quality
            3. Use metronome
            4. Practice all 12 keys
            5. Vary patterns daily
            6. Record yourself
            """)
            
            st.markdown("---")
            
            st.subheader("Difficulty")
            difficulty = scale_info.get('difficulty', 1)
            st.write("⭐" * difficulty + "☆" * (5 - difficulty))
    
    with tab2:
        st.subheader("🔄 12-Key Rotation Practice")
        st.markdown("Practice the same scale in all 12 keys systematically")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            all_keys = get_all_keys()
            
            if st.button("🔄 Start 12-Key Rotation", use_container_width=True):
                st.session_state.scale_current_key_index = 0
                st.session_state.scale_completed_keys = []
                st.session_state.scale_practice_start_time = datetime.now()
            
            if st.session_state.scale_current_key_index < len(all_keys):
                current_key = all_keys[st.session_state.scale_current_key_index]
                
                st.markdown(f"### Current Key: **{current_key}**")
                st.markdown(f"Progress: {st.session_state.scale_current_key_index + 1} / {len(all_keys)}")
                
                progress_bar = st.progress(st.session_state.scale_current_key_index / len(all_keys))
                
                try:
                    scale_notes = generate_scale_pattern(current_key, st.session_state.scale_type, 
                                                        st.session_state.scale_pattern, 
                                                        st.session_state.scale_octaves)
                    
                    notes_display = " → ".join(scale_notes)
                    st.code(notes_display, language=None)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("✅ Next Key", use_container_width=True):
                        st.session_state.scale_completed_keys.append(current_key)
                        st.session_state.scale_current_key_index += 1
                        
                        duration_seconds = 30
                        record_scale_practice(current_key, st.session_state.scale_type, 
                                            st.session_state.scale_pattern, 
                                            st.session_state.scale_bpm, 
                                            duration_seconds / 60)
                        
                        if st.session_state.scale_current_key_index >= len(all_keys):
                            st.balloons()
                            st.success("🎉 Completed all 12 keys! Excellent work!")
                        
                        st.rerun()
                
                with col_b:
                    if st.button("⏭️ Skip Key", use_container_width=True):
                        st.session_state.scale_current_key_index += 1
                        st.rerun()
            
            else:
                st.success("🎉 All 12 keys completed!")
                
                if st.session_state.scale_practice_start_time:
                    elapsed = datetime.now() - st.session_state.scale_practice_start_time
                    st.write(f"Total time: {elapsed.seconds // 60} minutes {elapsed.seconds % 60} seconds")
                
                if st.button("🔄 Start Again", use_container_width=True):
                    st.session_state.scale_current_key_index = 0
                    st.session_state.scale_completed_keys = []
                    st.session_state.scale_practice_start_time = datetime.now()
                    st.rerun()
        
        with col2:
            st.subheader("Completed Keys")
            if st.session_state.scale_completed_keys:
                for key in st.session_state.scale_completed_keys:
                    st.success(f"✅ {key}")
            else:
                st.info("No keys completed yet")
            
            st.markdown("---")
            
            st.subheader("Key Order")
            st.caption("Circle of Fifths order:")
            st.code("C → G → D → A → E → B → Gb → Db → Ab → Eb → Bb → F")
    
    with tab3:
        st.subheader("📊 Scale Practice Progress")
        
        if 'practice_history' in st.session_state and st.session_state.practice_history:
            scale_records = [r for r in st.session_state.practice_history if r.get('category') == 'scale']
            
            if scale_records:
                st.write(f"**Total scale practice sessions:** {len(scale_records)}")
                
                total_time = sum(r.get('duration_min', 0) for r in scale_records)
                st.write(f"**Total practice time:** {total_time:.1f} minutes")
                
                keys_practiced = set(r.get('key') for r in scale_records)
                st.write(f"**Keys practiced:** {len(keys_practiced)} / 12")
                
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
                recent_records = sorted(scale_records, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
                
                for record in recent_records:
                    st.write(f"**{record.get('key')} {record.get('scale_type')}** - "
                           f"{record.get('bpm')} BPM - "
                           f"{record.get('pattern')} - "
                           f"{record.get('date')}")
            else:
                st.info("No scale practice recorded yet. Start practicing!")
        else:
            st.info("No practice history available. Start practicing to see your progress!")
    
    st.markdown("---")
    
    st.subheader("📚 Scale Practice Guide")
    
    with st.expander("🎯 Why Practice Scales?"):
        st.write("""
        Scales are the foundation of jazz improvisation:
        
        - **Technique**: Develop finger dexterity and muscle memory
        - **Ear Training**: Internalize the sound of each scale
        - **Theory**: Understand chord-scale relationships
        - **Vocabulary**: Scales provide raw material for improvisation
        - **All Keys**: Jazz musicians must be fluent in all 12 keys
        
        Every great jazz musician has spent countless hours practicing scales. 
        It's not glamorous, but it's essential.
        """)
    
    with st.expander("🎼 Scale Practice Routine"):
        st.write("""
        **Daily Scale Routine (20-30 minutes):**
        
        1. **Warm-up** (5 min)
           - Long tones on each note
           - Major scale in C, slow and steady
        
        2. **Major Scales** (10 min)
           - All 12 keys, ascending and descending
           - Start at 80 BPM, increase gradually
        
        3. **Modes** (5 min)
           - Dorian and Mixolydian (most common in jazz)
           - Focus on 2-3 keys per day
        
        4. **Blues Scales** (5 min)
           - Minor blues in all keys
           - Practice with blues feel
        
        5. **Patterns** (5 min)
           - Thirds, fourths, 1-2-3-5 pattern
           - Develops melodic vocabulary
        """)
    
    with st.expander("🎺 Advanced Scale Concepts"):
        st.write("""
        **Chord-Scale Relationships:**
        
        - **Cmaj7** → C Ionian (Major) or C Lydian
        - **Dm7** → D Dorian or D Aeolian
        - **G7** → G Mixolydian or G Bebop Dominant
        - **Bm7b5** → B Locrian or B Half-Diminished
        
        **Bebop Scales:**
        - Add chromatic passing tones to create 8-note scales
        - Helps land chord tones on downbeats
        - Essential for authentic bebop sound
        
        **Altered Scales:**
        - For altered dominant chords (7#9, 7b9, 7#5, 7b5)
        - Creates tension that resolves to tonic
        - Advanced technique for outside playing
        """)


if __name__ == "__main__":
    main()
