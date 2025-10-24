"""
Ear Training Page - Interval recognition and call & response
"""

import streamlit as st
from pathlib import Path
import sys
import random
from datetime import datetime, date

sys.path.append(str(Path(__file__).parent.parent))

from utils import get_interval_name, get_all_keys, generate_scale


st.set_page_config(
    page_title="Ear Training - SAX Practice",
    page_icon="👂",
    layout="wide"
)


def init_ear_training_state():
    if 'ear_current_interval' not in st.session_state:
        st.session_state.ear_current_interval = None
    if 'ear_score' not in st.session_state:
        st.session_state.ear_score = 0
    if 'ear_total' not in st.session_state:
        st.session_state.ear_total = 0
    if 'ear_show_answer' not in st.session_state:
        st.session_state.ear_show_answer = False
    if 'ear_call_phrase' not in st.session_state:
        st.session_state.ear_call_phrase = None
    if 'ear_response_given' not in st.session_state:
        st.session_state.ear_response_given = False


def record_ear_training(exercise_type, score, total, duration_min):
    if 'practice_history' not in st.session_state:
        st.session_state.practice_history = []
    
    record = {
        'date': str(date.today()),
        'timestamp': datetime.now().isoformat(),
        'category': 'ear_training',
        'exercise_type': exercise_type,
        'score': score,
        'total': total,
        'accuracy': score / total if total > 0 else 0,
        'duration_min': duration_min,
        'success': score / total if total > 0 else 0
    }
    
    st.session_state.practice_history.append(record)
    st.session_state.total_practice_time = st.session_state.get('total_practice_time', 0) + duration_min
    
    from main import save_user_data
    save_user_data()


def generate_new_interval():
    intervals = list(range(1, 13))
    st.session_state.ear_current_interval = random.choice(intervals)
    st.session_state.ear_show_answer = False


def generate_call_phrase():
    keys = get_all_keys()
    key = random.choice(keys)
    
    scale_types = ['major', 'dorian', 'minor_blues']
    scale_type = random.choice(scale_types)
    
    try:
        scale = generate_scale(key, scale_type, octaves=1)
        
        phrase_length = random.randint(4, 8)
        phrase = random.sample(scale, min(phrase_length, len(scale)))
        
        st.session_state.ear_call_phrase = {
            'key': key,
            'scale_type': scale_type,
            'phrase': phrase
        }
        st.session_state.ear_response_given = False
    except:
        st.session_state.ear_call_phrase = None


def main():
    init_ear_training_state()
    
    st.title("👂 Ear Training")
    st.markdown("Develop your musical ear with interval recognition and call & response")
    
    tab1, tab2, tab3 = st.tabs(["🎵 Interval Recognition", "🗣️ Call & Response", "📊 Progress"])
    
    with tab1:
        st.subheader("🎵 Interval Recognition")
        st.markdown("Learn to identify intervals by ear - essential for transcription and improvisation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### How to Practice")
            st.write("""
            1. Click "New Interval" to generate a random interval
            2. Sing or play the interval on your saxophone
            3. Try to identify the interval
            4. Click "Show Answer" to check
            5. Mark if you got it correct or not
            """)
            
            st.markdown("---")
            
            if st.button("🎲 New Interval", use_container_width=True):
                generate_new_interval()
                st.rerun()
            
            if st.session_state.ear_current_interval is not None:
                st.markdown("### Current Interval")
                
                if not st.session_state.ear_show_answer:
                    st.info("🎵 Listen and identify the interval...")
                    
                    if st.button("👁️ Show Answer", use_container_width=True):
                        st.session_state.ear_show_answer = True
                        st.rerun()
                else:
                    interval_name = get_interval_name(st.session_state.ear_current_interval)
                    st.success(f"**Answer: {interval_name}** ({st.session_state.ear_current_interval} semitones)")
                    
                    st.markdown("---")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("✅ Got it Right!", use_container_width=True):
                            st.session_state.ear_score += 1
                            st.session_state.ear_total += 1
                            generate_new_interval()
                            st.rerun()
                    
                    with col_b:
                        if st.button("❌ Got it Wrong", use_container_width=True):
                            st.session_state.ear_total += 1
                            generate_new_interval()
                            st.rerun()
            else:
                st.info("Click 'New Interval' to start practicing")
            
            st.markdown("---")
            
            st.markdown("### Your Score")
            if st.session_state.ear_total > 0:
                accuracy = (st.session_state.ear_score / st.session_state.ear_total) * 100
                st.metric("Accuracy", f"{accuracy:.1f}%")
                st.write(f"Correct: {st.session_state.ear_score} / {st.session_state.ear_total}")
                
                st.progress(st.session_state.ear_score / st.session_state.ear_total)
                
                if st.button("💾 Save Session", use_container_width=True):
                    record_ear_training('interval_recognition', 
                                      st.session_state.ear_score,
                                      st.session_state.ear_total,
                                      st.session_state.ear_total * 0.5)
                    st.success("Session saved!")
                    st.session_state.ear_score = 0
                    st.session_state.ear_total = 0
                    st.rerun()
            else:
                st.info("No attempts yet. Start practicing!")
        
        with col2:
            st.subheader("Interval Reference")
            
            intervals_ref = [
                (1, "Minor 2nd", "Jaws theme"),
                (2, "Major 2nd", "Happy Birthday"),
                (3, "Minor 3rd", "Greensleeves"),
                (4, "Major 3rd", "When the Saints"),
                (5, "Perfect 4th", "Here Comes the Bride"),
                (6, "Tritone", "The Simpsons"),
                (7, "Perfect 5th", "Star Wars theme"),
                (8, "Minor 6th", "The Entertainer"),
                (9, "Major 6th", "My Bonnie"),
                (10, "Minor 7th", "Star Trek theme"),
                (11, "Major 7th", "Take On Me"),
                (12, "Octave", "Somewhere Over the Rainbow"),
            ]
            
            for semitones, name, song in intervals_ref:
                with st.expander(f"{name} ({semitones})"):
                    st.write(f"**Reference:** {song}")
    
    with tab2:
        st.subheader("🗣️ Call & Response")
        st.markdown("Practice responding to musical phrases - develops your improvisational ear")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### How to Practice")
            st.write("""
            1. Click "New Call Phrase" to generate a random phrase
            2. Play the "call" phrase on your saxophone
            3. Improvise a "response" phrase that answers the call
            4. The response should be similar in length and style
            5. Mark when you've completed your response
            """)
            
            st.markdown("---")
            
            if st.button("🎲 New Call Phrase", use_container_width=True):
                generate_call_phrase()
                st.rerun()
            
            if st.session_state.ear_call_phrase is not None:
                call = st.session_state.ear_call_phrase
                
                st.markdown("### Call Phrase")
                st.write(f"**Key:** {call['key']}")
                st.write(f"**Scale:** {call['scale_type'].replace('_', ' ').title()}")
                
                phrase_display = " → ".join(call['phrase'])
                st.code(phrase_display, language=None)
                
                st.markdown("---")
                
                st.markdown("### Your Response")
                st.info("🎺 Play your response phrase now...")
                
                if not st.session_state.ear_response_given:
                    if st.button("✅ Response Complete", use_container_width=True):
                        st.session_state.ear_response_given = True
                        st.success("Great! Ready for the next phrase?")
                        st.rerun()
                else:
                    st.success("Response recorded!")
                    
                    if st.button("➡️ Next Phrase", use_container_width=True):
                        generate_call_phrase()
                        st.rerun()
            else:
                st.info("Click 'New Call Phrase' to start practicing")
            
            st.markdown("---")
            
            st.markdown("### Tips for Good Responses")
            st.write("""
            - **Mirror the rhythm**: Use similar rhythmic patterns
            - **Answer the question**: If call goes up, response might go down
            - **Use the same scale**: Stay in the same key and scale
            - **Similar length**: Match the length of the call phrase
            - **Add variation**: Don't just repeat - add your own ideas
            - **End strong**: Resolve to a strong note (root, 3rd, or 5th)
            """)
        
        with col2:
            st.subheader("Call & Response Concepts")
            
            with st.expander("🎵 What is Call & Response?"):
                st.write("""
                Call and response is a fundamental concept in jazz and African-American music:
                
                - One musician plays a phrase (the "call")
                - Another responds with an answering phrase
                - Creates musical conversation
                - Essential for playing in a group
                - Develops listening skills
                """)
            
            with st.expander("🎺 Practice Strategies"):
                st.write("""
                **Beginner:**
                - Repeat the call phrase exactly
                - Change just one or two notes
                - Use only scale notes
                
                **Intermediate:**
                - Vary the rhythm
                - Invert the melody (up becomes down)
                - Add chromatic passing tones
                
                **Advanced:**
                - Completely different melody
                - Change the harmony
                - Use outside notes
                - Create tension and release
                """)
    
    with tab3:
        st.subheader("📊 Ear Training Progress")
        
        if 'practice_history' in st.session_state and st.session_state.practice_history:
            ear_records = [r for r in st.session_state.practice_history 
                         if r.get('category') == 'ear_training']
            
            if ear_records:
                st.write(f"**Total ear training sessions:** {len(ear_records)}")
                
                total_time = sum(r.get('duration_min', 0) for r in ear_records)
                st.write(f"**Total practice time:** {total_time:.1f} minutes")
                
                interval_records = [r for r in ear_records 
                                  if r.get('exercise_type') == 'interval_recognition']
                
                if interval_records:
                    total_correct = sum(r.get('score', 0) for r in interval_records)
                    total_attempts = sum(r.get('total', 0) for r in interval_records)
                    
                    if total_attempts > 0:
                        overall_accuracy = (total_correct / total_attempts) * 100
                        st.write(f"**Overall interval accuracy:** {overall_accuracy:.1f}%")
                        st.progress(total_correct / total_attempts)
                
                st.markdown("---")
                
                st.subheader("Recent Sessions")
                recent_records = sorted(ear_records, 
                                      key=lambda x: x.get('timestamp', ''), 
                                      reverse=True)[:10]
                
                for record in recent_records:
                    exercise = record.get('exercise_type', 'unknown')
                    if exercise == 'interval_recognition':
                        score = record.get('score', 0)
                        total = record.get('total', 0)
                        accuracy = (score / total * 100) if total > 0 else 0
                        st.write(f"**Interval Recognition** - "
                               f"{score}/{total} ({accuracy:.1f}%) - "
                               f"{record.get('date')}")
                    else:
                        st.write(f"**{exercise}** - {record.get('date')}")
            else:
                st.info("No ear training sessions recorded yet. Start practicing!")
        else:
            st.info("No practice history available.")
    
    st.markdown("---")
    
    st.subheader("📚 Ear Training Guide")
    
    with st.expander("🎯 Why Ear Training?"):
        st.write("""
        Ear training is crucial for jazz musicians:
        
        - **Transcription**: Learn solos by ear from recordings
        - **Improvisation**: Play what you hear in your head
        - **Communication**: Respond to other musicians
        - **Chord Recognition**: Identify progressions by ear
        - **Intonation**: Play in tune with better pitch awareness
        
        Great jazz musicians have exceptional ears. Charlie Parker, John Coltrane, 
        and other masters could transcribe anything they heard.
        """)
    
    with st.expander("🎼 Interval Practice Tips"):
        st.write("""
        **How to Practice Intervals:**
        
        1. **Sing First**: Always sing intervals before playing them
        2. **Use References**: Associate each interval with a familiar song
        3. **Both Directions**: Practice ascending and descending
        4. **Context**: Practice intervals within scales and chords
        5. **Daily Practice**: 10-15 minutes every day is better than long sessions
        
        **Common Interval Mistakes:**
        - Confusing major 3rd with perfect 4th
        - Mixing up major 6th and minor 7th
        - Tritone can be tricky (exactly half an octave)
        
        **Advanced Practice:**
        - Identify intervals in real music
        - Transcribe melodies by ear
        - Identify chord qualities (major, minor, dominant)
        """)
    
    with st.expander("🗣️ Call & Response in Jazz"):
        st.write("""
        **Historical Context:**
        
        Call and response comes from African musical traditions and is fundamental to jazz:
        - Blues: Vocal line answered by guitar
        - Big Band: Saxes call, brass respond
        - Bebop: Trading fours (4-bar phrases)
        - Modern Jazz: Free-form conversation
        
        **Famous Examples:**
        - Miles Davis & John Coltrane on "So What"
        - Dizzy Gillespie & Charlie Parker trading phrases
        - Any jazz jam session!
        
        **Practice with Recordings:**
        1. Play along with recordings
        2. Pause after a phrase and respond
        3. Try to match the style and energy
        4. Gradually develop your own voice
        """)


if __name__ == "__main__":
    main()
