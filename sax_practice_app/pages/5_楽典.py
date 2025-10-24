"""
Music Theory (楽典) Page - Comprehensive jazz theory reference
"""

import streamlit as st
from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    get_all_keys, generate_chord, SCALE_INTERVALS, CHORD_INTERVALS,
    get_chord_scale_options, get_guide_tones
)


st.set_page_config(
    page_title="Music Theory - SAX Practice",
    page_icon="📚",
    layout="wide"
)


def load_theory_data():
    scales_file = Path(__file__).parent.parent / 'data' / 'theory' / 'scales.yml'
    progressions_file = Path(__file__).parent.parent / 'data' / 'theory' / 'progressions.yml'
    
    with open(scales_file, 'r', encoding='utf-8') as f:
        scales_data = yaml.safe_load(f)
    
    with open(progressions_file, 'r', encoding='utf-8') as f:
        progressions_data = yaml.safe_load(f)
    
    return scales_data, progressions_data


def main():
    st.title("📚 Music Theory (楽典)")
    st.markdown("Comprehensive jazz theory reference and learning guide")
    
    scales_data, progressions_data = load_theory_data()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎵 Basics", "🎼 Scales", "🎹 Chords", "🔄 Progressions", 
        "🎺 Jazz Concepts", "📖 Glossary"
    ])
    
    with tab1:
        st.header("🎵 Music Theory Basics")
        
        with st.expander("🎹 Notes and Pitch"):
            st.markdown("""
            Western music uses 12 notes (chromatic scale):
            
            **C - C# - D - D# - E - F - F# - G - G# - A - A# - B**
            
            Or with flats:
            
            **C - Db - D - Eb - E - F - Gb - G - Ab - A - Bb - B**
            
            - C# = Db (same pitch, different name)
            - D# = Eb
            - F# = Gb
            - G# = Ab
            - A# = Bb
            
            After B comes C again, one octave higher. An octave is 12 semitones.
            """)
        
        with st.expander("📏 Intervals"):
            st.markdown("""
            An interval is the distance between two notes.
            
            | Semitones | Name | Example (from C) |
            |-----------|------|------------------|
            | 0 | Unison | C - C |
            | 1 | Minor 2nd | C - Db |
            | 2 | Major 2nd | C - D |
            | 3 | Minor 3rd | C - Eb |
            | 4 | Major 3rd | C - E |
            | 5 | Perfect 4th | C - F |
            | 6 | Tritone (Aug 4th/Dim 5th) | C - F# |
            | 7 | Perfect 5th | C - G |
            | 8 | Minor 6th | C - Ab |
            | 9 | Major 6th | C - A |
            | 10 | Minor 7th | C - Bb |
            | 11 | Major 7th | C - B |
            | 12 | Octave | C - C |
            
            - **Perfect**: Unison, 4th, 5th, Octave
            - **Major/Minor**: 2nd, 3rd, 6th, 7th
            - **Augmented**: Raised by half-step
            - **Diminished**: Lowered by half-step
            """)
        
        with st.expander("🎼 Key Signatures"):
            st.markdown("""
            A key defines which notes are natural (white keys) and which are sharp/flat.
            
            Moving clockwise adds sharps, counterclockwise adds flats:
            
            **C → G → D → A → E → B → F#/Gb → Db → Ab → Eb → Bb → F → C**
            
            - **C Major**: No sharps or flats
            - **G Major**: 1 sharp (F#)
            - **D Major**: 2 sharps (F#, C#)
            - **A Major**: 3 sharps (F#, C#, G#)
            - **F Major**: 1 flat (Bb)
            - **Bb Major**: 2 flats (Bb, Eb)
            - **Eb Major**: 3 flats (Bb, Eb, Ab)
            
            Every major key has a relative minor (starts on the 6th degree):
            - C Major → A minor
            - G Major → E minor
            - F Major → D minor
            """)
        
        with st.expander("🎷 Transposing Instruments"):
            st.markdown("""
            
            **Alto Saxophone (Eb):**
            - Sounds a major 6th (9 semitones) lower than written
            - Written C sounds as Eb
            - To play in concert C, read in A
            
            **Tenor Saxophone (Bb):**
            - Sounds a major 9th (14 semitones) lower than written
            - Written C sounds as Bb (one octave lower)
            - To play in concert C, read in D
            
            **Soprano Saxophone (Bb):**
            - Sounds a major 2nd (2 semitones) lower than written
            - Written C sounds as Bb
            - To play in concert C, read in D
            
            **Baritone Saxophone (Eb):**
            - Sounds a major 13th (21 semitones) lower than written
            - Written C sounds as Eb (one octave lower)
            - To play in concert C, read in A
            
            - Easier fingerings across different instruments
            - Consistent notation for saxophone family
            - Must transpose when playing with concert pitch instruments (piano, guitar, bass)
            """)
        
        with st.expander("⏱️ Rhythm and Time"):
            st.markdown("""
            - **Whole note**: 4 beats
            - **Half note**: 2 beats
            - **Quarter note**: 1 beat
            - **Eighth note**: 1/2 beat
            - **Sixteenth note**: 1/4 beat
            
            - **4/4**: 4 quarter notes per measure (most common)
            - **3/4**: 3 quarter notes per measure (waltz)
            - **6/8**: 6 eighth notes per measure (compound meter)
            - **5/4**: 5 quarter notes per measure (Take Five)
            - **7/4**: 7 quarter notes per measure (advanced)
            
            In jazz, eighth notes are played unevenly:
            - First eighth note is longer (2/3 of beat)
            - Second eighth note is shorter (1/3 of beat)
            - Creates the characteristic "swing" feel
            - Ratio varies with tempo (67:33 is standard)
            """)
    
    with tab2:
        st.header("🎼 Scales")
        
        st.markdown("""
        Scales are the foundation of melody and improvisation. 
        Master these scales in all 12 keys for complete fluency.
        """)
        
        scale_categories = {
            "Major Scales": ['major', 'ionian', 'lydian'],
            "Minor Scales": ['natural_minor', 'harmonic_minor', 'melodic_minor', 'aeolian'],
            "Modes": ['dorian', 'phrygian', 'mixolydian', 'locrian'],
            "Blues Scales": ['major_blues', 'minor_blues'],
            "Bebop Scales": ['bebop_major', 'bebop_dominant'],
            "Symmetrical Scales": ['whole_tone', 'diminished_half_whole', 'diminished_whole_half']
        }
        
        for category, scale_list in scale_categories.items():
            with st.expander(f"📖 {category}"):
                for scale_type in scale_list:
                    if scale_type in scales_data['scales']:
                        scale_info = scales_data['scales'][scale_type]
                        st.markdown(f"### {scale_info['name']} ({scale_info['name_ja']})")
                        st.write(f"**Intervals:** {scale_info['intervals']}")
                        st.write(f"**Description:** {scale_info['description']}")
                        st.write(f"**日本語:** {scale_info['description_ja']}")
                        st.write(f"**Difficulty:** {'⭐' * scale_info.get('difficulty', 1)}")
                        
                        example_key = 'C'
                        intervals = scale_info['intervals']
                        notes = []
                        note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
                        for interval in intervals:
                            notes.append(note_names[interval])
                        
                        st.code(f"Example in C: {' - '.join(notes)}", language=None)
                        st.markdown("---")
        
        st.markdown("---")
        
        st.subheader("🎯 Scale Practice Tips")
        st.info("""
        1. **Start with Major**: Master major scales in all 12 keys first
        2. **Add Modes**: Learn Dorian and Mixolydian (most common in jazz)
        3. **Blues Scales**: Essential for jazz vocabulary
        4. **Bebop Scales**: Add chromatic passing tones for authentic sound
        5. **Practice Patterns**: Don't just play up and down - use thirds, fourths, sequences
        6. **All Keys**: True mastery means fluency in all 12 keys
        7. **Use Metronome**: Start slow, increase gradually
        8. **Sing First**: Always sing scales before playing them
        """)
    
    with tab3:
        st.header("🎹 Chords")
        
        st.markdown("""
        Chords are the harmonic foundation of jazz. Understanding chord construction 
        and function is essential for improvisation.
        """)
        
        with st.expander("🎵 Triads (3-note chords)"):
            st.markdown("""
            - **Formula**: Root - Major 3rd - Perfect 5th
            - **Intervals**: 0 - 4 - 7 semitones
            - **Example (C)**: C - E - G
            - **Sound**: Happy, stable
            
            - **Formula**: Root - Minor 3rd - Perfect 5th
            - **Intervals**: 0 - 3 - 7 semitones
            - **Example (C)**: C - Eb - G
            - **Sound**: Sad, dark
            
            - **Formula**: Root - Minor 3rd - Diminished 5th
            - **Intervals**: 0 - 3 - 6 semitones
            - **Example (C)**: C - Eb - Gb
            - **Sound**: Tense, unstable
            
            - **Formula**: Root - Major 3rd - Augmented 5th
            - **Intervals**: 0 - 4 - 8 semitones
            - **Example (C)**: C - E - G#
            - **Sound**: Mysterious, floating
            """)
        
        with st.expander("🎼 Seventh Chords (4-note chords)"):
            st.markdown("""
            - **Formula**: Root - Major 3rd - Perfect 5th - Major 7th
            - **Intervals**: 0 - 4 - 7 - 11
            - **Example (C)**: C - E - G - B
            - **Use**: I chord, IV chord in major keys
            - **Sound**: Bright, jazzy, sophisticated
            
            - **Formula**: Root - Minor 3rd - Perfect 5th - Minor 7th
            - **Intervals**: 0 - 3 - 7 - 10
            - **Example (C)**: C - Eb - G - Bb
            - **Use**: ii chord, iii chord, vi chord
            - **Sound**: Mellow, smooth
            
            - **Formula**: Root - Major 3rd - Perfect 5th - Minor 7th
            - **Intervals**: 0 - 4 - 7 - 10
            - **Example (C)**: C - E - G - Bb
            - **Use**: V chord, creates tension that resolves
            - **Sound**: Bluesy, wants to resolve
            
            - **Formula**: Root - Minor 3rd - Diminished 5th - Minor 7th
            - **Intervals**: 0 - 3 - 6 - 10
            - **Example (C)**: C - Eb - Gb - Bb
            - **Use**: vii chord in major, ii chord in minor
            - **Sound**: Dark, jazzy
            
            - **Formula**: Root - Minor 3rd - Diminished 5th - Diminished 7th
            - **Intervals**: 0 - 3 - 6 - 9
            - **Example (C)**: C - Eb - Gb - Bbb (A)
            - **Use**: Passing chord, creates tension
            - **Sound**: Very tense, symmetrical
            """)
        
        with st.expander("🎹 Extended Chords (Tensions)"):
            st.markdown("""
            Add the 9th (2nd octave higher):
            - **Major 9th (maj9)**: Cmaj7 + D
            - **Minor 9th (m9)**: Cm7 + D
            - **Dominant 9th (9)**: C7 + D
            
            Add the 11th (4th octave higher):
            - Usually avoid natural 11th on major chords (clashes with 3rd)
            - **#11**: Raised 11th, very common in jazz (Lydian sound)
            
            Add the 13th (6th octave higher):
            - **13th chord**: Contains 7th, 9th, and 13th
            - Very rich, full sound
            
            - **b9**: Flat 9th (very tense, blues sound)
            - **#9**: Sharp 9th (Hendrix chord)
            - **#11**: Sharp 11th (Lydian, bright)
            - **b13**: Flat 13th (dark, minor sound)
            
            - **7(#11)**: Dominant with raised 11th
            - **7(b9)**: Dominant with flat 9th (very tense)
            - **7(#9)**: Dominant with sharp 9th
            - **7alt**: Altered dominant (b9, #9, b5, #5)
            """)
        
        st.markdown("---")
        
        st.subheader("🎯 Chord Practice")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Chord Builder")
            
            selected_key = st.selectbox("Root Note", options=get_all_keys(), key="chord_root")
            
            chord_types = list(CHORD_INTERVALS.keys())
            selected_chord_type = st.selectbox("Chord Type", options=chord_types, key="chord_type")
            
            try:
                chord_notes = generate_chord(selected_key, selected_chord_type)
                st.success(f"**{selected_key}{selected_chord_type}**: {' - '.join(chord_notes)}")
                
                guide_tones = get_guide_tones(selected_key, selected_chord_type)
                if guide_tones:
                    st.info(f"**Guide Tones**: {' - '.join(guide_tones)}")
                
                scales = get_chord_scale_options(selected_chord_type)
                st.write(f"**Recommended Scales**: {', '.join(scales)}")
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col2:
            st.markdown("### Chord Function")
            st.info("""
            **Tonic (I)**: Home, stable
            - Major 7th chords
            - Feels resolved
            
            **Subdominant (IV)**: Preparation
            - Major 7th or minor 7th
            - Moves away from tonic
            
            **Dominant (V)**: Tension
            - Dominant 7th chords
            - Wants to resolve to tonic
            
            **Pre-Dominant (ii)**: Setup
            - Minor 7th chords
            - Leads to dominant
            """)
    
    with tab4:
        st.header("🔄 Chord Progressions")
        
        st.markdown("""
        Chord progressions are sequences of chords that create harmonic movement.
        Understanding common progressions is key to jazz improvisation.
        """)
        
        with st.expander("🎵 ii-V-I: The Foundation"):
            st.markdown("""
            
            The ii-V-I progression is everywhere in jazz. Master this in all 12 keys.
            
            **In C Major:**
            - **ii**: Dm7 (D Dorian)
            - **V**: G7 (G Mixolydian)
            - **I**: Cmaj7 (C Ionian)
            
            Guide tones move smoothly:
            - Dm7: F (3rd) → C (7th)
            - G7: B (3rd) → F (7th)
            - Cmaj7: E (3rd) → B (7th)
            
            Notice: C→B→B and F→F→E (half-step motion!)
            
            - **ii-V**: Without resolution (common in bebop)
            - **iii-VI-ii-V-I**: Extended turnaround
            - **ii-V-I-VI**: With deceptive resolution
            - **Tritone Substitution**: Replace V with bII7
            
            1. Play guide tones only
            2. Add approach notes (chromatic)
            3. Use recommended scales
            4. Develop melodic ideas
            5. Practice in all 12 keys
            """)
        
        with st.expander("🎸 Blues Progression"):
            st.markdown("""
            
            The blues is fundamental to jazz. Every jazz musician must know the blues.
            
            **Basic Blues in C:**
            ```
            | C7  | C7  | C7  | C7  |
            | F7  | F7  | C7  | C7  |
            | G7  | F7  | C7  | G7  |
            ```
            
            
            **Bebop Blues:**
            ```
            | Cmaj7 | Fm7 Bb7 | Cmaj7 | Cm7 F7  |
            | Fm7   | Bb7     | Cmaj7 | Em7 A7  |
            | Dm7   | G7      | Cmaj7 | Dm7 G7  |
            ```
            
            **Minor Blues:**
            ```
            | Cm7 | Cm7 | Cm7 | Cm7 |
            | Fm7 | Fm7 | Cm7 | Cm7 |
            | G7  | Fm7 | Cm7 | G7  |
            ```
            
            Essential for blues improvisation:
            - **Minor Blues**: C - Eb - F - Gb - G - Bb
            - **Major Blues**: C - D - Eb - E - G - A
            
            - Bend notes (especially 3rd and 5th)
            - Use blue notes (b3, b5, b7)
            - Call and response phrasing
            - Repetition with variation
            """)
        
        with st.expander("🎹 Rhythm Changes"):
            st.markdown("""
            
            Rhythm changes are a common bebop progression based on Gershwin's "I Got Rhythm".
            
            **A Section (8 bars):**
            ```
            | Cmaj7 Am7 | Dm7 G7 | Cmaj7 Am7 | Dm7 G7 |
            | Cmaj7 Am7 | Dm7 G7 | Cmaj7    | Dm7 G7 |
            ```
            
            **B Section (Bridge, 8 bars):**
            ```
            | D7  | D7  | G7  | G7  |
            | C7  | C7  | F7  | F7  |
            ```
            
            **Form**: AABA (32 bars total)
            
            - "Oleo" (Sonny Rollins)
            - "Anthropology" (Charlie Parker)
            - "Moose the Mooche" (Charlie Parker)
            - "Rhythm-a-ning" (Thelonious Monk)
            
            - Very fast tempo (200+ BPM)
            - Focus on guide tones
            - Use bebop scales
            - Practice the bridge separately
            """)
        
        with st.expander("🎺 Common Substitutions"):
            st.markdown("""
            Replace a dominant chord with another dominant a tritone away:
            - **G7 → Db7** (in ii-V-I)
            - Same guide tones (enharmonically)
            - Creates chromatic bass motion
            
            **Example:**
            - Original: Dm7 - G7 - Cmaj7
            - Substituted: Dm7 - Db7 - Cmaj7
            
            Use diminished chords as passing chords:
            - Between any two chords a whole step apart
            - Creates chromatic motion
            
            Borrow chords from parallel minor:
            - In C major, use chords from C minor
            - Example: Fm7 in C major (from C minor)
            
            Turn any chord into a temporary V7:
            - To get to Dm7, use A7 (V7 of Dm)
            - Creates stronger resolution
            """)
    
    with tab5:
        st.header("🎺 Jazz Concepts")
        
        with st.expander("🎵 Swing Feel"):
            st.markdown("""
            
            Swing is the rhythmic feel that defines jazz. Eighth notes are played unevenly.
            
            **Straight 8ths** (Rock, Pop):
            ```
            ♪ ♪ ♪ ♪ = Even, equal length
            ```
            
            **Swing 8ths** (Jazz):
            ```
            ♪ ♪ ♪ ♪ = Long-short, long-short
            ```
            
            - **50:50** = Straight (no swing)
            - **60:40** = Light swing (fast tempos)
            - **67:33** = Standard swing (medium tempos)
            - **75:25** = Heavy swing (slow blues)
            
            1. Listen to great jazz recordings
            2. Tap your foot on beats 2 and 4
            3. Feel the triplet subdivision
            4. Don't overthink it - let it groove
            5. Play with other musicians
            
            - Slight accent on upbeats
            - Relaxed, not stiff
            - "Lay back" on the beat
            - Use ghost notes
            """)
        
        with st.expander("🎼 Bebop Language"):
            st.markdown("""
            
            Bebop is a jazz style developed in the 1940s characterized by:
            - Fast tempos
            - Complex harmony
            - Virtuosic improvisation
            - Emphasis on ii-V-I progressions
            
            8-note scales with chromatic passing tones:
            
            **Bebop Major:**
            - C D E F G Ab A B C
            - Adds b6 between 5 and 6
            
            **Bebop Dominant:**
            - C D E F G A Bb B C
            - Adds major 7 between b7 and root
            
            - Chord tones land on downbeats
            - Creates smooth, flowing lines
            - Authentic bebop sound
            
            - **Enclosures**: Surround target notes
            - **Approach notes**: Chromatic or diatonic
            - **Arpeggios**: Outline chord changes
            - **Sequences**: Repeat patterns
            - **Chromaticism**: Use passing tones
            
            - Charlie Parker (alto sax)
            - Dizzy Gillespie (trumpet)
            - Bud Powell (piano)
            - Max Roach (drums)
            """)
        
        with st.expander("🎹 Modal Jazz"):
            st.markdown("""
            
            Modal jazz uses modes (scales) as the basis for improvisation, 
            rather than chord progressions.
            
            **Characteristics:**
            - Fewer chord changes
            - Longer time on each chord
            - Focus on melodic development
            - More open, spacious sound
            
            - "So What" (Miles Davis) - D Dorian
            - "Impressions" (John Coltrane) - D Dorian
            - "Maiden Voyage" (Herbie Hancock) - Multiple modes
            
            From C major scale:
            1. **Ionian** (C): C D E F G A B
            2. **Dorian** (D): D E F G A B C
            3. **Phrygian** (E): E F G A B C D
            4. **Lydian** (F): F G A B C D E
            5. **Mixolydian** (G): G A B C D E F
            6. **Aeolian** (A): A B C D E F G
            7. **Locrian** (B): B C D E F G A
            
            - Stay in one mode for extended time
            - Develop motifs and ideas
            - Use mode's characteristic notes
            - Create tension and release within mode
            - Less is more - space is important
            """)
        
        with st.expander("🎺 Improvisation Concepts"):
            st.markdown("""
            
            **1. Scales**
            - Know all scales in all keys
            - Match scales to chords
            - Use appropriate modes
            
            **2. Arpeggios**
            - Outline chord changes
            - Use guide tones
            - Add extensions (9, 11, 13)
            
            **3. Patterns**
            - Bebop patterns
            - Sequences
            - Rhythmic motifs
            
            **4. Vocabulary**
            - Learn licks from masters
            - Transcribe solos
            - Build your own phrases
            
            
            **Beginner:**
            - Use chord tones only
            - Simple rhythms
            - Stay close to melody
            
            **Intermediate:**
            - Add approach notes
            - Use guide tones
            - Develop motifs
            - Play through changes
            
            **Advanced:**
            - Outside playing
            - Superimposition
            - Rhythmic displacement
            - Personal voice
            
            1. **Transcription**: Learn solos note-for-note
            2. **Singing**: Sing what you want to play
            3. **Call & Response**: Answer musical phrases
            4. **Limitations**: Practice with constraints
            5. **Recording**: Record and analyze yourself
            """)
    
    with tab6:
        st.header("📖 Jazz Glossary")
        
        glossary = {
            "Altered Chord": "Dominant chord with altered tensions (b9, #9, b5, #5)",
            "Approach Note": "Note that leads into a target note (chromatic or diatonic)",
            "Arpeggio": "Playing the notes of a chord one at a time",
            "Bebop": "Jazz style from 1940s with fast tempos and complex harmony",
            "Blue Note": "Flatted 3rd, 5th, or 7th that creates blues sound",
            "Changes": "Chord progression of a tune",
            "Chorus": "One complete cycle through a tune's form",
            "Comping": "Accompanying, playing chords behind a soloist",
            "Dorian": "Minor mode with raised 6th, most common for minor chords",
            "Drop 2": "Voicing where 2nd note from top is dropped an octave",
            "Enclosure": "Surrounding a target note from above and below",
            "Guide Tones": "3rd and 7th of a chord, define chord quality",
            "Head": "The main melody of a tune",
            "ii-V-I": "Most common progression in jazz (minor 7th, dominant 7th, major 7th)",
            "Lay Back": "Playing slightly behind the beat for relaxed feel",
            "Lick": "A short musical phrase or pattern",
            "Mixolydian": "Major mode with flatted 7th, used for dominant chords",
            "Modal": "Based on modes rather than chord changes",
            "Motif": "Short musical idea that is developed",
            "Outside": "Playing notes outside the key/chord for tension",
            "Passing Tone": "Non-chord tone that connects two chord tones",
            "Pocket": "The groove, the rhythmic feel",
            "Rhythm Changes": "Progression based on 'I Got Rhythm'",
            "Shell Voicing": "Root, 3rd, and 7th only (minimal voicing)",
            "Standard": "Well-known jazz composition",
            "Substitution": "Replacing a chord with another chord",
            "Swing": "Uneven eighth notes that create jazz feel",
            "Target Note": "Goal note, usually a chord tone",
            "Trading": "Taking turns soloing (trading fours = 4 bars each)",
            "Tritone": "Interval of 6 semitones (augmented 4th)",
            "Tritone Sub": "Replacing dominant with dominant a tritone away",
            "Turnaround": "Progression that returns to the beginning (I-VI-ii-V)",
            "Voice Leading": "Smooth motion between chord tones",
            "Walking Bass": "Bass line that walks through chord changes",
        }
        
        st.markdown("### Common Jazz Terms")
        
        for term, definition in sorted(glossary.items()):
            with st.expander(f"**{term}**"):
                st.write(definition)
    
    st.markdown("---")
    
    st.info("""
    💡 **Study Tip**: Music theory is best learned by doing. 
    Practice these concepts on your saxophone, not just in your head. 
    Theory should always serve the music, not the other way around.
    """)


if __name__ == "__main__":
    main()
