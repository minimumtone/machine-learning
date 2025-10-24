"""
Statistics Page - Practice progress tracking and visualization
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime, date, timedelta
from collections import defaultdict
import json

sys.path.append(str(Path(__file__).parent.parent))


st.set_page_config(
    page_title="Statistics - SAX Practice",
    page_icon="📊",
    layout="wide"
)


def get_practice_stats():
    if 'practice_history' not in st.session_state or not st.session_state.practice_history:
        return None
    
    records = st.session_state.practice_history
    
    total_sessions = len(records)
    total_time = sum(r.get('duration_min', 0) for r in records)
    
    categories = defaultdict(int)
    for r in records:
        categories[r.get('category', 'unknown')] += 1
    
    dates = [r.get('date') for r in records if r.get('date')]
    unique_dates = set(dates)
    days_practiced = len(unique_dates)
    
    keys_practiced = set()
    for r in records:
        if r.get('key'):
            keys_practiced.add(r.get('key'))
    
    bpms = [r.get('bpm', 0) for r in records if r.get('bpm')]
    max_bpm = max(bpms) if bpms else 0
    avg_bpm = sum(bpms) / len(bpms) if bpms else 0
    
    return {
        'total_sessions': total_sessions,
        'total_time': total_time,
        'categories': dict(categories),
        'days_practiced': days_practiced,
        'keys_practiced': len(keys_practiced),
        'max_bpm': max_bpm,
        'avg_bpm': avg_bpm,
        'unique_dates': unique_dates
    }


def get_daily_practice_time():
    if 'practice_history' not in st.session_state or not st.session_state.practice_history:
        return {}
    
    daily_time = defaultdict(float)
    for record in st.session_state.practice_history:
        date_str = record.get('date')
        duration = record.get('duration_min', 0)
        if date_str:
            daily_time[date_str] += duration
    
    return dict(daily_time)


def get_category_breakdown():
    if 'practice_history' not in st.session_state or not st.session_state.practice_history:
        return {}
    
    category_time = defaultdict(float)
    for record in st.session_state.practice_history:
        category = record.get('category', 'unknown')
        duration = record.get('duration_min', 0)
        category_time[category] += duration
    
    return dict(category_time)


def get_weekly_stats():
    if 'practice_history' not in st.session_state or not st.session_state.practice_history:
        return {}
    
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    
    weekly_data = defaultdict(lambda: {'sessions': 0, 'time': 0})
    
    for record in st.session_state.practice_history:
        date_str = record.get('date')
        if date_str:
            record_date = date.fromisoformat(date_str)
            if record_date >= week_start:
                day_name = record_date.strftime('%A')
                weekly_data[day_name]['sessions'] += 1
                weekly_data[day_name]['time'] += record.get('duration_min', 0)
    
    return dict(weekly_data)


def get_bpm_progress():
    if 'practice_history' not in st.session_state or not st.session_state.practice_history:
        return []
    
    bpm_records = []
    for record in st.session_state.practice_history:
        if record.get('bpm') and record.get('date'):
            bpm_records.append({
                'date': record['date'],
                'bpm': record['bpm'],
                'category': record.get('category', 'unknown')
            })
    
    bpm_records.sort(key=lambda x: x['date'])
    return bpm_records


def main():
    st.title("📊 Practice Statistics")
    st.markdown("Track your progress and visualize your practice journey")
    
    stats = get_practice_stats()
    
    if not stats:
        st.info("No practice data available yet. Start practicing to see your statistics!")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📅 Calendar", "🎯 Goals", "📋 History"])
    
    with tab1:
        st.header("📈 Practice Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", stats['total_sessions'])
        
        with col2:
            st.metric("Total Time", f"{stats['total_time']:.1f} min")
        
        with col3:
            st.metric("Days Practiced", stats['days_practiced'])
        
        with col4:
            st.metric("Max BPM", int(stats['max_bpm']))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Practice by Category")
            
            category_time = get_category_breakdown()
            
            if category_time:
                category_labels = {
                    'scale': '🎼 Scales',
                    'chord_progression': '🎹 Chord Progressions',
                    'ear_training': '👂 Ear Training',
                    'metronome': '🎵 Metronome',
                    'unknown': '❓ Other'
                }
                
                for category, time in sorted(category_time.items(), key=lambda x: x[1], reverse=True):
                    label = category_labels.get(category, category)
                    percentage = (time / stats['total_time']) * 100 if stats['total_time'] > 0 else 0
                    st.write(f"{label}: {time:.1f} min ({percentage:.1f}%)")
                    st.progress(time / stats['total_time'] if stats['total_time'] > 0 else 0)
            else:
                st.info("No category data available")
        
        with col2:
            st.subheader("Keys Practiced")
            
            st.metric("Keys Mastered", f"{stats['keys_practiced']} / 12")
            
            progress = stats['keys_practiced'] / 12
            st.progress(progress)
            
            if stats['keys_practiced'] == 12:
                st.success("🎉 All 12 keys practiced! Excellent!")
            elif stats['keys_practiced'] >= 8:
                st.info(f"Great progress! {12 - stats['keys_practiced']} keys to go.")
            else:
                st.warning(f"Keep going! {12 - stats['keys_practiced']} more keys to practice.")
        
        st.markdown("---")
        
        st.subheader("BPM Progress")
        
        bpm_progress = get_bpm_progress()
        
        if bpm_progress:
            dates = [record['date'] for record in bpm_progress]
            bpms = [record['bpm'] for record in bpm_progress]
            
            st.line_chart(dict(zip(dates, bpms)))
            
            st.caption(f"Average BPM: {stats['avg_bpm']:.1f} | Max BPM: {stats['max_bpm']:.0f}")
        else:
            st.info("No BPM data available")
        
        st.markdown("---")
        
        st.subheader("Practice Streak")
        
        current_streak = st.session_state.get('current_streak', 0)
        st.metric("Current Streak", f"{current_streak} days")
        
        if current_streak >= 30:
            st.success("🔥 Amazing! 30+ day streak!")
        elif current_streak >= 7:
            st.info("💪 Great! One week streak!")
        elif current_streak >= 3:
            st.info("👍 Good! Keep it up!")
    
    with tab2:
        st.header("📅 Practice Calendar")
        
        daily_time = get_daily_practice_time()
        
        if daily_time:
            today = date.today()
            
            st.subheader("Last 30 Days")
            
            days_to_show = 30
            date_range = [today - timedelta(days=i) for i in range(days_to_show)]
            date_range.reverse()
            
            cols_per_row = 7
            for i in range(0, len(date_range), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(date_range):
                        day = date_range[i + j]
                        day_str = day.isoformat()
                        time_practiced = daily_time.get(day_str, 0)
                        
                        with col:
                            if time_practiced > 0:
                                st.success(f"**{day.day}**\n{time_practiced:.0f}m")
                            elif day == today:
                                st.info(f"**{day.day}**\nToday")
                            elif day < today:
                                st.error(f"**{day.day}**\n-")
                            else:
                                st.write(f"**{day.day}**")
            
            st.markdown("---")
            
            st.subheader("This Week")
            
            weekly_stats = get_weekly_stats()
            
            days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for day in days_of_week:
                if day in weekly_stats:
                    sessions = weekly_stats[day]['sessions']
                    time = weekly_stats[day]['time']
                    st.write(f"**{day}**: {sessions} sessions, {time:.1f} minutes")
                else:
                    st.write(f"**{day}**: No practice")
        else:
            st.info("No practice data available")
    
    with tab3:
        st.header("🎯 Goals & Targets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Goal")
            
            target_minutes = st.session_state.get('daily_minutes', 30)
            
            today_str = date.today().isoformat()
            today_practice = sum(
                r.get('duration_min', 0) for r in st.session_state.practice_history
                if r.get('date') == today_str
            )
            
            progress = min(today_practice / target_minutes, 1.0) if target_minutes > 0 else 0
            
            st.metric("Today's Practice", f"{today_practice:.1f} / {target_minutes} min")
            st.progress(progress)
            
            if progress >= 1.0:
                st.success("🎉 Daily goal achieved!")
            elif progress >= 0.5:
                st.info("💪 Halfway there!")
            else:
                st.warning(f"Keep going! {target_minutes - today_practice:.1f} minutes to go.")
        
        with col2:
            st.subheader("BPM Target")
            
            target_bpm = st.session_state.get('target_bpm', 120)
            max_bpm = stats['max_bpm']
            
            progress = min(max_bpm / target_bpm, 1.0) if target_bpm > 0 else 0
            
            st.metric("Max BPM", f"{max_bpm:.0f} / {target_bpm}")
            st.progress(progress)
            
            if progress >= 1.0:
                st.success("🎉 BPM target achieved!")
            elif progress >= 0.8:
                st.info("💪 Almost there!")
            else:
                st.warning(f"{target_bpm - max_bpm:.0f} BPM to go.")
        
        st.markdown("---")
        
        st.subheader("12-Key Mastery")
        
        from utils import get_all_keys
        all_keys = get_all_keys()
        
        keys_practiced_set = set()
        for record in st.session_state.practice_history:
            if record.get('key'):
                keys_practiced_set.add(record['key'])
        
        cols = st.columns(6)
        for i, key in enumerate(all_keys):
            with cols[i % 6]:
                if key in keys_practiced_set:
                    st.success(f"✅ {key}")
                else:
                    st.info(f"⭕ {key}")
        
        progress = len(keys_practiced_set) / len(all_keys)
        st.progress(progress)
        st.caption(f"{len(keys_practiced_set)} / {len(all_keys)} keys practiced")
        
        st.markdown("---")
        
        st.subheader("Weekly Goal")
        
        weekly_target = target_minutes * 7
        
        weekly_stats = get_weekly_stats()
        weekly_total = sum(day['time'] for day in weekly_stats.values())
        
        progress = min(weekly_total / weekly_target, 1.0) if weekly_target > 0 else 0
        
        st.metric("This Week", f"{weekly_total:.1f} / {weekly_target} min")
        st.progress(progress)
        
        if progress >= 1.0:
            st.success("🎉 Weekly goal achieved!")
        elif progress >= 0.7:
            st.info("💪 Great week so far!")
        else:
            st.warning(f"{weekly_target - weekly_total:.1f} minutes to go this week.")
    
    with tab4:
        st.header("📋 Practice History")
        
        st.subheader("Recent Sessions")
        
        records = sorted(
            st.session_state.practice_history,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        limit = st.slider("Show last N sessions", min_value=10, max_value=100, value=20, step=10)
        
        for i, record in enumerate(records[:limit]):
            with st.expander(f"Session {i+1}: {record.get('date')} - {record.get('category', 'unknown').replace('_', ' ').title()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Date:** {record.get('date')}")
                    st.write(f"**Category:** {record.get('category', 'unknown').replace('_', ' ').title()}")
                    st.write(f"**Duration:** {record.get('duration_min', 0):.1f} minutes")
                
                with col2:
                    if record.get('key'):
                        st.write(f"**Key:** {record['key']}")
                    if record.get('bpm'):
                        st.write(f"**BPM:** {record['bpm']}")
                    if record.get('scale_type'):
                        st.write(f"**Scale:** {record['scale_type']}")
                    if record.get('progression'):
                        st.write(f"**Progression:** {record['progression']}")
        
        st.markdown("---")
        
        if st.button("📥 Export Practice Data", use_container_width=True):
            data_file = Path(__file__).parent.parent / 'data' / 'practice_export.json'
            data_file.parent.mkdir(exist_ok=True)
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.practice_history, f, indent=2, ensure_ascii=False)
            
            st.success(f"Data exported to {data_file}")
        
        if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
            if st.button("⚠️ Confirm Clear All Data", use_container_width=True, type="primary"):
                st.session_state.practice_history = []
                st.session_state.total_practice_time = 0
                st.session_state.current_streak = 0
                
                from main import save_user_data
                save_user_data()
                
                st.success("All data cleared!")
                st.rerun()
    
    st.markdown("---")
    
    st.subheader("📚 Statistics Guide")
    
    with st.expander("📊 Understanding Your Stats"):
        st.markdown("""
        **Total Sessions**: Number of practice sessions recorded
        
        **Total Time**: Cumulative practice time across all sessions
        
        **Days Practiced**: Number of unique days with practice
        
        **Max BPM**: Highest tempo achieved in any practice session
        
        **Current Streak**: Consecutive days with practice
        
        **Keys Practiced**: Number of different keys you've practiced in
        
        **Category Breakdown**: Time spent on each type of practice
        """)
    
    with st.expander("🎯 Setting Effective Goals"):
        st.markdown("""
        **SMART Goals:**
        - **Specific**: "Practice major scales in all 12 keys"
        - **Measurable**: "Reach 120 BPM"
        - **Achievable**: Start with realistic targets
        - **Relevant**: Focus on what matters for your development
        - **Time-bound**: "Within 3 months"
        
        **Recommended Goals:**
        - Practice 30 minutes daily (minimum)
        - Master one new scale per week
        - Learn one new standard per month
        - Increase BPM by 5 each week
        - Practice in all 12 keys monthly
        
        **Tracking Progress:**
        - Record every practice session
        - Review stats weekly
        - Adjust goals as needed
        - Celebrate milestones
        - Stay consistent
        """)
    
    with st.expander("💡 Practice Tips"):
        st.markdown("""
        **Quality over Quantity:**
        - Focused 30 minutes > Unfocused 2 hours
        - Practice with intention
        - Use metronome
        - Record yourself
        
        **Consistency is Key:**
        - Daily practice is better than long weekend sessions
        - Build a routine
        - Same time each day helps
        - Even 15 minutes counts
        
        **Balanced Practice:**
        - Don't neglect any area
        - Rotate through different exercises
        - Warm up properly
        - Cool down and reflect
        
        **Track Everything:**
        - Record all practice sessions
        - Note what works
        - Identify weak areas
        - Celebrate progress
        """)


if __name__ == "__main__":
    main()
