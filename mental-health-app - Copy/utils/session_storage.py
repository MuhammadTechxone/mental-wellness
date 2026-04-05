"""
MindMirror AI - Session-Only Storage
No files created - data lives only in browser session
"""

import streamlit as st
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import hashlib


class SessionTracker:
    """
    Session-only storage - Data cleared when browser tab closes
    Perfect for privacy and mobile users
    """
    
    def __init__(self, user_id: Optional[str] = None):
        """Initialize session storage - no files created!"""
        
        # Initialize session state if not exists
        if 'session_assessments' not in st.session_state:
            st.session_state.session_assessments = []
        
        if 'session_user_id' not in st.session_state:
            # Create temporary session ID (only for this browsing session)
            st.session_state.session_user_id = hashlib.md5(
                str(datetime.now().timestamp()).encode()
            ).hexdigest()[:8]
        
        # Store user_id if provided (for returning users - but in session-only,
        # returning users will get a new ID anyway)
        if user_id and user_id.strip():
            st.session_state.session_user_id = user_id.strip()
        
        self.user_id = st.session_state.session_user_id
        self.assessments = st.session_state.session_assessments
    
    def add_assessment(self, results: Dict, responses: Dict) -> None:
        """
        Add assessment to current session
        NO files created - stored in browser memory only
        """
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%b %d, %Y'),
            'time': datetime.now().strftime('%I:%M %p'),
            'primary': results['primary_indicator'],
            'primary_pct': round(results['primary_percentage'], 1),
            'secondary': results.get('secondary_indicator'),
            'secondary_pct': round(results.get('secondary_percentage', 0), 1),
            'anxiety': round(results['distribution'].get('anxiety', 0), 1),
            'depression': round(results['distribution'].get('depression', 0), 1),
            'suicidal': round(results['distribution'].get('suicidal', 0), 1),
            'normal': round(results['distribution'].get('normal', 0), 1)
        }
        
        # Store in session state (browser memory)
        st.session_state.session_assessments.append(assessment)
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Return assessment history as DataFrame
        Data comes from browser memory, NOT from files
        """
        if not st.session_state.session_assessments:
            return pd.DataFrame()
        
        df = pd.DataFrame(st.session_state.session_assessments)
        df['assessment'] = range(1, len(df) + 1)
        
        # Select and rename columns for display
        display_df = df[[
            'assessment', 'date', 'primary', 'primary_pct',
            'anxiety', 'depression', 'suicidal', 'normal'
        ]]
        
        return display_df
    
    def get_progress_summary(self) -> Dict:
        """
        Generate progress summary from session data
        """
        df = self.get_history_dataframe()
        if df.empty:
            return {}
        
        first = df.iloc[0]
        last = df.iloc[-1]
        
        # Calculate trends
        trend = {
            'normal': round(last['normal'] - first['normal'], 1),
            'anxiety': round(last['anxiety'] - first['anxiety'], 1),
            'depression': round(last['depression'] - first['depression'], 1),
            'suicidal': round(last['suicidal'] - first['suicidal'], 1)
        }
        
        # Find most common primary indicator
        most_common = df['primary'].mode()
        most_common_primary = most_common[0] if not most_common.empty else None
        
        return {
            'total_assessments': len(df),
            'first_date': first['date'],
            'last_date': last['date'],
            'trend': trend,
            'most_common_primary': most_common_primary
        }
    
    def has_history(self) -> bool:
        """Check if user has any assessments in this session"""
        return len(st.session_state.session_assessments) > 0
    
    def clear_history(self):
        """Clear all session data"""
        st.session_state.session_assessments = []
        st.session_state.session_user_id = hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:8]
        self.user_id = st.session_state.session_user_id