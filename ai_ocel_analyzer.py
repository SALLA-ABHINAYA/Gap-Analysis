import os
import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from openai import OpenAI  # Changed to standard OpenAI
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict


class AIOCELAnalyzer:
    """AI-powered OCEL log analyzer using OpenAI"""

    def __init__(self, api_key: str = None):
        """Initialize the analyzer with OpenAI credentials"""
        # Use parameter or get from secrets
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("Missing required OpenAI API key")

        self.client = OpenAI(
            api_key=self.api_key
        )
        self.ocel_data = None
        self.events_df = None
        self.stats = {}

    def load_ocel(self, file_path: str) -> None:
        """Load and process OCEL JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.ocel_data = json.load(f)

            # Convert events to DataFrame for easier analysis
            events = []
            for event in self.ocel_data['ocel:events']:
                event_data = {
                    'event_id': event['ocel:id'],
                    'timestamp': pd.to_datetime(event['ocel:timestamp']),
                    'activity': event['ocel:activity'],
                    'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                    'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                    'object_type': event.get('ocel:attributes', {}).get('object_type', 'Unknown'),
                    'objects': [obj['id'] for obj in event.get('ocel:objects', [])]
                }
                events.append(event_data)

            self.events_df = pd.DataFrame(events)
            self._calculate_statistics()

        except Exception as e:
            raise Exception(f"Error loading OCEL file: {str(e)}")

    def _calculate_statistics(self) -> None:
        """Calculate comprehensive OCEL statistics"""
        if self.events_df is None:
            return

        self.stats = {
            'general': {
                'total_events': len(self.events_df),
                'total_cases': self.events_df['case_id'].nunique(),
                'date_range': {
                    'start': self.events_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': self.events_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'total_resources': self.events_df['resource'].nunique()
            },
            'activities': {
                'unique_count': self.events_df['activity'].nunique(),
                'distribution': self.events_df['activity'].value_counts().to_dict()
            },
            'resources': {
                'distribution': self.events_df['resource'].value_counts().to_dict(),
                'activity_counts': self.events_df.groupby('resource')['activity'].nunique().to_dict()
            },
            'object_types': {
                'count': len(self.ocel_data.get('ocel:object-types', [])),
                'types': self.ocel_data.get('ocel:object-types', [])
            }
        }

    def analyze_with_ai(self, question: str) -> str:
        """Analyze OCEL data using OpenAI"""
        try:
            # Get detailed case information
            case_events = self.events_df.sort_values('timestamp')

            # Create detailed events context
            events_context = []
            for _, event in case_events.iterrows():
                event_str = f"""
                Event: {event['event_id']}
                - Activity: {event['activity']}
                - Timestamp: {event['timestamp']}
                - Resource: {event['resource']}
                - Case ID: {event['case_id']}
                - Object Type: {event['object_type']}
                - Related Objects: {', '.join(event['objects'])}
                """
                events_context.append(event_str)

            # Create a context string with statistics and events
            context = f"""
            OCEL Log Analysis Context:

            General Statistics:
            - Total Events: {self.stats['general']['total_events']}
            - Total Cases: {self.stats['general']['total_cases']}
            - Date Range: {self.stats['general']['date_range']['start']} to {self.stats['general']['date_range']['end']}
            - Activities: {', '.join(list(self.stats['activities']['distribution'].keys()))}
            - Resources: {', '.join(list(self.stats['resources']['distribution'].keys()))}
            - Object Types: {', '.join(self.stats['object_types']['types'])}

            Detailed Event Log:
            {''.join(events_context)}

            Based on the above process event log data, please answer this question:
            {question}

            Provide a detailed analysis focusing on the specific events, their sequence, timing, and relationships between different objects and activities.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4 model
                messages=[
                    {"role": "system",
                     "content": "You are an expert process mining analyst. Analyze the OCEL log and provide insights."},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error in AI analysis: {str(e)}"

    def create_visualizations(self):
        """Create interactive visualizations of OCEL data"""
        figures = {}

        # Activity Distribution
        fig_activities = px.bar(
            x=list(self.stats['activities']['distribution'].keys()),
            y=list(self.stats['activities']['distribution'].values()),
            title="Activity Distribution",
            labels={'x': 'Activity', 'y': 'Count'}
        )
        figures['activity_distribution'] = fig_activities

        # Resource Workload
        fig_resources = px.bar(
            x=list(self.stats['resources']['distribution'].keys()),
            y=list(self.stats['resources']['distribution'].values()),
            title="Resource Workload",
            labels={'x': 'Resource', 'y': 'Number of Events'}
        )
        figures['resource_workload'] = fig_resources

        # Timeline
        timeline_data = self.events_df.groupby([pd.Grouper(key='timestamp', freq='D')])[
            'event_id'].count().reset_index()
        fig_timeline = px.line(
            timeline_data,
            x='timestamp',
            y='event_id',
            title="Event Timeline",
            labels={'timestamp': 'Date', 'event_id': 'Number of Events'}
        )
        figures['event_timeline'] = fig_timeline

        return figures


def create_ai_ocel_ui():
    """Create Streamlit UI for AI-powered OCEL analysis"""
    st.subheader("🤖 AI-Powered OCEL Analysis")

    # Get OpenAI API key from secrets
    api_key = st.secrets.get("OPENAI_API_KEY", "")

    if not api_key:
        st.error("OpenAI API key not configured. Please check your secrets.")
        return

    try:
        analyzer = AIOCELAnalyzer(api_key)

        # Load OCEL file
        ocel_path = "ocpm_output/process_data.json"
        if not os.path.exists(ocel_path):
            st.warning("Please process data in the Process Analysis tab first.")
            return

        analyzer.load_ocel(ocel_path)

        # Display Statistics
        st.subheader("📊 OCEL Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", analyzer.stats['general']['total_events'])
        with col2:
            st.metric("Total Cases", analyzer.stats['general']['total_cases'])
        with col3:
            st.metric("Total Resources", analyzer.stats['general']['total_resources'])

        # Display visualizations
        st.subheader("📈 Visualizations")
        figures = analyzer.create_visualizations()

        viz_tabs = st.tabs(["Activities", "Resources", "Timeline"])
        with viz_tabs[0]:
            st.plotly_chart(figures['activity_distribution'], use_container_width=True)
        with viz_tabs[1]:
            st.plotly_chart(figures['resource_workload'], use_container_width=True)
        with viz_tabs[2]:
            st.plotly_chart(figures['event_timeline'], use_container_width=True)

        # AI Analysis
        st.subheader("🔍 AI Analysis")

        # Show available cases and activities
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Available Cases:**")
            cases = sorted(analyzer.events_df['case_id'].unique())
            for case in cases:
                st.write(f"- {case}")

        with col2:
            st.write("**Activities in Process:**")
            activities = sorted(analyzer.events_df['activity'].unique())
            for activity in activities:
                st.write(f"- {activity}")

        # Example questions and input
        st.write("**Example Questions:**")
        st.write("- Describe Case 1 activities and their sequence")
        st.write("- What is the typical duration of activities in Case 1?")
        st.write("- How many different object types are involved in Case 1?")

        question = st.text_input(
            "Ask a question about the process:",
            placeholder="e.g., Describe the complete flow of Case 1"
        )

        if question:
            with st.spinner("Analyzing..."):
                analysis = analyzer.analyze_with_ai(question)
                st.write(analysis)

        # Show detailed statistics
        with st.expander("View Detailed Statistics"):
            st.json(analyzer.stats)

    except Exception as e:
        st.error(f"Error in AI OCEL analysis: {str(e)}")