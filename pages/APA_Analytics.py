# APA_Analytics.py
import streamlit as st
from ocpm_analysis import create_ocpm_ui
import os
from Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer
from pathlib import Path
from openai import OpenAI
import pandas as pd
import plotly.express as px
import json
from typing import Dict, List
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class IntegratedAPAAnalyzer:
    """Integrated APA Analytics with AI Analysis capabilities"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")

        self.client = OpenAI(api_key=self.api_key)
        self.ocel_data = None
        self.events_df = None
        self.stats = {}

    def load_ocel(self, file_path: str) -> None:
        """Load and process OCEL data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.ocel_data = json.load(f)

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
        """Calculate OCEL statistics"""
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
                'distribution': self.events_df['resource'].value_counts().to_dict()
            },
            'object_types': {
                'count': len(self.ocel_data.get('ocel:object-types', [])),
                'types': self.ocel_data.get('ocel:object-types', [])
            }
        }

    def analyze_with_ai(self, question: str) -> str:
        """AI-powered OCEL analysis"""
        try:
            case_events = self.events_df.sort_values('timestamp')
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

            context = f"""
            OCEL Log Analysis Context:
            General Statistics:
            - Total Events: {self.stats['general']['total_events']}
            - Total Cases: {self.stats['general']['total_cases']}
            - Date Range: {self.stats['general']['date_range']['start']} to {self.stats['general']['date_range']['end']}
            - Object Types: {', '.join(self.stats['object_types']['types'])}

            Detailed Event Log:
            {''.join(events_context)}

            Based on this process event log data, please answer:
            {question}
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
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
        """Create interactive visualizations"""
        figures = {}

        # Activity Distribution
        fig_activities = px.bar(
            x=list(self.stats['activities']['distribution'].keys()),
            y=list(self.stats['activities']['distribution'].values()),
            title="Activity Distribution",
            labels={'x': 'Activity', 'y': 'Count'}
        )
        figures['activity_distribution'] = fig_activities

        # Resource Distribution
        fig_resources = px.bar(
            x=list(self.stats['resources']['distribution'].keys()),
            y=list(self.stats['resources']['distribution'].values()),
            title="Resource Distribution",
            labels={'x': 'Resource', 'y': 'Count'}
        )
        figures['resource_distribution'] = fig_resources

        return figures


def run_unfairness_analysis():
    """Run unfairness analysis using UnfairOCELAnalyzer"""
    st.subheader("Unfairness Analysis")

    ocel_path = st.session_state.get('ocel_path')
    if not ocel_path:
        ocel_path = find_ocel_file()

    if not ocel_path or not os.path.exists(ocel_path):
        st.warning("⚠️ Please process data in the Process Analysis tab first.")
        return

    try:
        analyzer = UnfairOCELAnalyzer(ocel_path)
        plots, metrics = analyzer.get_analysis_plots()

        # Show analysis results
        tabs = st.tabs([
            "Resource Discrimination",
            "Time Bias",
            "Case Priority",
            "Handover Patterns"
        ])

        with tabs[0]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['resource_discrimination'])
            with col2:
                st.markdown("### Resource Bias Findings")
                for resource, data in metrics['resource'].items():
                    if isinstance(data, dict) and data.get('bias_score', 0) > 0.2:
                        st.warning(f"⚠️ {resource}: {data['bias_score']:.2f} bias score")
                        if st.button(f"Show traces for {resource}"):
                            trace_report = analyzer.generate_trace_report('resource', resource)
                            st.text(trace_report)

        with tabs[1]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['time_bias'])
            with col2:
                st.markdown("### Processing Time Statistics")
                for resource, data in metrics['time'].items():
                    if isinstance(data, dict):
                        with st.expander(f"{resource} Details"):
                            st.write(f"Mean time: {data.get('mean_time', 0):.2f} hours")
                            st.write(f"Std Dev: {data.get('std_dev', 0):.2f} hours")
                            if st.button(f"Show traces for {resource} times"):
                                trace_report = analyzer.generate_trace_report('resource', resource)
                                st.text(trace_report)

        with tabs[2]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['case_priority'])
            with col2:
                st.markdown("### Case Priority Analysis")
                for case_type, data in metrics['case'].items():
                    if isinstance(data, dict) and abs(data.get('bias_score', 0)) > 0.2:
                        status = "👎" if data.get('bias_score', 0) > 0 else "👍"
                        st.info(f"{case_type}: {status}")
                        if st.button(f"Show traces for {case_type}"):
                            trace_report = analyzer.generate_trace_report('case', case_type)
                            st.text(trace_report)

        with tabs[3]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['handover'])
            with col2:
                st.markdown("### Significant Handovers")
                for pattern, data in metrics['handover'].items():
                    if isinstance(data, dict) and data.get('percentage', 0) > 10:
                        st.write(f"🔄 {pattern}: {data['percentage']:.1f}%")
                        if st.button(f"Show traces for {pattern}"):
                            trace_report = analyzer.generate_trace_report('handover', pattern)
                            st.text(trace_report)

    except Exception as e:
        st.error(f"Error in unfairness analysis: {str(e)}")
        import traceback
        st.error(f"Detailed error:\n{traceback.format_exc()}")


def find_ocel_file():
    """Find the OCEL file in expected locations"""
    possible_paths = [
        "ocpm_output/process_data.json",
        "ocpm_data/process_data.json"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def run_ai_analysis():
    """Run AI-powered analysis"""
    st.subheader("🤖 AI-Powered Process Analysis")

    ocel_path = st.session_state.get('ocel_path')
    if not ocel_path or not os.path.exists(ocel_path):
        st.warning("⚠️ Please process data in the Process Analysis tab first.")
        return

    try:
        analyzer = IntegratedAPAAnalyzer()
        analyzer.load_ocel(ocel_path)

        # Display Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", analyzer.stats['general']['total_events'])
        with col2:
            st.metric("Total Cases", analyzer.stats['general']['total_cases'])
        with col3:
            st.metric("Total Resources", analyzer.stats['general']['total_resources'])

        # AI Analysis Section
        st.subheader("Ask Questions About Your Process")

        st.write("**Example Questions:**")
        st.write("- What are the most common activity sequences?")
        st.write("- How are different resources utilized across cases?")
        st.write("- What is the typical process flow for trades?")

        question = st.text_input(
            "Ask a question about the process:",
            placeholder="e.g., What are the main process patterns?"
        )

        if question:
            with st.spinner("Analyzing..."):
                analysis = analyzer.analyze_with_ai(question)
                st.write(analysis)

        # Show visualizations
        st.subheader("Process Visualizations")
        figures = analyzer.create_visualizations()

        viz_tabs = st.tabs(["Activities", "Resources"])
        with viz_tabs[0]:
            st.plotly_chart(figures['activity_distribution'], use_container_width=True)
        with viz_tabs[1]:
            st.plotly_chart(figures['resource_distribution'], use_container_width=True)

    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")


def setup_apa_page():
    """Set up the integrated APA Analytics page"""
    st.title("📊 IRMAI APA Analytics")

    st.info("APA provides comprehensive process analysis including AI-powered insights.")

    # Create directories
    os.makedirs("ocpm_data", exist_ok=True)
    os.makedirs("ocpm_output", exist_ok=True)

    # Create tabs for different analyses
    main_tabs = st.tabs(["Process Analysis", "Unfairness Analysis", "AI Insights"])

    with main_tabs[0]:
        create_ocpm_ui()

        if 'ocpm_df' in st.session_state:
            ocel_path = Path("ocpm_output/process_data.json")
            st.session_state['ocel_path'] = str(ocel_path)

    with main_tabs[1]:
        run_unfairness_analysis()

    with main_tabs[2]:
        run_ai_analysis()


if __name__ == "__main__":
    st.set_page_config(
        page_title="IRMAI APA Analytics",
        page_icon="📊",
        layout="wide"
    )
    setup_apa_page()