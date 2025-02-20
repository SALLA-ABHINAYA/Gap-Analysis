# pages/6_Gap_Analysis.py
import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from typing import Dict, List, Any
import logging
from ocpm_gap_analysis import OCPMAnalyzer

logger = logging.getLogger(__name__)


class GapAnalysisUI:
    """UI Component for OCPM Gap Analysis"""

    def __init__(self):
        self.gap_file = "ocpm_output/gap_analysis.csv"
        self.guidelines_file = "ocpm_data/expected_guidelines.txt"

        # Ensure output directory exists
        os.makedirs("ocpm_output", exist_ok=True)

    def _load_gap_data(self) -> pd.DataFrame:
        """Load gap analysis data"""
        try:
            return pd.read_csv(self.gap_file)
        except Exception as e:
            logger.error(f"Error loading gap data: {str(e)}")
            st.error("Please run gap analysis first")
            return pd.DataFrame()

    def _create_severity_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create severity distribution chart"""
        fig = go.Figure()

        # Add severity score distribution
        fig.add_trace(go.Histogram(
            x=df['severity_score'],
            nbinsx=20,
            name='Severity Distribution'
        ))

        # Add thresholds
        fig.add_vline(x=80, line_dash="dash", line_color="red",
                      annotation_text="High Severity")
        fig.add_vline(x=50, line_dash="dash", line_color="orange",
                      annotation_text="Medium Severity")

        fig.update_layout(
            title="Gap Severity Distribution",
            xaxis_title="Severity Score",
            yaxis_title="Number of Cases"
        )

        return fig

    def _create_gap_type_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create gap type distribution chart"""
        gap_types = {
            'Missing Steps': df['missing_steps'].str.len().sum(),
            'Extra Steps': df['extra_steps'].str.len().sum(),
            'Sequence Issues': df['sequence_issues'].str.len().sum(),
            'Timing Violations': df['timing_violations'].str.len().sum(),
            'Object Violations': df['object_violations'].str.len().sum(),
            'Compliance Gaps': df['compliance_gaps'].str.len().sum()
        }

        fig = go.Figure(data=[
            go.Bar(
                x=list(gap_types.keys()),
                y=list(gap_types.values()),
                marker_color=['red', 'orange', 'yellow', 'blue', 'green', 'purple']
            )
        ])

        fig.update_layout(
            title="Gap Type Distribution",
            xaxis_title="Gap Type",
            yaxis_title="Number of Occurrences"
        )

        return fig

    def display_case_details(self, case_data: pd.Series):
        """Display detailed analysis for a case"""
        st.subheader(f"Case Analysis: {case_data['case_id']}")

        # Metrics row
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Severity Score", f"{case_data['severity_score']}/100")

        # Tabs for different types of gaps
        gap_tabs = st.tabs([
            "Missing & Extra Steps",
            "Sequence Issues",
            "Timing Violations",
            "Recommendations"
        ])

        # Missing and Extra Steps Tab
        with gap_tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Missing Steps")
                for step in eval(case_data['missing_steps']):
                    st.error(f"• {step}")
            with col2:
                st.markdown("##### Extra Steps")
                for step in eval(case_data['extra_steps']):
                    st.warning(f"• {step}")

        # Sequence Issues Tab
        with gap_tabs[1]:
            st.markdown("##### Sequence Issues")
            for issue in eval(case_data['sequence_issues']):
                st.markdown(f"**Issue**: {issue.get('issue', '')}")
                cols = st.columns(2)
                with cols[0]:
                    st.info("Expected Sequence:")
                    st.write(' → '.join(issue.get('expected_sequence', [])))
                with cols[1]:
                    st.warning("Actual Sequence:")
                    st.write(' → '.join(issue.get('actual_sequence', [])))
                st.divider()

        # Timing Violations Tab
        with gap_tabs[2]:
            st.markdown("##### Timing Violations")
            for violation in eval(case_data['timing_violations']):
                st.markdown(f"**Phase**: {violation.get('phase', '')}")
                cols = st.columns(3)
                cols[0].metric("Expected", violation.get('expected_time', ''))
                cols[1].metric("Actual", violation.get('actual_time', ''))
                cols[2].metric("Status", "Delayed" if violation.get('actual_time', '') > violation.get('expected_time',
                                                                                                       '') else "On Time",
                               delta=violation.get('difference', ''))
                st.divider()

        # Recommendations Tab
        with gap_tabs[3]:
            st.markdown("##### Recommendations")
            for idx, rec in enumerate(eval(case_data['recommendations']), 1):
                st.info(f"{idx}. {rec}")

    def render(self):
        """Render the gap analysis UI"""
        st.title("OCPM Gap Analysis")

        # Add run analysis button
        if st.button("Run Gap Analysis"):
            with st.spinner("Running gap analysis..."):
                try:
                    analyzer = OCPMAnalyzer(self.guidelines_file)
                    analyzer.generate_report()
                    st.success("Gap analysis completed!")
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
                    return

        # Load and display data
        df = self._load_gap_data()
        if df.empty:
            return

        # Overview section
        st.header("Gap Analysis Overview")

        # Display charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(self._create_severity_chart(df), use_container_width=True)
        with col2:
            st.plotly_chart(self._create_gap_type_chart(df), use_container_width=True)

        # Case details section
        st.header("Detailed Case Analysis")
        selected_case = st.selectbox(
            "Select Case for Detailed Analysis",
            df['case_id'].unique()
        )

        if selected_case:
            self.display_case_details(df[df['case_id'] == selected_case].iloc[0])


def main():
    st.set_page_config(page_title="Gap Analysis", layout="wide")
    gap_ui = GapAnalysisUI()
    gap_ui.render()


if __name__ == "__main__":
    main()