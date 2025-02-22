# pages/6_Gap_Analysis.py
import os
from collections import defaultdict
from typing import List, Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

    def _get_gap_type_description(self, gap_type: str) -> str:
        """Get description for each gap type"""
        descriptions = {
            'Control Gaps': 'Missing mandatory controls and compliance requirements in the process',
            'Unsupported Control Gaps': 'Additional unauthorized or non-compliant control steps',
            'Control Flow Gaps': 'Control and compliance steps executed out of required sequence',
            'SLA breach': 'Regulatory and compliance SLA violations',
            'Object Violations': 'Control violations in object interactions and relationships',
            'Compliance Gaps': 'Direct violations of compliance rules and regulatory requirements'
        }
        return descriptions.get(gap_type, '')

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
        # Map internal names to display names
        display_names = {
            'missing_steps': 'Control Gaps',
            'extra_steps': 'Unsupported Control Gaps',
            'sequence_issues': 'Control Flow Gaps',
            'timing_violations': 'SLA breach',
            'object_violations': 'Object Violations',
            'compliance_gaps': 'Compliance Gaps'
        }

        # Calculate values using internal names
        gap_types = {
            display_names[k]: df[k].str.len().sum() for k in [
                'missing_steps', 'extra_steps', 'sequence_issues',
                'timing_violations', 'object_violations', 'compliance_gaps'
            ]
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
            "Control Gaps",
            "Control Flow Gaps",
            "Timing Violations",
            "Object Violations",
            "Compliance Gaps",
            "Recommendations"
        ])

        # Missing and Extra Steps Tab
        with gap_tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Control Gaps")
                for step in eval(case_data['missing_steps']):
                    st.error(f"• {step}")
            with col2:
                st.markdown("##### Unsupported Control Gaps")
                for step in eval(case_data['extra_steps']):
                    st.warning(f"• {step}")

        # Sequence Issues Tab
        with gap_tabs[1]:
            st.markdown("##### Control Flow Gaps")
            for issue in eval(case_data['sequence_issues']):
                st.markdown(f"**Issue**: {issue.get('issue', '')}")
                cols = st.columns(2)
                with cols[0]:
                    st.info("Expected Control Flow:")
                    st.write(' → '.join(issue.get('expected_sequence', [])))
                with cols[1]:
                    st.warning("Actual Control Flow:")
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

        # Object Violations Tab
        with gap_tabs[3]:
            st.markdown("##### Object Violations")
            for violation in eval(case_data['object_violations']):
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Step**: {violation.get('step', '')}")
                with cols[1]:
                    st.error(f"**Issue**: {violation.get('issue', '')}")
                if violation.get('affected_objects'):
                    st.write("**Affected Objects:**")
                    for obj in violation.get('affected_objects', []):
                        st.warning(f"• {obj}")
                st.divider()

        # Compliance Gaps Tab
        with gap_tabs[4]:
            st.markdown("##### Compliance Gaps")
            for gap in eval(case_data['compliance_gaps']):
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Issue**: {gap.get('issue', '')}")
                with cols[1]:
                    st.error(f"**Impact**: {gap.get('impact', '')}")
                st.divider()

        # Recommendations Tab
        with gap_tabs[5]:
            st.markdown("##### Recommendations")
            for idx, rec in enumerate(eval(case_data['recommendations']), 1):
                st.info(f"{idx}. {rec}")

    def _get_enhanced_gap_description(self, gap_type: str) -> str:
        """Get enhanced description with OCPM context for each gap type"""
        descriptions = {
            'Missing Steps': """
                - **OCPM Context**: Indicates missing mandatory object lifecycle steps
                - **Object Impact**: Incomplete object state transitions
                - **Business Impact**: 
                    - Incomplete object processing
                    - Regulatory compliance risks
                    - Process integrity issues
                - **Common Causes**:
                    - Skipped process steps
                    - System integration issues
                    - Resource availability problems
            """,
            'Extra Steps': """
                - **OCPM Context**: Unauthorized or redundant object interactions
                - **Object Impact**: Unnecessary object state changes
                - **Business Impact**:
                    - Process inefficiency
                    - Resource waste
                    - Increased complexity
                - **Common Causes**:
                    - Process confusion
                    - Lack of standardization
                    - System automation issues
            """,
            'Sequence Issues': """
                - **OCPM Context**: Incorrect order of object lifecycle events
                - **Object Impact**: Invalid object state transitions
                - **Business Impact**:
                    - Process inconsistency
                    - Compliance violations
                    - Data integrity issues
                - **Common Causes**:
                    - Process knowledge gaps
                    - System control issues
                    - Resource coordination problems
            """,
            'Timing Violations': """
                - **OCPM Context**: Delayed object lifecycle transitions
                - **Object Impact**: Object state transition delays
                - **Business Impact**:
                    - SLA violations
                    - Customer satisfaction impact
                    - Resource bottlenecks
                - **Common Causes**:
                    - Resource constraints
                    - System performance issues
                    - Process bottlenecks
            """,
            'Object Violations': """
                - **OCPM Context**: Invalid object relationships or states
                - **Object Impact**: Incorrect object interactions
                - **Business Impact**:
                    - Data consistency issues
                    - Process integrity problems
                    - Compliance risks
                - **Common Causes**:
                    - Data quality issues
                    - System integration problems
                    - Process design flaws
            """,
            'Compliance Gaps': """
                - **OCPM Context**: Regulatory rule violations in object handling
                - **Object Impact**: Non-compliant object states
                - **Business Impact**:
                    - Regulatory exposure
                    - Audit findings
                    - Legal risks
                - **Common Causes**:
                    - Control failures
                    - Process design issues
                    - Training gaps
            """
        }
        return descriptions.get(gap_type, '')

    def _get_top_gaps_for_object(self, gaps: Dict, gap_types: List[str]) -> str:
        """Get formatted string of top gap types for an object"""
        gap_counts = [(gap_type, gaps[gap_type]) for gap_type in gap_types if gap_type != 'total']
        gap_counts.sort(key=lambda x: x[1], reverse=True)
        top_gaps = gap_counts[:3]  # Get top 3 gaps

        result = ""
        for gap_type, count in top_gaps:
            if count > 0:
                percentage = (count / gaps['total'] * 100)
                result += f"\n                - {gap_type}: {count} ({percentage:.1f}%)"

        return result or "\n                - No significant gaps"

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

        # Display charts with explanations
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(self._create_severity_chart(df), use_container_width=True)
            with st.expander("📊 Understanding Severity Distribution in OCPM Context"):
                # Calculate severity stats
                high_severity = len(df[df['severity_score'] > 80])
                medium_severity = len(df[df['severity_score'].between(50, 80)])
                low_severity = len(df[df['severity_score'] < 50])
                avg_severity = df['severity_score'].mean()

                # Calculate object type impact
                object_impacts = {}
                for _, row in df.iterrows():
                    try:
                        violations = eval(row['object_violations'])
                        for violation in violations:
                            # Extract object type from violation data
                            affected_objects = violation.get('affected_objects', [])
                            if not affected_objects:
                                # Check for object type in event details
                                obj_type = violation.get('step', '').split()[0]  # Try to get object type from step name
                                if not obj_type or obj_type not in ['Trade', 'Order', 'Position', 'Market', 'Client']:
                                    obj_type = 'Unclassified'
                            else:
                                obj_type = affected_objects[0]
                                if not obj_type or obj_type == 'Unknown':
                                    # Attempt to derive from violation step or context
                                    obj_type = violation.get('step', '').split()[0]
                                    if not obj_type or obj_type not in ['Trade', 'Order', 'Position', 'Market',
                                                                        'Client']:
                                        obj_type = 'Unclassified'

                            if obj_type not in object_impacts:
                                object_impacts[obj_type] = {
                                    'count': 0,
                                    'total_severity': 0,
                                    'violation_types': set(),
                                    'affected_steps': set()
                                }

                            object_impacts[obj_type]['count'] += 1
                            object_impacts[obj_type]['total_severity'] += row['severity_score']
                            object_impacts[obj_type]['violation_types'].add(violation.get('issue', 'Unknown Issue'))
                            object_impacts[obj_type]['affected_steps'].add(violation.get('step', 'Unknown Step'))

                    except Exception as e:
                        logger.error(f"Error processing violation for row {row['case_id']}: {str(e)}")
                        continue

                # Calculate average severity by object type
                for obj_type in object_impacts:
                    object_impacts[obj_type]['avg_severity'] = (
                            object_impacts[obj_type]['total_severity'] / object_impacts[obj_type]['count']
                    )

                st.markdown("""
                ### Severity Distribution Analysis in Object-Centric Context

                The severity distribution represents the criticality of process deviations in an object-centric context. 

                #### Severity Score Calculation
                The severity score (0-100) is calculated based on:
                - **Object Interaction Impact**: Weight of affected object relationships
                - **Process Flow Deviation**: Degree of deviation from expected object lifecycles
                - **Business Rule Violations**: Severity of business/regulatory rule breaches
                - **Cross-Object Dependencies**: Impact on dependent object types

                #### Severity Thresholds
                - **High Severity (>80)**: Critical impact on multiple object lifecycles or core business rules
                - **Medium Severity (50-80)**: Significant impact on single object lifecycle or process flow
                - **Low Severity (<50)**: Minor deviations with limited object interaction impact
                """)

                st.markdown(f"""
                #### Current Analysis Statistics
                - **High Severity Cases**: {high_severity} ({(high_severity / len(df) * 100):.1f}%)
                - **Medium Severity Cases**: {medium_severity} ({(medium_severity / len(df) * 100):.1f}%)
                - **Low Severity Cases**: {low_severity} ({(low_severity / len(df) * 100):.1f}%)
                - **Average Severity Score**: {avg_severity:.1f}

                #### Object Type Impact Analysis
                Shows how different process objects are affected by violations:
                """)

                for obj_type, impact in object_impacts.items():
                    if obj_type == 'Unclassified':
                        st.markdown(f"""
                        **Unclassified Objects** (Activities without clear object classification):
                        - Violations: {impact['count']}
                        - Average Severity: {impact['avg_severity']:.1f}
                        - Common Issues:
                            {', '.join(list(impact['violation_types'])[:3])}
                        - Affected Steps:
                            {', '.join(list(impact['affected_steps'])[:3])}
                        - **Note**: These violations need object type classification review
                        """)
                    else:
                        st.markdown(f"""
                        **{obj_type} Objects**:
                        - Violations: {impact['count']}
                        - Average Severity: {impact['avg_severity']:.1f}
                        - Violation Types:
                            {', '.join(list(impact['violation_types'])[:3])}
                        - Affected Process Steps:
                            {', '.join(list(impact['affected_steps'])[:3])}
                        """)

                if 'Unclassified' in object_impacts:
                    st.warning("""
                    ⚠️ **About Unclassified Objects**:
                    - These represent process steps where object type couldn't be clearly determined
                    - Common causes:
                        1. Missing object type metadata in event log
                        2. System integration gaps
                        3. Process mapping inconsistencies
                    - Recommendation: Review these cases for proper object classification
                    """)

                st.markdown("""
                #### Business Impact Significance
                - **Operational Impact**: Higher severity indicates increased risk of process breakdown
                - **Compliance Risk**: Severity correlates with regulatory compliance exposure
                - **Resource Utilization**: High severity cases often indicate resource allocation issues
                - **Process Performance**: Direct correlation with process efficiency and SLA adherence
                """)

        with col2:
            st.plotly_chart(self._create_gap_type_chart(df), use_container_width=True)
            with st.expander("📈 Understanding Gap Types in OCPM Perspective"):
                # Calculate gap type stats with OCPM context
                gap_types = {
                    'Missing Steps': df['missing_steps'].apply(lambda x: len(eval(x))).sum(),
                    'Extra Steps': df['extra_steps'].apply(lambda x: len(eval(x))).sum(),
                    'Sequence Issues': df['sequence_issues'].apply(lambda x: len(eval(x))).sum(),
                    'Timing Violations': df['timing_violations'].apply(lambda x: len(eval(x))).sum(),
                    'Object Violations': df['object_violations'].apply(lambda x: len(eval(x))).sum(),
                    'Compliance Gaps': df['compliance_gaps'].apply(lambda x: len(eval(x))).sum()
                }
                total_gaps = sum(gap_types.values())

                # Calculate object-centric impact metrics
                object_type_gaps = defaultdict(lambda: defaultdict(int))
                for _, row in df.iterrows():
                    for violation in eval(row['object_violations']):
                        obj_type = violation.get('affected_objects', ['Unknown'])[0]
                        object_type_gaps[obj_type]['total'] += 1
                        for gap_type in gap_types.keys():
                            if len(eval(row[gap_type.lower().replace(' ', '_')])) > 0:
                                object_type_gaps[obj_type][gap_type] += 1

                st.markdown("""
                ### Gap Type Analysis in Object-Centric Process Mining

                Gap types represent different categories of process deviations in OCPM context:

                #### Gap Classification in OCPM
                Each gap type indicates specific object-centric process issues:
                """)

                st.markdown("""
                #### Gap Type Definitions and Impact
                """)

                for gap_type, count in gap_types.items():
                    percentage = (count / total_gaps * 100) if total_gaps > 0 else 0
                    st.markdown(f"""
                    ##### {gap_type}: {count} occurrences ({percentage:.1f}%)
                    {self._get_enhanced_gap_description(gap_type)}
                    """)

                st.markdown("""
                #### Object-Centric Impact Analysis
                Shows how different object types are affected by gaps:
                """)

                for obj_type, gaps in object_type_gaps.items():
                    st.markdown(f"""
                    ##### {obj_type} Object Impact:
                    - Total Violations: {gaps['total']}
                    - Most Common Gap Types:
                        {self._get_top_gaps_for_object(gaps, gap_types.keys())}
                    """)

                st.markdown("""
                #### Business Process Impact
                - **Process Efficiency**: Gaps indicate bottlenecks in object lifecycle management
                - **Resource Optimization**: Identifies resource allocation issues across object types
                - **Compliance Management**: Highlights regulatory risks in object handling
                - **Process Optimization**: Guides improvement of object-centric process flows
                """)

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