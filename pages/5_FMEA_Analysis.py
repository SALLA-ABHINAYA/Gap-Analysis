# pages/5_FMEA_Analysis.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Increase pandas display limit
pd.set_option("styler.render.max_elements", 600000)


@dataclass
class OCELFailureMode:
    """Represents a failure mode in OCEL analysis"""
    id: str
    activity: str
    object_type: str
    description: str
    severity: int = 0
    likelihood: int = 0
    detectability: int = 0
    effects: List[str] = field(default_factory=list)
    causes: List[str] = field(default_factory=list)


class OCELEnhancedFMEA:
    """FMEA analyzer specifically for OCEL logs"""

    def __init__(self, ocel_data: Dict):
        self.ocel_data = ocel_data
        self.events = pd.DataFrame(ocel_data['ocel:events'])
        self.object_types = ocel_data.get('ocel:object-types', [])
        self.activity_stats = self._compute_activity_stats()
        self.failure_modes = []

    def _compute_activity_stats(self) -> Dict:
        """Compute activity statistics for FMEA calculations"""
        stats = {}
        for _, event in self.events.iterrows():
            activity = event['ocel:activity']
            if activity not in stats:
                stats[activity] = {
                    'count': 0,
                    'objects': [],
                    'resources': set(),
                    'completion_rate': 0,
                    'error_rate': 0
                }
            stats[activity]['count'] += 1
            if 'ocel:objects' in event:
                stats[activity]['objects'].extend([obj['id'] for obj in event['ocel:objects']])
            if 'ocel:attributes' in event and 'resource' in event['ocel:attributes']:
                stats[activity]['resources'].add(event['ocel:attributes']['resource'])

        return stats

    def calculate_severity(self, failure_mode: OCELFailureMode) -> int:
        """Calculate severity rating (1-10) based on activity and object type"""
        base_severity = 5

        # Increase severity for critical object types
        if failure_mode.object_type == 'Trade':
            base_severity += 2
        elif failure_mode.object_type in ['Risk', 'Settlement']:
            base_severity += 1

        # Check activity criticality
        critical_keywords = ['trade', 'risk', 'settlement', 'compliance']
        if any(keyword in failure_mode.activity.lower() for keyword in critical_keywords):
            base_severity += 1

        return min(base_severity, 10)

    def calculate_likelihood(self, failure_mode: OCELFailureMode) -> int:
        """Calculate likelihood rating (1-10) based on historical data"""
        base_likelihood = 5

        activity_stats = self.activity_stats.get(failure_mode.activity, {})

        # Assess based on frequency
        activity_count = activity_stats.get('count', 0)
        if activity_count > 100:
            base_likelihood += 2
        elif activity_count > 50:
            base_likelihood += 1

        # Assess based on complexity
        object_count = len(activity_stats.get('objects', []))
        if object_count > 10:
            base_likelihood += 2
        elif object_count > 5:
            base_likelihood += 1

        return min(base_likelihood, 10)

    def calculate_detectability(self, failure_mode: OCELFailureMode) -> int:
        """Calculate detectability rating (1-10) where 1 is easily detectable"""
        base_detectability = 5

        activity_stats = self.activity_stats.get(failure_mode.activity, {})

        # Adjust based on monitoring capabilities
        resource_count = len(activity_stats.get('resources', set()))
        if resource_count > 3:
            base_detectability += 1

        # Adjust based on object type
        if failure_mode.object_type == 'Trade':
            base_detectability -= 1
        elif failure_mode.object_type == 'Risk':
            base_detectability -= 2

        return min(max(base_detectability, 1), 10)

    def identify_failure_modes(self) -> List[Dict]:
        """Identify and analyze potential failure modes"""
        failure_modes = []

        for activity in self.activity_stats:
            object_types = set()
            events_for_activity = self.events[self.events['ocel:activity'] == activity]

            for event in events_for_activity.to_dict('records'):
                if 'ocel:objects' in event:
                    for obj in event['ocel:objects']:
                        object_types.add(obj.get('type', 'Unknown'))

            for obj_type in object_types:
                failure_mode = OCELFailureMode(
                    id=f"FM_{len(failure_modes)}",
                    activity=activity,
                    object_type=obj_type,
                    description=f"Potential failure in {activity} for {obj_type}"
                )

                # Calculate metrics
                failure_mode.severity = self.calculate_severity(failure_mode)
                failure_mode.likelihood = self.calculate_likelihood(failure_mode)
                failure_mode.detectability = self.calculate_detectability(failure_mode)

                # Add effects and causes
                failure_mode.effects = [
                    f"Impact on {obj_type} processing",
                    "Potential process delay",
                    "Data quality issues"
                ]
                failure_mode.causes = [
                    "Process complexity",
                    "Resource constraints",
                    "System limitations"
                ]

                failure_modes.append({
                    'id': failure_mode.id,
                    'activity': failure_mode.activity,
                    'object_type': failure_mode.object_type,
                    'description': failure_mode.description,
                    'severity': failure_mode.severity,
                    'likelihood': failure_mode.likelihood,
                    'detectability': failure_mode.detectability,
                    'rpn': failure_mode.severity * failure_mode.likelihood * failure_mode.detectability,
                    'effects': failure_mode.effects,
                    'causes': failure_mode.causes
                })

        return sorted(failure_modes, key=lambda x: x['rpn'], reverse=True)


def paginate_dataframe(df: pd.DataFrame, page_size: int = 1000) -> pd.DataFrame:
    """Handle pagination for large dataframes"""
    n_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)

    page = st.sidebar.number_input(
        'Page',
        min_value=1,
        max_value=n_pages,
        value=1
    )

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))

    return df.iloc[start_idx:end_idx].copy()


def display_fmea_analysis(fmea_results: List[Dict]):
    """Display FMEA analysis results with pagination"""
    try:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Failure Modes", len(fmea_results))
        with col2:
            high_risk = sum(1 for r in fmea_results if r['rpn'] > 200)
            st.metric("High Risk (RPN > 200)", high_risk)
        with col3:
            medium_risk = sum(1 for r in fmea_results if 100 < r['rpn'] <= 200)
            st.metric("Medium Risk", medium_risk)
        with col4:
            low_risk = sum(1 for r in fmea_results if r['rpn'] <= 100)
            st.metric("Low Risk", low_risk)

        # RPN Distribution
        df = pd.DataFrame(fmea_results)
        fig = go.Figure(data=go.Histogram(
            x=df['rpn'],
            nbinsx=20,
            name='RPN Distribution'
        ))
        fig.add_vline(x=200, line_dash="dash", line_color="red",
                      annotation_text="Critical Threshold")
        fig.update_layout(
            title='RPN Distribution',
            xaxis_title='RPN',
            yaxis_title='Count'
        )
        st.plotly_chart(fig)

        # Detailed analysis table
        st.subheader("Failure Modes Analysis")

        display_columns = [
            'activity', 'object_type', 'description',
            'severity', 'likelihood', 'detectability', 'rpn'
        ]

        display_df = df[display_columns].copy()
        paginated_df = paginate_dataframe(display_df)

        # Apply styling only to paginated data
        styled_df = paginated_df.style.background_gradient(
            subset=['rpn'],
            cmap='Reds',
            vmin=df['rpn'].min(),
            vmax=df['rpn'].max()
        )

        st.dataframe(styled_df)

        # Critical recommendations
        st.subheader("Critical Risk Recommendations")
        high_risk_items = [r for r in fmea_results if r['rpn'] > 200]

        if high_risk_items:
            for item in high_risk_items:
                with st.expander(f"{item['activity']} (RPN: {item['rpn']})"):
                    st.write("**Description:**", item['description'])
                    cols = st.columns(3)
                    with cols[0]:
                        st.write(f"**Severity:** {item['severity']}")
                    with cols[1]:
                        st.write(f"**Likelihood:** {item['likelihood']}")
                    with cols[2]:
                        st.write(f"**Detectability:** {item['detectability']}")

                    if 'effects' in item:
                        st.write("**Effects:**")
                        for effect in item['effects']:
                            st.write(f"- {effect}")

                    if 'causes' in item:
                        st.write("**Causes:**")
                        for cause in item['causes']:
                            st.write(f"- {cause}")
        else:
            st.info("No high-risk failure modes identified")

    except Exception as e:
        logger.error(f"Error in display_fmea_analysis: {str(e)}", exc_info=True)
        st.error(f"Error displaying analysis: {str(e)}")


# Page execution
st.set_page_config(page_title="FMEA Analysis", layout="wide")
st.title("FMEA Analysis")

try:
    # Load OCEL data
    with open("ocpm_output/process_data.json", 'r') as f:
        ocel_data = json.load(f)

    # Initialize analyzer and get results
    analyzer = OCELEnhancedFMEA(ocel_data)
    fmea_results = analyzer.identify_failure_modes()

    # Display results
    display_fmea_analysis(fmea_results)

except Exception as e:
    st.error(f"Error in FMEA analysis: {str(e)}")
    logger.error(f"FMEA analysis error: {str(e)}", exc_info=True)