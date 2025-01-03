import traceback

import streamlit as st
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from functools import lru_cache
import warnings
import logging
from pathlib import Path


warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OutlierMetrics:
    z_score: float
    is_outlier: bool
    details: Dict[str, Any]


class UnfairOCELAnalyzer:
    """Enhanced analyzer for identifying unfairness and outliers in process mining"""

    def __init__(self, ocel_path: str):
        """Initialize analyzer with OCEL file path"""
        try:
            logger.info(f"Initializing UnfairOCELAnalyzer with {ocel_path}")
            # Load OCEL data
            with open(ocel_path, 'r', encoding='utf-8') as f:
                self.ocel_data = json.load(f)

            if not isinstance(self.ocel_data, dict):
                raise ValueError(f"Invalid OCEL data type: {type(self.ocel_data)}")

            if 'ocel:events' not in self.ocel_data:
                raise ValueError("Missing ocel:events in data")

            # Initialize caches and metrics
            self._metrics_cache = {}
            self._df_cache = {}
            self._trace_cache = {}
            self.outlier_cache = {}
            self.failure_patterns = set()

            # Process data efficiently
            self._process_data()
            self._process_outliers()

        except Exception as e:
            logger.error(f"Error initializing analyzer: {str(e)}")
            raise

    def _calculate_resource_metrics(self):
        """Calculate resource metrics"""
        return {
            'workload': self.relationships_df.groupby('resource').size(),
            'avg_duration': self.relationships_df.groupby('resource')['timestamp'].agg(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
        }

    def _calculate_time_metrics(self):
        """Calculate time-based metrics"""
        return {
            'activity_durations': self.relationships_df.groupby('activity')['timestamp'].agg(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
        }

    def _calculate_case_metrics(self):
        """Calculate case-based metrics"""
        return {
            'complexity': self.relationships_df.groupby('case_id')['activity'].nunique(),
            'duration': self.relationships_df.groupby('case_id')['timestamp'].agg(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
        }

    def _calculate_handover_metrics(self):
        """Calculate handover metrics between resources"""
        return {
            'handovers': self.relationships_df.groupby('case_id')['resource'].agg(list).apply(
                lambda x: len([i for i in range(len(x) - 1) if x[i] != x[i + 1]])
            )
        }

    def _create_resource_plot(self, metrics):
        """Create resource discrimination plot"""
        fig, ax = plt.subplots()
        metrics['workload'].plot(kind='bar', ax=ax)
        ax.set_title('Resource Workload Distribution')
        return fig

    def _create_time_plot(self, metrics):
        """Create time bias plot"""
        fig, ax = plt.subplots()
        metrics['activity_durations'].plot(kind='bar', ax=ax)
        ax.set_title('Activity Duration Distribution')
        return fig

    def _create_case_plot(self, metrics):
        """Create case priority plot"""
        fig, ax = plt.subplots()
        metrics['complexity'].plot(kind='hist', ax=ax)
        ax.set_title('Case Complexity Distribution')
        return fig

    def _create_handover_plot(self, metrics):
        """Create handover analysis plot"""
        fig, ax = plt.subplots()
        metrics['handovers'].plot(kind='hist', ax=ax)
        ax.set_title('Handover Distribution')
        return fig

    def _process_data(self):
        """Process OCEL data into efficient DataFrame format"""
        try:
            logger.info("Processing OCEL data")
            events_data = []
            relationships_data = []
            trace_data = []

            for event in self.ocel_data['ocel:events']:
                event_base = {
                    'event_id': event['ocel:id'],
                    'timestamp': pd.to_datetime(event['ocel:timestamp']),
                    'activity': event['ocel:activity'],
                    'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                    'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown')
                }

                trace_data.append({
                    **event_base,
                    'objects': [obj['id'] for obj in event.get('ocel:objects', [])],
                    'raw_event': event
                })

                events_data.append(event_base)

                for obj in event.get('ocel:objects', []):
                    relationships_data.append({
                        **event_base,
                        'object_id': obj['id'],
                        'object_type': obj.get('type', 'Unknown')
                    })

            if not events_data:
                raise ValueError("No events data found in OCEL file")

            # Create optimized DataFrames
            self.events_df = pd.DataFrame(events_data)
            self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'])
            self.events_df.set_index('event_id', inplace=True)

            self.relationships_df = pd.DataFrame(relationships_data)
            self.relationships_df['timestamp'] = pd.to_datetime(self.relationships_df['timestamp'])
            self.relationships_df.sort_values(['case_id', 'timestamp'], inplace=True)

            self.trace_df = pd.DataFrame(trace_data)
            self._build_trace_index()

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

    def _build_trace_index(self):
        """Build indices for quick trace lookups"""
        logger.info("Building trace indices")
        self.resource_events = {}
        self.case_events = {}
        self.activity_events = {}

        for _, row in self.trace_df.iterrows():
            # Index by resource
            if row['resource'] not in self.resource_events:
                self.resource_events[row['resource']] = []
            self.resource_events[row['resource']].append(row['event_id'])

            # Index by case
            if row['case_id'] not in self.case_events:
                self.case_events[row['case_id']] = []
            self.case_events[row['case_id']].append(row['event_id'])

            # Index by activity
            if row['activity'] not in self.activity_events:
                self.activity_events[row['activity']] = []
            self.activity_events[row['activity']].append(row['event_id'])

    def _process_outliers(self):
        """Process and detect outliers in event data"""
        try:
            logger.info("Processing outliers")
            self.outliers = {
                'duration': self._detect_duration_outliers(),
                'resource_load': self._detect_resource_outliers(),
                'case_complexity': self._detect_case_outliers(),
                'failures': self._detect_failure_patterns()
            }
        except Exception as e:
            logger.error(f"Error processing outliers: {str(e)}")
            raise

    def _detect_duration_outliers(self) -> Dict[str, OutlierMetrics]:
        # Convert timestamps to minutes for better granularity
        duration_data = self.relationships_df.groupby(['activity', 'case_id']).agg({
            'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 60  # Converting to minutes
        }).reset_index()

        # Add activity-wise business thresholds (in minutes)
        business_thresholds = {
            'Trade Initiated': 30,
            'Market Data Validation': 45,
            'Quote Provided': 20,
            # Add other activities...
        }

        outliers = {}
        for activity in duration_data['activity'].unique():
            durations = duration_data[duration_data['activity'] == activity]['timestamp']
            if len(durations) > 1:
                z_scores = np.abs(stats.zscore(durations))
                threshold = business_thresholds.get(activity, 60)  # Default 60 min

                outliers[activity] = OutlierMetrics(
                    z_score=float(np.mean(z_scores)),
                    is_outlier=any(durations > threshold),
                    details={
                        'mean_duration_minutes': float(np.mean(durations)),
                        'outlier_cases': duration_data[
                            (duration_data['activity'] == activity) &
                            (durations > threshold)
                            ]['case_id'].tolist(),
                        'threshold_minutes': threshold
                    }
                )
        return outliers

    def _detect_resource_outliers(self) -> Dict[str, OutlierMetrics]:
        """Detect resource workload outliers"""
        resource_loads = self.relationships_df.groupby('resource').size()
        if len(resource_loads) > 1:
            z_scores = np.abs(stats.zscore(resource_loads))

            outliers = {}
            for resource, workload in resource_loads.items():
                z_score = z_scores[resource]
                outliers[resource] = OutlierMetrics(
                    z_score=float(z_score),
                    is_outlier=z_score > 3,
                    details={
                        'workload': int(workload),  # Changed from 'load'
                        'percentage': float(workload / len(self.relationships_df) * 100),
                        'avg_duration': float(self.relationships_df[
                                                  self.relationships_df['resource'] == resource
                                                  ].groupby('case_id')['timestamp'].agg(
                            lambda x: (x.max() - x.min()).total_seconds() / 3600
                        ).mean())
                    }
                )

            return outliers
        return {}

    def _detect_case_outliers(self) -> Dict[str, OutlierMetrics]:
        case_metrics = self.relationships_df.groupby('case_id').agg({
            'activity': 'nunique',  # Changed from lambda
            'resource': 'nunique',
            'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600,
            'object_type': 'nunique'
        })

        # Adjusted thresholds based on OCEL log patterns
        thresholds = {
            'activity': 10,  # Changed from activities
            'resource': 3,
            'timestamp': 24,
            'object_type': 2
        }

        outliers = {}
        for metric in case_metrics.columns:  # Using actual DataFrame columns
            data = case_metrics[metric]
            threshold = thresholds.get(metric, np.percentile(data, 95))

            outliers[f'case_{metric}'] = OutlierMetrics(
                z_score=float(np.mean(data > threshold)),
                is_outlier=any(data > threshold),
                details={
                    'outlier_cases': case_metrics[data > threshold].index.tolist(),
                    'threshold': threshold,
                    'max_value': float(data.max()),
                    'complexity_level': 'High' if data.max() > threshold * 1.5 else 'Medium'
                }
            )
        return outliers

    def _detect_failure_patterns(self) -> Dict[str, Any]:
        expected_flow = {
            'Trade': ['Trade Initiated', 'Market Data Validation', 'Quote Provided'],
            'Market': ['Quote Requested', 'Market Making', 'Quote Provided'],
            'Risk': ['Risk Assessment', 'Risk Validation', 'Risk Report']
        }

        failures = {
            'incomplete_cases': [],
            'long_running': [],
            'resource_switches': [],
            'rework_activities': [],
            'sequence_violations': []  # New category
        }

        # Enhanced incomplete case detection
        for case_id in self.relationships_df['case_id'].unique():
            case_data = self.relationships_df[self.relationships_df['case_id'] == case_id]
            case_activities = case_data['activity'].tolist()
            object_types = case_data['object_type'].unique()

            # Check sequence violations
            for obj_type, expected_sequence in expected_flow.items():
                if obj_type in object_types:
                    actual_sequence = [act for act in case_activities if act in expected_sequence]
                    if actual_sequence != expected_sequence:
                        failures['sequence_violations'].append({
                            'case_id': case_id,
                            'object_type': obj_type,
                            'expected': expected_sequence,
                            'actual': actual_sequence
                        })

        # Add timing thresholds for long-running cases (in hours)
        timing_thresholds = {
            'Trade': 24,
            'Market': 12,
            'Risk': 48
        }

        return failures

    def create_outlier_visualizations(self) -> Dict[str, go.Figure]:
        """Create visualizations for outlier analysis"""
        figs = {}

        try:
            # Resource workload outliers
            if self.outliers.get('resource_load'):
                resource_data = []
                for resource, metrics in self.outliers['resource_load'].items():
                    resource_data.append({
                        'Resource': resource,
                        'Workload': metrics.details['workload'],
                        'Z-Score': metrics.z_score,
                        'Is Outlier': metrics.is_outlier
                    })

                if resource_data:
                    figs['resource_outliers'] = px.scatter(
                        resource_data,
                        x='Resource',
                        y='Workload',
                        size='Z-Score',
                        color='Is Outlier',
                        title='Resource Workload Distribution'
                    )

            # Duration outliers
            if self.outliers.get('duration'):
                duration_data = []
                for activity, metrics in self.outliers['duration'].items():
                    duration_data.append({
                        'Activity': activity,
                        'Z-Score': metrics.z_score,
                        'Is Outlier': metrics.is_outlier
                    })

                if duration_data:
                    figs['duration_outliers'] = px.scatter(
                        duration_data,
                        x='Activity',
                        y='Z-Score',
                        color='Is Outlier',
                        title='Activity Duration Outliers'
                    )

            # Case complexity outliers
            if self.outliers.get('case_complexity'):
                case_data = []
                for metric, metrics in self.outliers['case_complexity'].items():
                    if metrics.details.get('outlier_cases'):
                        case_data.append({
                            'Metric': metric,
                            'Z-Score': metrics.z_score,
                            'Outlier Count': len(metrics.details['outlier_cases'])
                        })

                if case_data:
                    figs['case_outliers'] = px.bar(
                        case_data,
                        x='Metric',
                        y='Outlier Count',
                        color='Z-Score',
                        title='Case Complexity Outliers'
                    )

            # Failure patterns
            if self.outliers.get('failures'):
                failure_data = [
                    {'Pattern': 'Incomplete Cases',
                     'Count': len(self.outliers['failures'].get('incomplete_cases', []))},
                    {'Pattern': 'Long Running', 'Count': len(self.outliers['failures'].get('long_running', []))},
                    {'Pattern': 'Resource Switches',
                     'Count': len(self.outliers['failures'].get('resource_switches', []))},
                    {'Pattern': 'Rework Activities',
                     'Count': len(self.outliers['failures'].get('rework_activities', []))}
                ]

                if failure_data:
                    figs['failure_patterns'] = px.bar(
                        failure_data,
                        x='Pattern',
                        y='Count',
                        title='Process Failure Patterns'
                    )

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

        return figs

    def get_analysis_plots(self) -> Tuple[Dict[str, plt.Figure], Dict]:
        """Generate all analysis plots and metrics"""
        try:
            metrics = {
                'resource': self._calculate_resource_metrics(),
                'time': self._calculate_time_metrics(),
                'case': self._calculate_case_metrics(),
                'handover': self._calculate_handover_metrics()
            }

            plots = {
                'resource_discrimination': self._create_resource_plot(metrics['resource']),
                'time_bias': self._create_time_plot(metrics['time']),
                'case_priority': self._create_case_plot(metrics['case']),
                'handover': self._create_handover_plot(metrics['handover'])
            }

            return plots, metrics

        except Exception as e:
            logger.error(f"Error generating analysis plots: {str(e)}")
            return {}, {}

    def display_enhanced_analysis(self):
        """Display enhanced analysis safely"""
        try:
            plots, metrics = self.get_analysis_plots()
            outlier_plots = self.create_outlier_visualizations()

            tabs = st.tabs(["Resource Analysis", "Time Analysis", "Case Analysis", "Failure Patterns"])

            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    if 'resource_outliers' in outlier_plots:
                        st.plotly_chart(outlier_plots['resource_outliers'], use_container_width=True)
                    else:
                        st.info("No resource outliers detected")
                with col2:
                    if 'resource_discrimination' in plots:
                        st.pyplot(plots['resource_discrimination'])
                    else:
                        st.info("No resource discrimination data available")

            with tabs[1]:
                col1, col2 = st.columns(2)
                with col1:
                    if 'duration_outliers' in outlier_plots:
                        st.plotly_chart(outlier_plots['duration_outliers'], use_container_width=True)
                    if 'time_bias' in plots:
                        st.pyplot(plots['time_bias'])
                with col2:
                    if self.outliers.get('duration'):
                        st.write("Duration Outliers Found:")
                        for activity, metrics in self.outliers['duration'].items():
                            if metrics.is_outlier:
                                st.warning(
                                    f"{activity}: {len(metrics.details['outlier_cases'])} cases exceeded threshold")
                                with st.expander("View Details"):
                                    st.write(f"Mean Duration: {metrics.details['mean_duration_minutes']:.2f} minutes")
                                    st.write(f"Threshold: {metrics.details['threshold_minutes']} minutes")
                                    st.write("Affected Cases:", metrics.details['outlier_cases'])

            with tabs[2]:
                col1, col2 = st.columns(2)
                with col1:
                    if 'case_outliers' in outlier_plots:
                        st.plotly_chart(outlier_plots['case_outliers'], use_container_width=True)
                    if 'case_priority' in plots:
                        st.pyplot(plots['case_priority'])
                with col2:
                    if self.outliers.get('case_complexity'):
                        st.write("Case Complexity Outliers:")
                        for metric, details in self.outliers['case_complexity'].items():
                            if details.is_outlier:
                                st.warning(f"{metric}: {len(details.details['outlier_cases'])} complex cases")
                                with st.expander("View Details"):
                                    st.write(f"Threshold: {details.details['threshold']}")
                                    st.write(f"Max Value: {details.details['max_value']}")
                                    st.write(f"Complexity Level: {details.details['complexity_level']}")
                                    st.write("Complex Cases:", details.details['outlier_cases'])

            with tabs[3]:
                col1, col2 = st.columns(2)
                with col1:
                    if 'failure_patterns' in outlier_plots:
                        st.plotly_chart(outlier_plots['failure_patterns'], use_container_width=True)
                with col2:
                    if self.outliers.get('failures'):
                        failures = self.outliers['failures']
                        st.write("Failure Patterns Detected:")

                        # Display incomplete cases
                        if failures.get('incomplete_cases'):
                            with st.expander(f"Incomplete Cases ({len(failures['incomplete_cases'])})"):
                                for case in failures['incomplete_cases']:
                                    st.write(f"Case ID: {case['case_id']}")
                                    if 'missing_activities' in case:
                                        st.write("Missing Activities:", case['missing_activities'])

                        # Display long running cases
                        if failures.get('long_running'):
                            with st.expander(f"Long Running Cases ({len(failures['long_running'])})"):
                                for case in failures['long_running']:
                                    st.write(f"Case ID: {case['case_id']}")
                                    st.write(f"Duration: {case.get('duration', 'N/A')} hours")

                        # Display sequence violations
                        if failures.get('sequence_violations'):
                            with st.expander(f"Sequence Violations ({len(failures['sequence_violations'])})"):
                                for violation in failures['sequence_violations']:
                                    st.write(f"Case ID: {violation['case_id']}")
                                    st.write(f"Object Type: {violation['object_type']}")
                                    st.write("Expected:", violation['expected'])
                                    st.write("Actual:", violation['actual'])

        except Exception as e:
            st.error(f"Error in enhanced analysis: {str(e)}")
            st.error(f"Detailed error:\n{traceback.format_exc()}")