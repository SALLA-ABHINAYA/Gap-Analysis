import traceback
from collections import defaultdict

import streamlit as st
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import warnings
import logging

from utils import get_azure_openai_client

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


class OCPMProcessValidator:
    """Handles validation of process execution using OCPM model and thresholds"""

    def __init__(self):
        """Initialize validator with OCPM model and thresholds"""
        try:
            # Load OCPM model
            with open('ocpm_output/output_ocel.json', 'r') as f:
                self.ocpm_model = json.load(f)

            # Load timing thresholds
            with open('ocpm_output/output_ocel_threshold.json', 'r') as f:
                self.thresholds = json.load(f)

            # Validate both files exist and have required structure
            self._validate_loaded_data()

        except Exception as e:
            logger.error(f"Error initializing OCPMProcessValidator: {str(e)}")
            raise

    def _validate_loaded_data(self):
        """Validates that loaded data has required structure"""
        # Validate OCPM model
        for obj_type, data in self.ocpm_model.items():
            required_keys = {'activities', 'attributes', 'relationships'}
            if not all(key in data for key in required_keys):
                raise ValueError(f"Invalid OCPM model structure for {obj_type}")

        # Validate thresholds
        for obj_type, data in self.thresholds.items():
            required_keys = {'total_duration_hours', 'default_gap_hours', 'activity_thresholds'}
            if not all(key in data for key in required_keys):
                raise ValueError(f"Invalid threshold structure for {obj_type}")

    def get_expected_flow(self) -> Dict[str, List[str]]:
        """Convert OCPM model to expected flow"""
        return {
            obj_type: data['activities']
            for obj_type, data in self.ocpm_model.items()
        }

    def get_timing_thresholds(self) -> Dict[str, Dict]:
        """Convert OCPM thresholds to timing rules"""
        return {
            obj_type: {
                'total_duration': data['total_duration_hours'],
                'activity_gaps': {
                    activity: thresholds['max_gap_after_hours']
                    for activity, thresholds in data['activity_thresholds'].items()
                }
            }
            for obj_type, data in self.thresholds.items()
        }


class UnfairOCELAnalyzer:
    """Enhanced analyzer for identifying unfairness and outliers in process mining"""

    # Update __init__ method in UnfairOCELAnalyzer

    def __init__(self, ocel_path: str):
        """Initialize analyzer with OCEL file path"""
        try:
            logger.info(f"Initializing UnfairOCELAnalyzer with {ocel_path}")

            # Initialize OpenAI client first
            self.client = get_azure_openai_client()
            logger.info("OpenAI client initialized")

            self.process_validator = OCPMProcessValidator()

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

            logger.info("UnfairOCELAnalyzer initialization completed successfully")

        except Exception as e:
            logger.error(f"Error initializing analyzer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _calculate_resource_metrics(self):
        """Calculate resource metrics"""
        return {
            'workload': self.relationships_df.groupby('resource').size(),
            'avg_duration': self.relationships_df.groupby('resource')['timestamp'].agg(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
        }

    def _create_outlier_plots(self):
        """Create all outlier visualization plots"""
        logger.info("Creating outlier visualization plots")
        self.outlier_plots = {}

        try:
            # Resource Outlier Plot
            resource_loads = pd.Series({k: len(v) for k, v in self.resource_events.items()})
            fig = go.Figure(data=[
                go.Scatter(
                    x=list(resource_loads.index),
                    y=list(resource_loads.values),
                    mode='markers',
                    name='Workload',
                    marker=dict(
                        size=15,
                        color='blue',
                        opacity=0.7
                    )
                )
            ])

            fig.update_layout(
                title='Resource Workload Distribution',
                xaxis_title='Resource',
                yaxis_title='Workload'
            )
            self.outlier_plots['resource_outliers'] = fig

            # Duration Outlier Plot
            if 'duration' in self.outliers and self.outliers['duration']:
                # Extract duration metrics from detection method
                duration_data = []
                for activity, metrics in self.outliers['duration'].items():
                    duration_data.append({
                        'Activity': activity,
                        'Z-Score': metrics.z_score,
                        'Is Outlier': metrics.is_outlier,
                        'Violation Rate': metrics.details['violation_rate'],
                        'Total Events': metrics.details['total_events']
                    })

                duration_df = pd.DataFrame(duration_data)

                # Create enhanced visualization
                fig = px.scatter(
                    duration_df,
                    x='Activity',
                    y='Z-Score',
                    color='Is Outlier',
                    size='Violation Rate',
                    hover_data=['Total Events', 'Violation Rate'],
                    title='Activity Duration Outliers (Business Logic Violations)',
                    color_discrete_map={True: 'red', False: 'blue'}
                )

                # Add threshold line and annotations
                fig.add_hline(
                    y=3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Outlier Threshold (Z=3)",
                    annotation_position="bottom right"
                )

                fig.update_layout(
                    xaxis_title="Activity",
                    yaxis_title="Z-Score (Violation Severity)",
                    hovermode="closest",
                    showlegend=True,
                    height=600
                )

                self.outlier_plots['duration_outliers'] = fig

            # Case Outlier Plot
            case_complexities = []
            for case_id, events in self.case_events.items():
                case_complexities.append({
                    'case_id': case_id,
                    'complexity': len(events),
                    'unique_activities': len(set(events))
                })

            case_df = pd.DataFrame(case_complexities)
            fig = px.box(
                case_df,
                y=['complexity', 'unique_activities'],
                title='Case Complexity Distribution'
            )
            self.outlier_plots['case_outliers'] = fig

            # Failure Pattern Plot
            if hasattr(self, 'outliers') and 'failures' in self.outliers:
                failures = self.outliers['failures']
                failure_counts = {
                    'Sequence Violations': len(failures.get('sequence_violations', [])),
                    'Incomplete Cases': len(failures.get('incomplete_cases', [])),
                    'Long Running': len(failures.get('long_running', [])),
                    'Resource Switches': len(failures.get('resource_switches', [])),
                    'Rework': len(failures.get('rework_activities', []))
                }

                fig = go.Figure(data=[
                    go.Bar(
                        x=list(failure_counts.keys()),
                        y=list(failure_counts.values())
                    )
                ])

                fig.update_layout(
                    title='Process Failure Patterns Distribution',
                    xaxis_title='Failure Pattern Type',
                    yaxis_title='Count'
                )
                self.outlier_plots['failure_patterns'] = fig

            logger.info("Successfully created outlier plots")

        except Exception as e:
            logger.error(f"Error creating outlier plots: {str(e)}")
            logger.error(traceback.format_exc())

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

    # Update _process_data method
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

            # Build indices
            self._build_trace_index()

            # Process outliers
            self._process_outliers()

            # Create visualization plots - Add this line
            self._create_outlier_plots()

            logger.info("Successfully processed OCEL data")

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            logger.error(traceback.format_exc())
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
        """Enhanced duration outlier detection using OCPM model and thresholds"""
        try:
            # Create events DataFrame (same as before)
            events_df = pd.DataFrame([{
                'event_id': event['ocel:id'],
                'timestamp': pd.to_datetime(event['ocel:timestamp']),
                'activity': event['ocel:activity'],
                'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                'object_ids': [obj['id'] for obj in event.get('ocel:objects', [])],
                'object_types': list(set(obj.get('type', 'Unknown') for obj in event.get('ocel:objects', [])))
            } for event in self.ocel_data['ocel:events']])

            events_df = events_df.sort_values('timestamp')

            # Get timing thresholds from OCPM validator
            timing_rules = self.process_validator.get_timing_thresholds()

            outliers = {}
            for activity in events_df['activity'].unique():
                activity_events = events_df[events_df['activity'] == activity]

                # Initialize tracking (same as before)
                outlier_events = {
                    'outside_hours': [],
                    'sequence_position': [],
                    'timing_gap': [],
                    'resource_handover': []
                }

                metrics = {
                    'total_events': len(activity_events),
                    'outside_hours_count': 0,
                    'sequence_violations': 0,
                    'timing_violations': 0,
                    'resource_violations': 0
                }

                # Process each event
                for idx, event in activity_events.iterrows():
                    for obj_type in event['object_types']:
                        if obj_type in timing_rules:
                            thresholds = timing_rules[obj_type]
                            activity_threshold = thresholds['activity_gaps'].get(
                                activity,
                                thresholds.get('default_gap_hours', 1)
                            )

                            prev_events = events_df[
                                (events_df['case_id'] == event['case_id']) &
                                (events_df['object_types'].apply(lambda x: obj_type in x)) &
                                (events_df.index < idx)
                                ]

                            if not prev_events.empty:
                                last_event = prev_events.iloc[-1]
                                gap_hours = (event['timestamp'] -
                                             last_event['timestamp']).total_seconds() / 3600

                                if gap_hours > activity_threshold:
                                    # Modified structure to match display expectations
                                    outlier_events['timing_gap'].append({
                                        'event_id': event['event_id'],
                                        'case_id': event['case_id'],
                                        'timestamp': event['timestamp'],
                                        'details': {  # Add details structure
                                            'time_gap_minutes': gap_hours * 60,
                                            'threshold_minutes': activity_threshold * 60,
                                            'previous_event': last_event['event_id'],
                                            'related_objects': event['object_ids']
                                        },
                                        'object_id': event['object_ids'][0] if event['object_ids'] else None,
                                        'objects': event['object_ids']
                                    })
                                    metrics['timing_violations'] += 1

                            # Sequence position check (modified to match expected structure)
                            expected_pos = self._get_expected_activity_index(activity, obj_type)
                            if expected_pos is not None:
                                actual_pos = len(prev_events)
                                if abs(actual_pos - expected_pos) > 1:
                                    outlier_events['sequence_position'].append({
                                        'event_id': event['event_id'],
                                        'case_id': event['case_id'],
                                        'details': {  # Add details structure
                                            'expected_position': expected_pos,
                                            'actual_position': actual_pos,
                                            'sequence_context': prev_events['activity'].tolist()
                                        },
                                        'object_type': obj_type
                                    })
                                    metrics['sequence_violations'] += 1

                # Calculate violation score (same as before)
                violation_score = sum(metrics.values()) / (metrics['total_events'] * 3) if metrics[
                                                                                               'total_events'] > 0 else 0

                outliers[activity] = OutlierMetrics(
                    z_score=float(violation_score * 10),
                    is_outlier=violation_score > 0.3,
                    details={
                        'metrics': metrics,
                        'outlier_events': outlier_events,
                        'violation_rate': violation_score,
                        'total_events': metrics['total_events'],
                        'activity_stats': {  # Added to match expected structure
                            'avg_duration': float(activity_events.groupby('case_id')['timestamp']
                                                  .agg(lambda x: (x.max() - x.min()).total_seconds() / 60).mean()),
                            'resource_distribution': activity_events['resource'].value_counts().to_dict(),
                            'object_type_distribution': pd.Series([ot for ots in activity_events['object_types']
                                                                   for ot in ots]).value_counts().to_dict()
                        }
                    }
                )

            return outliers

        except Exception as e:
            logger.error(f"Error in duration outlier detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _get_expected_activity_index(self, activity: str, object_type: str) -> Optional[int]:
        """
        Determine expected position of activity in process based on OCPM model.

        Args:
            activity: The activity name to find
            object_type: The object type this activity belongs to

        Returns:
            Optional[int]: Index of activity in sequence, or None if not found
        """
        try:
            # Get activity sequence from OCPM model
            activities = self.process_validator.get_expected_flow().get(object_type, [])
            if not activities:
                return None

            return activities.index(activity)
        except ValueError:
            return None

    def _detect_resource_outliers(self) -> Dict[str, OutlierMetrics]:
        """Optimized resource outlier detection with full event traceability"""
        try:
            # Pre-process events into a DataFrame for faster analysis
            events_df = pd.DataFrame([{
                'event_id': event['ocel:id'],
                'timestamp': pd.to_datetime(event['ocel:timestamp']),
                'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                'activity': event['ocel:activity'],
                'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                'object_types': [obj.get('type', 'Unknown') for obj in event.get('ocel:objects', [])]
            } for event in self.ocel_data['ocel:events']])

            # Calculate metrics using vectorized operations
            workload_metrics = events_df.groupby('resource').agg({
                'event_id': 'count',
                'case_id': 'nunique',
                'activity': 'nunique',
                'object_types': lambda x: len(set([item for sublist in x for item in sublist]))
            }).rename(columns={
                'event_id': 'total_events',
                'case_id': 'unique_cases',
                'activity': 'activity_variety',
                'object_types': 'object_variety'
            })

            # Calculate z-scores for all metrics at once using vectorized operations
            z_scores = pd.DataFrame()
            for col in workload_metrics.columns:
                z_scores[col] = (workload_metrics[col] - workload_metrics[col].mean()) / workload_metrics[col].std()

            # Calculate composite z-scores
            composite_z = np.abs(z_scores).mean(axis=1)

            outliers = {}
            for resource in workload_metrics.index:
                # Get resource specific events
                resource_events = events_df[events_df['resource'] == resource]

                outliers[resource] = OutlierMetrics(
                    z_score=float(composite_z[resource]),
                    is_outlier=composite_z[resource] > 3,
                    details={
                        'metrics': workload_metrics.loc[resource].to_dict(),
                        'events': {
                            'all_events': resource_events['event_id'].tolist(),
                            'cases': resource_events['case_id'].unique().tolist(),
                            'activities': resource_events['activity'].unique().tolist(),
                            'object_types': list(set([t for types in resource_events['object_types'] for t in types]))
                        },
                        'outlier_patterns': {
                            'high_workload': workload_metrics.loc[resource, 'total_events'] > workload_metrics[
                                'total_events'].mean() + 2 * workload_metrics['total_events'].std(),
                            'high_variety': workload_metrics.loc[resource, 'activity_variety'] > workload_metrics[
                                'activity_variety'].mean() + 2 * workload_metrics['activity_variety'].std(),
                        }
                    }
                )

            return outliers

        except Exception as e:
            logger.error(f"Error in resource outlier detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _detect_case_outliers(self) -> Dict[str, OutlierMetrics]:
        """Analyzes case complexity outliers using multivariate metrics and z-scores.

            Process:
            1. Converts OCEL events into a DataFrame with key case metrics:
               - Total events per case
               - Unique activities per case
               - Resource variety per case
               - Case duration in hours
               - Object type variety

            2. Calculates z-scores for all metrics simultaneously using vectorized operations
               for performance optimization.

            3. Creates composite z-score by averaging absolute z-scores across all metrics
               to identify overall case complexity outliers.

            4. For each case, captures:
               - Core metrics (events, activities, resources, duration)
               - Complete event sequence with activity and resource patterns
               - Temporal information (start, end, duration)
               - Specific outlier patterns (high complexity, high variety, long duration)
               - Full event traceability for detailed investigation

            Outlier Definition:
            - Cases with composite z-score > 3 are flagged as outliers
            - Individual metrics contributing to outlier status are tracked
            - Both unusually complex and unusually simple cases can be identified

            Returns:
                Dict[str, OutlierMetrics]: Dictionary mapping case IDs to OutlierMetrics objects containing:
                    - z_score: Composite z-score for the case
                    - is_outlier: Boolean indicating if case is an outlier (z-score > 3)
                    - details: Dict with complete metrics, events, temporal data and outlier patterns
            """
        try:
            # Pre-process events into a DataFrame for faster analysis
            events_df = pd.DataFrame([{
                'event_id': event['ocel:id'],
                'timestamp': pd.to_datetime(event['ocel:timestamp']),
                'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                'activity': event['ocel:activity'],
                'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                'object_types': [obj.get('type', 'Unknown') for obj in event.get('ocel:objects', [])]
            } for event in self.ocel_data['ocel:events']])

            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.expand_frame_repr', False)

            logger.info(" events_df Build up [start]")
            logger.info("\n%s", events_df)
            logger.info(" events_df Build up [end]")

            # Calculate case metrics using vectorized operations
            case_metrics = events_df.groupby('case_id').agg({
                'event_id': 'count',
                'activity': 'nunique',
                'resource': 'nunique',
                'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600,
                'object_types': lambda x: len(set([item for sublist in x for item in sublist]))
            }).rename(columns={
                'event_id': 'total_events',
                'activity': 'activity_variety',
                'resource': 'resource_variety',
                'timestamp': 'duration_hours',
                'object_types': 'object_variety'
            })

            logger.info(" case_metrics Build up [start]")
            logger.info("\n%s", case_metrics)
            logger.info(" case_metrics Build up [end]")

            # Calculate z-scores for all metrics at once
            z_scores = pd.DataFrame()
            for col in case_metrics.columns:
                z_scores[col] = (case_metrics[col] - case_metrics[col].mean()) / case_metrics[col].std()

            # Calculate composite z-scores
            composite_z = np.abs(z_scores).mean(axis=1)

            outliers = {}
            for case_id in case_metrics.index:
                # Get case specific events
                case_events = events_df[events_df['case_id'] == case_id]

                outliers[case_id] = OutlierMetrics(
                    z_score=float(composite_z[case_id]),
                    is_outlier=composite_z[case_id] > 3,
                    details={
                        'metrics': case_metrics.loc[case_id].to_dict(),
                        'events': {
                            'all_events': case_events['event_id'].tolist(),
                            'activity_sequence': case_events.sort_values('timestamp')['activity'].tolist(),
                            'resource_sequence': case_events.sort_values('timestamp')['resource'].tolist(),
                            'object_types': list(set([t for types in case_events['object_types'] for t in types]))
                        },
                        'temporal': {
                            'start_time': case_events['timestamp'].min(),
                            'end_time': case_events['timestamp'].max(),
                            'duration_hours': case_metrics.loc[case_id, 'duration_hours']
                        },
                        'outlier_patterns': {
                            'high_complexity': case_metrics.loc[case_id, 'total_events'] > case_metrics[
                                'total_events'].mean() + 2 * case_metrics['total_events'].std(),
                            'high_variety': case_metrics.loc[case_id, 'activity_variety'] > case_metrics[
                                'activity_variety'].mean() + 2 * case_metrics['activity_variety'].std(),
                            'long_duration': case_metrics.loc[case_id, 'duration_hours'] > case_metrics[
                                'duration_hours'].mean() + 2 * case_metrics['duration_hours'].std()
                        }
                    }
                )

            return outliers

        except Exception as e:
            logger.error(f"Error in case outlier detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _detect_failure_patterns(self) -> Dict[str, Any]:
        """Enhanced failure pattern detection with full event traceability"""
        try:
            expected_flow = self.process_validator.get_expected_flow()
            logger.info(f"Expected flow derived as : {expected_flow}")

            timing_thresholds = self.process_validator.get_timing_thresholds()
            logger.info(f"Timing Thresholds derived as : {timing_thresholds}")

            failures = {
                'sequence_violations': [],
                'incomplete_cases': [],
                'long_running': [],
                'resource_switches': [],
                'rework_activities': [],
                'timing_violations': []
            }

            # Process each case
            for case_id, case_events in self.case_events.items():
                logger.info(f"Running for case_id | case_events : {case_id} | {case_events}")
                case_data = pd.DataFrame([
                    {
                        'event_id': event['ocel:id'],
                        'timestamp': pd.to_datetime(event['ocel:timestamp']),
                        'activity': event['ocel:activity'],
                        'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                        'object_types': [obj.get('type', 'Unknown') for obj in event.get('ocel:objects', [])],
                        'object_ids': [obj['id'] for obj in event.get('ocel:objects', [])]
                    }
                    for event in self.ocel_data['ocel:events']
                    if event.get('ocel:attributes', {}).get('case_id') == case_id
                ]).sort_values('timestamp')

                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.expand_frame_repr', False)

                logger.info(" Case_data Build up [start]")
                logger.info("\n%s", case_data)
                logger.info(" Case_data Build up [end]")

                # Track sequence violations
                for obj_type, expected_sequence in expected_flow.items():
                    if obj_type in set().union(*case_data['object_types']):
                        actual_sequence = case_data[case_data['object_types'].apply(lambda x: obj_type in x)][
                            'activity'].tolist()

                        if actual_sequence != expected_sequence:
                            sequence_violation = {
                                'case_id': case_id,
                                'object_type': obj_type,
                                'expected_sequence': expected_sequence,
                                'actual_sequence': actual_sequence,
                                'events': case_data[case_data['object_types'].apply(lambda x: obj_type in x)][
                                    'event_id'].tolist(),
                                'first_violation': None,
                                'missing_activities': [],
                                'wrong_order_activities': []
                            }

                            # Find first violation and missing activities
                            expected_set = set(expected_sequence)
                            actual_set = set(actual_sequence)
                            sequence_violation['missing_activities'] = list(expected_set - actual_set)

                            # Find wrong order activities
                            for i in range(len(actual_sequence)):
                                if i < len(expected_sequence) and actual_sequence[i] != expected_sequence[i]:
                                    sequence_violation['wrong_order_activities'].append({
                                        'position': i,
                                        'expected': expected_sequence[i],
                                        'actual': actual_sequence[i],
                                        'event_id': case_data.iloc[i]['event_id']
                                    })
                                    if sequence_violation['first_violation'] is None:
                                        sequence_violation['first_violation'] = {
                                            'event_id': case_data.iloc[i]['event_id'],
                                            'timestamp': case_data.iloc[i]['timestamp'],
                                            'resource': case_data.iloc[i]['resource']
                                        }

                            failures['sequence_violations'].append(sequence_violation)

                # Track incomplete cases by leveraging sequence violation analysis
                for violation in failures['sequence_violations']:
                    if violation['missing_activities']:
                        case_id = violation['case_id']
                        obj_type = violation['object_type']

                        # Filter events for specific case and object type
                        case_events = case_data[
                            (case_data['event_id'].str.startswith(f"{case_id}_")) &
                            (case_data['object_types'].apply(lambda x: obj_type in x))
                            ]

                        if not case_events.empty:
                            failures['incomplete_cases'].append({
                                'case_id': case_id,
                                'object_type': obj_type,
                                'missing_activities': violation['missing_activities'],
                                'completed_activities': case_events['activity'].tolist(),
                                'events': case_events['event_id'].tolist(),
                                'last_event': {
                                    'event_id': case_events.iloc[-1]['event_id'],
                                    'activity': case_events.iloc[-1]['activity'],
                                    'timestamp': case_events.iloc[-1]['timestamp'],
                                    'resource': case_events.iloc[-1]['resource']
                                }
                            })

                # Track timing violations
                for obj_type, timing_rules in timing_thresholds.items():
                    logger.info(f"Running timing violations for obj_type | timing_rules : {obj_type} | {timing_rules}")
                    obj_type_events = case_data[case_data['object_types'].apply(lambda x: obj_type in x)]
                    if not obj_type_events.empty:
                        case_duration = (obj_type_events['timestamp'].max() -
                                         obj_type_events['timestamp'].min()).total_seconds() / 3600

                        if case_duration > timing_rules['total_duration']:
                            timing_violation = {
                                'case_id': case_id,
                                'object_type': obj_type,
                                'actual_duration': case_duration,
                                'threshold': timing_rules['total_duration'],
                                'events': obj_type_events['event_id'].tolist(),
                                'activity_gaps': []
                            }

                            # Check activity-specific gaps
                            # Track timing violations
                            for obj_type, timing_rules in timing_thresholds.items():
                                # First, get all events for this case and object type
                                obj_type_events = case_data[case_data['object_types'].apply(lambda x: obj_type in x)]

                                if not obj_type_events.empty:
                                    # Check total duration for this case
                                    case_duration = (obj_type_events['timestamp'].max() -
                                                     obj_type_events['timestamp'].min()).total_seconds() / 3600

                                    timing_violation = None
                                    if case_duration > timing_rules['total_duration']:
                                        # Initialize violation record if we exceed total duration
                                        timing_violation = {
                                            'case_id': case_id,  # Important: Tracking which case had the violation
                                            'object_type': obj_type,
                                            'actual_duration': case_duration,
                                            'threshold': timing_rules['total_duration'],
                                            'events': obj_type_events['event_id'].tolist(),
                                            'activity_gaps': []
                                        }

                                    # Check gaps between activities in the sequence
                                    for activity, max_gap in timing_rules['activity_gaps'].items():
                                        # Find all occurrences of this activity
                                        activity_events = obj_type_events[obj_type_events['activity'] == activity]

                                        for idx in activity_events.index:
                                            # Get all events that happened before this activity in this case
                                            previous_events = obj_type_events[obj_type_events.index < idx]

                                            if not previous_events.empty:
                                                # Get the most recent previous event
                                                previous_event = previous_events.iloc[-1]
                                                current_event = obj_type_events.loc[idx]

                                                # Calculate gap between previous activity and current activity
                                                gap_hours = (current_event['timestamp'] -
                                                             previous_event['timestamp']).total_seconds() / 3600

                                                if gap_hours > max_gap:
                                                    # Create timing violation if we haven't already
                                                    if timing_violation is None:
                                                        timing_violation = {
                                                            'case_id': case_id,
                                                            'object_type': obj_type,
                                                            'actual_duration': case_duration,
                                                            'threshold': timing_rules['total_duration'],
                                                            'events': obj_type_events['event_id'].tolist(),
                                                            'activity_gaps': []
                                                        }

                                                    # Record the gap violation
                                                    timing_violation['activity_gaps'].append({
                                                        'activity': activity,
                                                        'previous_activity': previous_event['activity'],
                                                        'gap_hours': gap_hours,
                                                        'threshold_hours': max_gap,
                                                        'current_event_id': current_event['event_id'],
                                                        'previous_event_id': previous_event['event_id'],
                                                        'current_timestamp': current_event['timestamp'],
                                                        'previous_timestamp': previous_event['timestamp']
                                                    })

                                    # If we found any violations, add to the failures list
                                    if timing_violation is not None:
                                        failures['timing_violations'].append(timing_violation)

                            failures['timing_violations'].append(timing_violation)

                # Track resource switches
                resource_sequence = case_data['resource'].tolist()
                resource_changes = [(i, resource_sequence[i], resource_sequence[i + 1])
                                    for i in range(len(resource_sequence) - 1)
                                    if resource_sequence[i] != resource_sequence[i + 1]]

                if len(resource_changes) > 0:
                    failures['resource_switches'].append({
                        'case_id': case_id,
                        'total_switches': len(resource_changes),
                        'switches': [{
                            'position': pos,
                            'from_resource': res1,
                            'to_resource': res2,
                            'event_id': case_data.iloc[pos + 1]['event_id'],
                            'activity': case_data.iloc[pos + 1]['activity'],
                            'timestamp': case_data.iloc[pos + 1]['timestamp']
                        } for pos, res1, res2 in resource_changes],
                        'events': case_data['event_id'].tolist()
                    })

                # Track rework activities
                activity_counts = case_data['activity'].value_counts()
                rework = activity_counts[activity_counts > 1]
                if not rework.empty:
                    failures['rework_activities'].append({
                        'case_id': case_id,
                        'rework_activities': {
                            activity: {
                                'count': count,
                                'events': case_data[case_data['activity'] == activity]['event_id'].tolist(),
                                'timestamps': case_data[case_data['activity'] == activity]['timestamp'].tolist(),
                                'resources': case_data[case_data['activity'] == activity]['resource'].tolist()
                            }
                            for activity, count in rework.items()
                        },
                        'events': case_data['event_id'].tolist()
                    })

            return failures

        except Exception as e:
            logger.error(f"Error in failure pattern detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _count_resource_interactions(self) -> int:
        """Count number of resource interactions in process"""
        interactions = 0
        for case_id, case_events in self.case_events.items():
            case_resources = set()
            for event_id in case_events:
                event = next((e for e in self.ocel_data['ocel:events'] if e['ocel:id'] == event_id), None)
                if event:
                    resource = event.get('ocel:attributes', {}).get('resource', 'Unknown')
                    case_resources.add(resource)
            if len(case_resources) > 1:
                interactions += len(case_resources) - 1
        return interactions

    def _count_major_violations(self, duration_data: Dict) -> int:
        """Count number of major timing violations"""
        major_violations = 0
        for metrics in duration_data.values():
            if hasattr(metrics, 'details'):
                violations = metrics.details.get('outlier_events', {}).get('timing_gap', [])
                major_violations += sum(1 for v in violations
                                        if v['details'].get('time_gap_minutes', 0) >
                                        v['details'].get('threshold_minutes', 0) * 2)
        return major_violations

    def _identify_bottlenecks(self, duration_data: Dict) -> int:
        """Identify number of process bottlenecks"""
        return sum(1 for metrics in duration_data.values()
                   if hasattr(metrics, 'is_outlier') and metrics.is_outlier)

    def _count_object_interactions(self, case_data: Dict) -> int:
        """Count cases with multiple object interactions"""
        multi_object_cases = 0
        for metrics in case_data.values():
            if hasattr(metrics, 'details'):
                events = metrics.details.get('events', {})
                if len(set(events.get('object_types', []))) > 1:
                    multi_object_cases += 1
        return multi_object_cases

    def _count_case_variants(self, case_data: Dict) -> int:
        """Count unique case variants based on activity sequences"""
        variants = set()
        for metrics in case_data.values():
            if hasattr(metrics, 'details'):
                activity_sequence = tuple(metrics.details.get('events', {}).get('activity_sequence', []))
                if activity_sequence:
                    variants.add(activity_sequence)
        return len(variants)

    def _format_ai_response(self, response_text: str) -> Dict:
        """Format AI analysis response into structured sections"""
        try:
            sections = response_text.split('\n\n')
            formatted_response = {
                'summary': sections[0] if sections else '',
                'insights': [],
                'recommendations': []
            }

            current_section = 'insights'
            for line in sections[1:]:
                line = line.strip()
                if not line:
                    continue

                if line.lower().startswith('recommend'):
                    current_section = 'recommendations'
                    continue

                formatted_response[current_section].append(line)

            return formatted_response

        except Exception as e:
            logger.error(f"Error formatting AI response: {str(e)}")
            return {
                'summary': 'Error formatting response',
                'insights': [],
                'recommendations': []
            }

    def get_explanation(self, tab_type: str, metrics: Dict) -> Dict:
        """Generate AI explanation for OCEL analysis results"""
        logger.info(f"Generating OCEL explanation for: {tab_type}")

        try:
            context = self._build_analysis_context(tab_type, metrics)

            prompt = f"""
            Analyze this Object-Centric Process Mining data for {context['title']}:

            Process Context:
            {self._format_process_context()}

            Analysis Metrics:
            {self._format_metrics(context['metrics'])}

            Specific Patterns:
            {self._format_patterns(context['patterns'])}

            Object Type Interactions:
            {self._format_object_interactions()}

            Provide detailed analysis focusing on:
            1. Object-centric interactions and their impact on process performance
            2. Process conformance, deviations, and their root causes
            3. Resource and time utilization patterns affecting multiple object types
            4. Multi-perspective performance insights across object lifecycles
            5. Specific recommendations for process improvement

            Format response as:
            1. Key findings about object interactions and process patterns
            2. Critical insights about performance and conformance
            3. Specific, actionable recommendations for process improvement
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an expert in object-centric process mining analysis, focusing on multi-object interactions and process patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return self._format_ai_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                "error": "Unable to generate explanation",
                "details": str(e)
            }

    def _build_analysis_context(self, tab_type: str, metrics: Dict) -> Dict:
        """Build comprehensive analysis context from outliers data"""
        context = {
            'title': '',
            'metrics': [],
            'patterns': []
        }

        if tab_type == 'failure':
            failures = self.outliers.get('failures', {})
            context['title'] = 'Process Failure Pattern Analysis'
            context['metrics'] = [
                f"Total sequence violations: {len(failures.get('sequence_violations', []))}",
                f"Incomplete cases: {len(failures.get('incomplete_cases', []))}",
                f"Timing violations: {len(failures.get('timing_violations', []))}",
                f"Rework activities: {len(failures.get('rework_activities', []))}",
                f"Resource switches: {len(failures.get('resource_switches', []))}"
            ]
            context['patterns'] = self._extract_failure_patterns(failures)

        elif tab_type == 'resource':
            resource_data = self.outliers.get('resource_load', {})
            context['metrics'] = []
            for resource, metrics in resource_data.items():
                if hasattr(metrics, 'details'):
                    stats = metrics.details.get('metrics', {})
                    context['metrics'].append(
                        f"{resource}: {stats.get('total_events', 0)} events, "
                        f"{len(metrics.details['events'].get('cases', []))} cases, "
                        f"{len(metrics.details['events'].get('activities', []))} activities"
                    )
            context['title'] = 'Resource Utilization Analysis'
            context['patterns'] = self._extract_resource_patterns(resource_data)

        elif tab_type == 'time':
            duration_data = self.outliers.get('duration', {})
            context['title'] = 'Temporal Pattern Analysis'
            context['metrics'] = []
            for activity, metrics in duration_data.items():
                if hasattr(metrics, 'details'):
                    violation_count = len(metrics.details.get('outlier_events', {}).get('timing_gap', []))
                    context['metrics'].append(
                        f"{activity}: {violation_count} timing violations, "
                        f"Z-score: {metrics.z_score:.2f}"
                    )
            context['patterns'] = self._extract_timing_patterns(duration_data)

        elif tab_type == 'case':
            case_data = self.outliers.get('case_complexity', {})
            context['title'] = 'Case Complexity Analysis'
            context['metrics'] = []
            outlier_cases = [case_id for case_id, metrics in case_data.items()
                             if getattr(metrics, 'is_outlier', False)]
            context['metrics'].extend([
                f"Total cases analyzed: {len(case_data)}",
                f"Complex cases identified: {len(outlier_cases)}",
                f"Average case duration: {self._calculate_avg_case_duration():.2f} hours"
            ])
            context['patterns'] = self._extract_case_patterns(case_data)

        return context

    def _format_process_context(self) -> str:
        """Format overall process context"""
        process_stats = {
            'total_events': len(self.events_df) if hasattr(self, 'events_df') else 0,
            'object_types': self._get_unique_object_types(),
            'activities': self._get_unique_activities(),
            'object_interactions': self._analyze_object_interactions()
        }

        return f"""
        Process Overview:
        - Total events: {process_stats['total_events']}
        - Object types: {', '.join(process_stats['object_types'])}
        - Activities: {', '.join(process_stats['activities'])}
        - Object interactions: {process_stats['object_interactions']}
        """

    def _format_metrics(self, metrics: List[str]) -> str:
        """Format metrics for AI analysis"""
        return '\n'.join(f"- {metric}" for metric in metrics)

    def _format_patterns(self, patterns: List[str]) -> str:
        """Format patterns for AI analysis"""
        return '\n'.join(f"- {pattern}" for pattern in patterns)

    def _extract_failure_patterns(self, failures: Dict) -> List[str]:
        """Extract key failure patterns from outlier data"""
        patterns = []

        # Analyze sequence violations
        sequence_violations = failures.get('sequence_violations', [])
        if sequence_violations:
            violation_types = defaultdict(int)
            for violation in sequence_violations:
                for activity in violation.get('wrong_order_activities', []):
                    violation_types[f"{activity['expected']} -> {activity['actual']}"] += 1

            if violation_types:
                most_common = max(violation_types.items(), key=lambda x: x[1])
                patterns.append(f"Most common sequence violation: {most_common[0]} ({most_common[1]} occurrences)")

        # Analyze timing violations
        timing_violations = failures.get('timing_violations', [])
        if timing_violations:
            critical_gaps = [
                v for v in timing_violations
                if any(gap['gap_hours'] > gap['threshold_hours'] * 2
                       for gap in v.get('activity_gaps', []))
            ]
            if critical_gaps:
                patterns.append(f"Critical timing violations: {len(critical_gaps)} cases with severe delays")

        # Analyze resource switches
        resource_switches = failures.get('resource_switches', [])
        if resource_switches:
            high_switch_cases = [r for r in resource_switches if r['total_switches'] > 3]
            if high_switch_cases:
                patterns.append(f"Resource handover issues: {len(high_switch_cases)} cases with excessive switches")

        # Analyze rework
        rework = failures.get('rework_activities', [])
        if rework:
            rework_activities = defaultdict(int)
            for case in rework:
                for activity, details in case.get('rework_activities', {}).items():
                    rework_activities[activity] += details.get('count', 0)

            if rework_activities:
                most_rework = max(rework_activities.items(), key=lambda x: x[1])
                patterns.append(f"Most reworked activity: {most_rework[0]} ({most_rework[1]} times)")

        return patterns

    def _extract_resource_patterns(self, resource_data: Dict) -> List[str]:
        """Extract key resource patterns from outlier data"""
        patterns = []

        # Analyze workload distribution
        workloads = []
        for resource, metrics in resource_data.items():
            if hasattr(metrics, 'details'):
                stats = metrics.details.get('metrics', {})
                workloads.append((resource, stats.get('total_events', 0)))

        if workloads:
            workloads.sort(key=lambda x: x[1], reverse=True)
            patterns.append(f"Highest workload: {workloads[0][0]} ({workloads[0][1]} events)")

            # Calculate workload imbalance
            avg_workload = sum(w[1] for w in workloads) / len(workloads)
            imbalanced = [w for w in workloads if abs(w[1] - avg_workload) > avg_workload * 0.5]
            if imbalanced:
                patterns.append(f"Workload imbalance detected in {len(imbalanced)} resources")

        # Analyze resource specialization
        specialists = []
        for resource, metrics in resource_data.items():
            if hasattr(metrics, 'details'):
                activities = metrics.details['events'].get('activities', [])
                if len(activities) <= 2:
                    specialists.append(resource)

        if specialists:
            patterns.append(f"Specialized resources: {len(specialists)} resources with focused activities")

        return patterns

    def _extract_timing_patterns(self, duration_data: Dict) -> List[str]:
        """Extract key timing patterns from outlier data"""
        patterns = []

        # Analyze systematic delays
        systematic_delays = defaultdict(list)
        for activity, metrics in duration_data.items():
            if hasattr(metrics, 'details'):
                violations = metrics.details.get('outlier_events', {}).get('timing_gap', [])
                if violations:
                    for violation in violations:
                        if violation['details'].get('time_gap_minutes', 0) > violation['details'].get(
                                'threshold_minutes', 0):
                            systematic_delays[activity].append(violation)

        if systematic_delays:
            most_delayed = max(systematic_delays.items(), key=lambda x: len(x[1]))
            patterns.append(f"Most delayed activity: {most_delayed[0]} ({len(most_delayed[1])} violations)")

        # Analyze bottlenecks
        bottlenecks = []
        for activity, metrics in duration_data.items():
            if hasattr(metrics, 'details') and metrics.z_score > 2:
                bottlenecks.append(activity)

        if bottlenecks:
            patterns.append(f"Process bottlenecks identified in: {', '.join(bottlenecks)}")

        return patterns

    def _extract_case_patterns(self, case_data: Dict) -> List[str]:
        """Extract key case patterns from outlier data"""
        patterns = []

        # Analyze case complexity
        case_complexities = []
        for case_id, metrics in case_data.items():
            if hasattr(metrics, 'details'):
                total_events = metrics.details['metrics'].get('total_events', 0)
                case_complexities.append((case_id, total_events))

        if case_complexities:
            avg_complexity = sum(c[1] for c in case_complexities) / len(case_complexities)
            complex_cases = [c for c in case_complexities if c[1] > avg_complexity * 1.5]
            if complex_cases:
                patterns.append(f"Complex cases: {len(complex_cases)} cases with high complexity")

        # Analyze object interactions
        case_objects = defaultdict(set)
        for case_id, metrics in case_data.items():
            if hasattr(metrics, 'details'):
                case_objects[case_id].update(metrics.details['events'].get('object_types', []))

        multi_object_cases = [(case_id, len(objects)) for case_id, objects in case_objects.items() if len(objects) > 1]
        if multi_object_cases:
            patterns.append(f"Multi-object interactions: {len(multi_object_cases)} cases with multiple object types")

        return patterns

    def _get_unique_object_types(self) -> Set[str]:
        """Get unique object types from OCEL data"""
        return set(obj['type'] for event in self.ocel_data['ocel:events']
                   for obj in event.get('ocel:objects', []))

    def _get_unique_activities(self) -> Set[str]:
        """Get unique activities from OCEL data"""
        return set(event['ocel:activity'] for event in self.ocel_data['ocel:events'])

    def _analyze_object_interactions(self) -> str:
        """Analyze object type interactions"""
        interactions = defaultdict(set)
        for event in self.ocel_data['ocel:events']:
            objects = event.get('ocel:objects', [])
            if len(objects) > 1:
                object_types = {obj['type'] for obj in objects}
                if len(object_types) > 1:
                    key = tuple(sorted(object_types))
                    interactions[key].add(event['ocel:id'])

        if interactions:
            most_common = max(interactions.items(), key=lambda x: len(x[1]))
            return f"Most common interaction: {' - '.join(most_common[0])} ({len(most_common[1])} instances)"
        return "No significant object interactions found"

    def _format_object_interactions(self) -> str:
        """Format object interactions for AI analysis"""
        object_types = self._get_unique_object_types()
        interactions = defaultdict(int)

        for event in self.ocel_data['ocel:events']:
            objects = event.get('ocel:objects', [])
            if len(objects) > 1:
                object_types = {obj['type'] for obj in objects}
                if len(object_types) > 1:
                    key = tuple(sorted(object_types))
                    interactions[key] += 1

        return "\n".join(
            f"- {' - '.join(types)}: {count} interactions"
            for types, count in sorted(interactions.items(), key=lambda x: x[1], reverse=True)
        )

    def _calculate_avg_case_duration(self) -> float:
        """Calculate average case duration in hours"""
        case_durations = []
        for case_events in self.case_events.values():
            if case_events:
                timestamps = [
                    pd.to_datetime(event['ocel:timestamp'])
                    for event in self.ocel_data['ocel:events']
                    if event['ocel:id'] in case_events
                ]
                if timestamps:
                    duration = (max(timestamps) - min(timestamps)).total_seconds() / 3600
                    case_durations.append(duration)

        return sum(case_durations) / len(case_durations) if case_durations else 0.0

    def get_explanation(self, tab_type: str, metrics: Dict) -> Dict:
        """Generate AI explanation for OCEL analysis results"""
        logger.info(f"Generating OCEL explanation for: {tab_type}")

        try:
            context = self._build_analysis_context(tab_type, metrics)

            prompt = f"""
            Analyze this Object-Centric Process Mining data for {context['title']}:

            Process Context:
            {self._format_process_context()}

            Analysis Metrics:
            {self._format_metrics(context['metrics'])}

            Specific Patterns:
            {self._format_patterns(context['patterns'])}

            Object Type Interactions:
            {self._format_object_interactions()}

            Provide detailed analysis focusing on:
            1. Object-centric interactions and their impact on process performance
            2. Process conformance, deviations, and their root causes
            3. Resource and time utilization patterns affecting multiple object types
            4. Multi-perspective performance insights across object lifecycles
            5. Specific recommendations for process improvement

            Format response as:
            1. Key findings about object interactions and process patterns
            2. Critical insights about performance and conformance
            3. Specific, actionable recommendations for process improvement
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an expert in object-centric process mining analysis, focusing on multi-object interactions and process patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return self._format_ai_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                "error": "Unable to generate explanation",
                "details": str(e)
            }

    def _display_resource_details(self, resource: str, details: Dict):
        """Helper method to display resource details"""
        try:
            st.write("### Resource Details")

            # Show metrics
            cols = st.columns(3)
            cols[0].metric("Total Events", details['metrics'].get('total_events', 0))
            cols[1].metric("Cases", len(details['events'].get('cases', [])))
            cols[2].metric("Activities", len(details['events'].get('activities', [])))

            # Show activity distribution
            events_df = self._get_events_dataframe(details['events'].get('all_events', []))
            if not events_df.empty:
                st.write("### Activity Distribution")
                activity_counts = events_df['activity'].value_counts()
                activity_fig = px.bar(
                    x=activity_counts.index,
                    y=activity_counts.values,
                    title=f'Activities Performed by {resource}'
                )
                st.plotly_chart(activity_fig)

        except Exception as e:
            logger.error(f"Error displaying resource details: {str(e)}")
            st.error("Error displaying resource details")

    def _display_time_analysis(self, duration_data: Dict):
        """Helper method to display time analysis"""
        try:
            # Create duration outlier visualization
            duration_info = []
            for activity, metrics in duration_data.items():
                if hasattr(metrics, 'details'):
                    duration_info.append({
                        'Activity': activity,
                        'Z-Score': metrics.z_score,
                        'Is Outlier': metrics.is_outlier,
                        'Total Events': metrics.details.get('total_events', 0),
                        'Violation Count': len(metrics.details.get('outlier_events', {}).get('timing_gap', []))
                    })

            if duration_info:
                df = pd.DataFrame(duration_info)
                fig = px.scatter(
                    df,
                    x='Activity',
                    y='Z-Score',
                    size='Total Events',
                    color='Is Outlier',
                    title='Activity Duration Distribution'
                )
                st.plotly_chart(fig)

                # Display timing gaps if available
                self._display_timing_gaps(duration_data)

        except Exception as e:
            logger.error(f"Error in time analysis display: {str(e)}")
            raise

    def _display_case_analysis(self, case_data: Dict):
        """Helper method to display case analysis"""
        try:
            # Create case visualization
            case_info = []
            case_details = {}

            for case_id, metrics in case_data.items():
                if hasattr(metrics, 'details'):
                    case_info.append({
                        'Case': case_id,
                        'Z-Score': metrics.z_score,
                        'Is Outlier': metrics.is_outlier,
                        'Total Events': metrics.details['metrics'].get('total_events', 0),
                        'Activity Count': metrics.details['metrics'].get('activity_variety', 0)
                    })
                    case_details[case_id] = metrics.details

            if case_info:
                df = pd.DataFrame(case_info)
                fig = px.scatter(
                    df,
                    x='Case',
                    y='Z-Score',
                    size='Total Events',
                    color='Is Outlier',
                    title='Case Complexity Distribution'
                )

                fig.update_traces(
                    hovertemplate=(
                            "<b>%{x}</b><br>" +
                            "Z-Score: %{y:.2f}<br>" +
                            "Total Events: %{marker.size}<br>" +
                            "Activities: %{customdata[0]}<br>" +
                            "<extra></extra>"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add case details selection
                if case_details:
                    selected_case = st.selectbox(
                        "Select case for details",
                        options=list(case_details.keys())
                    )
                    if selected_case:
                        self._display_case_details(selected_case, case_details[selected_case])

        except Exception as e:
            logger.error(f"Error in case analysis display: {str(e)}")
            raise

    def _display_case_details(self, case_id: str, details: Dict):
        """Helper method to display detailed case information"""
        try:
            st.write("### Case Details")

            # Show if case is an outlier
            if details.get('outlier_patterns', {}).get('high_complexity'):
                st.warning("⚠️ This case has high complexity")
            if details.get('outlier_patterns', {}).get('long_duration'):
                st.warning("⚠️ This case has unusual duration")

            # Display metrics
            cols = st.columns(3)
            cols[0].metric(
                "Total Events",
                details['metrics'].get('total_events', 0)
            )
            cols[1].metric(
                "Activities",
                details['metrics'].get('activity_variety', 0)
            )
            cols[2].metric(
                "Duration (hrs)",
                f"{details['temporal'].get('duration_hours', 0):.1f}"
            )

            # Display timeline
            events_df = self._get_events_dataframe(details['events'].get('all_events', []))
            if not events_df.empty:
                events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

                # Create timeline visualization
                timeline_fig = px.scatter(
                    events_df,
                    x='timestamp',
                    y='activity',
                    color='resource',
                    title=f'Case Timeline - {case_id}',
                    labels={
                        'timestamp': 'Time',
                        'activity': 'Activity',
                        'resource': 'Resource'
                    }
                )

                timeline_fig.update_layout(
                    height=400,
                    showlegend=True,
                    hovermode='closest'
                )
                st.plotly_chart(timeline_fig)

                # Display event sequence
                st.write("### Event Sequence")
                sequence_df = events_df[['timestamp', 'activity', 'resource']].copy()
                sequence_df['duration_from_start'] = (
                                                             sequence_df['timestamp'] - sequence_df['timestamp'].min()
                                                     ).dt.total_seconds() / 3600

                # Add styling to highlight potential issues
                styled_df = sequence_df.style.background_gradient(
                    subset=['duration_from_start'],
                    cmap='YlOrRd'
                )
                st.dataframe(styled_df)

            # Display object interactions if available
            if 'object_types' in details:
                st.write("### Object Interactions")
                object_df = pd.DataFrame({
                    'Object Type': details['object_types'],
                    'Interaction Count': [1] * len(details['object_types'])
                }).groupby('Object Type').sum()

                st.bar_chart(object_df)

        except Exception as e:
            logger.error(f"Error displaying case details: {str(e)}")
            st.error(f"Error showing case details: {str(e)}")

    def _display_timing_gaps(self, duration_data: Dict):
        """Helper method to display timing gap analysis using pandas styling"""
        try:
            timing_gaps = []
            for activity, metrics in duration_data.items():
                if hasattr(metrics, 'details'):
                    for violation in metrics.details.get('outlier_events', {}).get('timing_gap', []):
                        prev_event = self._get_event_details(violation['details'].get('previous_event', ''))

                        timing_gaps.append({
                            'Case ID': violation.get('case_id', 'Unknown'),
                            'Current Activity': activity,
                            'Previous Activity': prev_event.get('Activity', 'Unknown'),
                            'Gap (Minutes)': round(violation['details'].get('time_gap_minutes', 0), 2),
                            'Threshold': violation['details'].get('threshold_minutes', 0)
                        })

            if timing_gaps:
                st.write("### Timing Gap Analysis")
                df = pd.DataFrame(timing_gaps)

                def highlight_gaps(row):
                    return ['background-color: #ffcdd2' if row['Gap (Minutes)'] > row['Threshold']
                            else 'background-color: #ffffff' for _ in row]

                styled_df = df.style.apply(highlight_gaps, axis=1)
                st.dataframe(styled_df)

        except Exception as e:
            logger.error(f"Error displaying timing gaps: {str(e)}")
            st.error("Error displaying timing gap analysis")

    def display_enhanced_analysis(self):
        """Display enhanced analysis with comprehensive outlier tracing and error handling"""
        try:
            # Validate that outliers dictionary exists and has required keys
            if not hasattr(self, 'outliers'):
                st.error("No outlier analysis data available. Please ensure data is processed first.")
                return

            # Create tabs for different analyses
            tabs = st.tabs(["Failure Patterns", "Resource Outlier", "Time Outlier", "Case Outlier"])

            # Failure Patterns Tab
            with tabs[0]:
                logger.debug("Processing Failure Patterns tab")
                with st.expander("Failure Pattern Detection Logic"):
                    st.markdown("""
                    # Understanding Failure Pattern Detection

                    ## Overview
                    The failure pattern detection system analyzes object-centric event logs to identify various types of process deviations and anomalies. It monitors six key categories of failures:
                    
                    1. Sequence Violations
                    2. Incomplete Cases
                    3. Long Running Cases
                    4. Resource Switches  
                    5. Rework Activities
                    6. Timing Violations
                    
                    ## Detection Process
                    
                    ### Data Preparation
                    The system first organizes event data into a structured format containing:
                    - Event IDs
                    - Timestamps
                    - Activities
                    - Resources
                    - Object Types
                    - Object IDs
                    
                    ### Failure Categories in Detail
                    
                    #### 1. Sequence Violations
                    ```python
                    actual_sequence = case_data[...]['activity'].tolist()
                    if actual_sequence != expected_sequence:
                        # Track violation details
                    ```
                    - Compares actual activity sequence against expected flow
                    - Records:
                      - Missing activities
                      - Wrong order activities
                      - First violation point
                      - Affected objects
                    
                    #### 2. Incomplete Cases
                    ```python
                    if violation['missing_activities']:
                        # Track incomplete case details
                    ```
                    - Identifies cases missing required activities
                    - Tracks:
                      - Missing activities list
                      - Completed activities
                      - Last known event
                      - Case context
                    
                    #### 3. Timing Violations
                    ```python
                    if case_duration > timing_rules['total_duration']:
                        # Track timing violation details
                    ```
                    Monitors two types of timing issues:
                    - Overall case duration exceeding thresholds
                    - Activity-specific gaps between events
                    - Records detailed gap analysis including:
                      - Previous activity
                      - Current activity
                      - Gap duration
                      - Threshold exceeded
                    
                    #### 4. Resource Switches
                    ```python
                    resource_changes = [(i, sequence[i], sequence[i+1])
                                       for i in range(len(sequence)-1)
                                       if sequence[i] != sequence[i+1]]
                    ```
                    - Detects handovers between different resources
                    - Tracks:
                      - Switch points
                      - From/To resources
                      - Associated activities
                      - Timestamps
                    
                    #### 5. Rework Activities
                    ```python
                    activity_counts = case_data['activity'].value_counts()
                    rework = activity_counts[activity_counts > 1]
                    ```
                    - Identifies repeated activities
                    - Records:
                      - Activity frequency
                      - Event sequences
                      - Resources involved
                      - Timestamps
                    
                    ## Visualization Components
                    
                    The failure pattern analysis is displayed in four key components:
                    
                    1. **Pattern Distribution Bar Chart**
                       - Shows count of each failure type
                       - Color-coded by severity
                       - Interactive tooltips with details
                    
                    2. **Detailed Pattern Analysis**
                       - Expandable sections for each pattern type
                       - Tabular view of specific failures
                       - Sorting and filtering capabilities
                    
                    3. **Metrics Summary**
                       - Key statistics about detected patterns
                       - Trend indicators
                       - Severity distribution
                    
                    4. **AI-Generated Insights**
                       - Pattern interpretation
                       - Key findings
                       - Improvement recommendations
                    
                    ## Usage in Process Analysis
                    
                    This failure pattern detection helps organizations:
                    1. Identify process bottlenecks
                    2. Monitor compliance violations
                    3. Optimize resource allocation
                    4. Improve process efficiency
                    5. Ensure quality control
                    
                    ## Implementation Details
                    
                    The code implements several advanced features:
                    - Full event traceability
                    - Multi-perspective analysis
                    - Object-centric correlation
                    - Temporal pattern detection
                    - Resource interaction analysis
                    
                    ## Data Structure Example
                    
                    A typical failure pattern record looks like:
                    ```json
                    {
                      "case_id": "Case_1",
                      "object_type": "Trade",
                      "actual_sequence": ["A", "B", "D"],
                      "expected_sequence": ["A", "B", "C", "D"],
                      "missing_activities": ["C"],
                      "events": ["evt_1", "evt_2", "evt_4"],
                      "first_violation": {
                        "event_id": "evt_2",
                        "timestamp": "2024-01-01T10:00:00",
                        "resource": "Trader_A"
                      }
                    }
                    ```
                    
                    ## Performance Considerations
                    
                    The detection system is optimized for:
                    - Efficient data processing
                    - Minimal memory footprint
                    - Real-time analysis capability
                    - Scalable pattern detection
                    
                    ## Error Handling
                    
                    The system includes comprehensive error handling:
                    - Data validation
                    - Exception logging
                    - Graceful degradation
                    - Recovery mechanisms
                    
                    ## Integration Points
                    
                    The failure pattern detection integrates with:
                    - Process mining analytics
                    - Conformance checking
                    - Performance analysis
                    - Resource optimization
                    
                    
                    """)

                # Validate failures data exists
                failures_data = self.outliers.get('failures', {})
                if not failures_data:
                    st.warning("No failure pattern data available")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        failure_counts = {
                            'Sequence Violations': len(failures_data.get('sequence_violations', [])),
                            'Incomplete Cases': len(failures_data.get('incomplete_cases', [])),
                            'Long Running': len(failures_data.get('long_running', [])),
                            'Resource Switches': len(failures_data.get('resource_switches', [])),
                            'Rework Activities': len(failures_data.get('rework_activities', []))
                        }

                        if any(failure_counts.values()):
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(failure_counts.keys()),
                                    y=list(failure_counts.values()),
                                    text=list(failure_counts.values()),
                                    textposition='auto',
                                )
                            ])

                            fig.update_layout(
                                title='Process Failure Patterns Distribution',
                                xaxis_title='Failure Pattern Type',
                                yaxis_title='Count',
                                showlegend=False
                            )
                            st.plotly_chart(fig)

                            # Show failure details in expandable sections
                            for pattern, failures in failures_data.items():
                                if failures:
                                    with st.expander(f"{pattern.replace('_', ' ').title()} ({len(failures)})"):
                                        try:
                                            failure_df = pd.DataFrame(failures)
                                            st.dataframe(failure_df)
                                        except Exception as e:
                                            logger.error(f"Error creating failure DataFrame: {str(e)}")
                                            st.error(f"Error displaying failure details: {str(e)}")

                    with col2:
                        failure_metrics = {
                            'sequence_violations': len(failures_data.get('sequence_violations', [])),
                            'incomplete_cases': len(failures_data.get('incomplete_cases', [])),
                            'long_running': len(failures_data.get('long_running', []))
                        }

                        st.markdown("### Understanding Failure Patterns")
                        try:
                            explanation = self.get_explanation('failure', failure_metrics)
                            st.markdown(explanation.get('summary', 'No summary available'))

                            if explanation.get('insights'):
                                st.markdown("#### Key Insights")
                                for insight in explanation['insights']:
                                    st.markdown(f"• {insight}")

                            if explanation.get('recommendations'):
                                st.markdown("#### Recommendations")
                                for rec in explanation['recommendations']:
                                    st.markdown(f"• {rec}")
                        except Exception as e:
                            logger.error(f"Error getting explanation: {str(e)}")
                            st.error("Unable to generate insights at this time")

            # Resource Analysis Tab
            with tabs[1]:
                logger.debug("Processing Resource Analysis tab")
                with st.expander("📊 Resource Complexity Detection Logic"):
                    st.markdown("""
                    # Understanding Resource Complexity Detection

                    ## Overview
                    The visualization shows resource workload distribution in object-centric process mining (OCPM), highlighting potential outliers in how resources interact with different process objects and activities.
                    
                    ## Metrics Calculation
                    
                    ### Base Metrics
                    - **Total Events**: Raw count of events handled by each resource
                    - **Unique Cases**: Number of distinct cases a resource works on
                    - **Activity Variety**: Number of different activities performed
                    - **Object Variety**: Unique object types handled
                    
                    ### Z-Score Analysis
                    The code calculates normalized z-scores for each metric using:
                    ```
                    z = (value - mean) / standard_deviation
                    ```
                    
                    ### Composite Score
                    A composite z-score is generated by averaging absolute z-scores across all metrics to identify overall outliers.
                    
                    ## Visualization Components
                    
                    ### Scatter Plot Elements
                    - **X-axis**: Individual resources
                    - **Y-axis**: Composite z-score
                    - **Bubble Size**: Total event count
                    - **Color**: Outlier status (z-score > 3)
                    
                    ### Interactive Features
                    - Hover shows detailed metrics
                    - Selection enables detailed resource analysis
                    
                    ## Outlier Detection
                    
                    ### Workload Patterns
                    - **High Workload**: Events > mean + 2*std
                    - **High Variety**: Activity count > mean + 2*std
                    
                    ### Resource Classification
                    Resources are flagged as outliers when:
                    - Composite z-score > 3
                    - Showing unusual patterns in:
                      - Event volume
                      - Case variety
                      - Activity diversity
                      - Object type interactions
                    
                    ## Interpretation Guide
                    
                    ### Normal Resource Profile
                    - Balanced workload distribution
                    - Expected activity variety
                    - Typical object type interactions
                    
                    ### Outlier Indicators
                    - Unusually high/low event counts
                    - Excessive activity variety
                    - Abnormal object type interactions
                    - Extreme composite z-scores
                    
                    ### Business Impact
                    - Resource overloading
                    - Specialization vs. generalization
                    - Process bottlenecks
                    - Workload imbalances
                    
                    ## Technical Implementation Notes
                    The implementation uses vectorized operations in pandas for efficiency:
                    - Grouped aggregations
                    - Vectorized z-score calculations
                    - Optimized outlier detection
                    
                    """)

                    # Validate resource data exists
                resource_data = self.outliers.get('resource_load', {})
                if not resource_data:
                    st.warning("No resource analysis data available")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Create resource workload visualization
                        workload_data = []
                        resource_details = {}

                        for resource, metrics in resource_data.items():
                            if isinstance(metrics, (dict, object)):  # Validate metrics object
                                try:
                                    workload_data.append({
                                        'Resource': resource,
                                        'Z-Score': getattr(metrics, 'z_score', 0),
                                        'Is Outlier': getattr(metrics, 'is_outlier', False),
                                        'Total Events': metrics.details['metrics'].get('total_events', 0) if hasattr(
                                            metrics, 'details') else 0,
                                        'Cases': len(metrics.details['events'].get('cases', [])) if hasattr(metrics,
                                                                                                            'details') else 0,
                                        'Activities': len(metrics.details['events'].get('activities', [])) if hasattr(
                                            metrics, 'details') else 0
                                    })

                                    # Store details for traceability
                                    if hasattr(metrics, 'details'):
                                        resource_details[resource] = {
                                            'metrics': metrics.details.get('metrics', {}),
                                            'events': metrics.details.get('events', {}),
                                            'patterns': metrics.details.get('outlier_patterns', {})
                                        }
                                except Exception as e:
                                    logger.error(f"Error processing resource metrics: {str(e)}")

                        if workload_data:
                            # Create resource plot
                            try:
                                fig = px.scatter(
                                    pd.DataFrame(workload_data),
                                    x='Resource',
                                    y='Z-Score',
                                    size='Total Events',
                                    color='Is Outlier',
                                    title='Resource Workload Distribution',
                                    custom_data=['Cases', 'Activities']
                                )

                                fig.update_traces(
                                    hovertemplate=(
                                            "<b>%{x}</b><br>" +
                                            "Z-Score: %{y:.2f}<br>" +
                                            "Total Events: %{marker.size}<br>" +
                                            "Cases: %{customdata[0]}<br>" +
                                            "Activities: %{customdata[1]}<br>" +
                                            "<extra></extra>"
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Show resource details
                                if resource_details:
                                    selected_resource = st.selectbox(
                                        "Select resource for details",
                                        options=list(resource_details.keys())
                                    )
                                    if selected_resource:
                                        self._display_resource_details(selected_resource,
                                                                       resource_details[selected_resource])
                            except Exception as e:
                                logger.error(f"Error creating resource visualization: {str(e)}")
                                st.error("Error displaying resource visualization")

                    with col2:
                        workload_metrics = {
                            'market_maker_b': len(self.resource_events.get('Market Maker B', [])),
                            'client_desk_d': len(self.resource_events.get('Client Desk D', [])),
                            'options_desk_a': len(self.resource_events.get('Options Desk A', []))
                        }
                        st.markdown("### Understanding Resource Distribution")
                        try:
                            explanation = self.get_explanation('resource', workload_metrics)
                            st.markdown(explanation.get('summary', 'No summary available'))

                            if explanation.get('insights'):
                                st.markdown("#### Key Insights")
                                for insight in explanation['insights']:
                                    st.markdown(f"• {insight}")

                            if explanation.get('recommendations'):
                                st.markdown("#### Recommendations")
                                for rec in explanation['recommendations']:
                                    st.markdown(f"• {rec}")
                        except Exception as e:
                            logger.error(f"Error getting resource explanation: {str(e)}")
                            st.error("Unable to generate resource insights")

            # Time Analysis Tab
            with tabs[2]:
                logger.debug("Processing Time Analysis tab")
                # Under the Time Analysis Tab, after your visualizations
                with st.expander("🔍 Time Outlier Detection Logic"):
                    st.markdown("""
                    # Object-Centric Process Mining: Time Outlier Detection

                    ## Overview
                    The time outlier detection analyzes process execution durations and timing patterns across different object types and activities. The visualization appears in the "Time Outlier" tab and consists of two key components:
                    
                    1. Duration Distribution Plot
                    2. Timing Gap Analysis Table
                    
                    ## Detection Process
                    
                    ### 1. Duration Data Collection
                    The system:
                    - Creates a DataFrame of all events with timestamps, activities, and object relationships
                    - Groups events by activity to analyze timing patterns
                    - Tracks multiple object types per event to handle object-centric complexity
                    
                    ### 2. Threshold Validation
                    For each activity-object combination:
                    - Retrieves timing rules from OCPM validator
                    - Validates against:
                      - Activity-specific thresholds
                      - Default gap thresholds per object type
                      - Overall process duration limits
                    
                    ### 3. Outlier Detection Logic
                    
                    #### Duration Outliers
                    ```python
                    gap_hours = (event['timestamp'] - last_event['timestamp']).total_seconds() / 3600
                    if gap_hours > activity_threshold:
                        outlier_events['timing_gap'].append({...})
                    ```
                    
                    System flags outliers when:
                    - Time gap between activities exceeds threshold
                    - Events occur out of expected sequence
                    - Activities take longer than typical duration
                    
                    #### Z-Score Calculation
                    ```python
                    violation_score = sum(metrics.values()) / (metrics['total_events'] * 3)
                    z_score = float(violation_score * 10)
                    is_outlier = violation_score > 0.3
                    ```
                    
                    ### 4. Visualization Components
                    
                    #### Duration Distribution Plot
                    - X-axis: Activities
                    - Y-axis: Z-Score
                    - Size: Number of events
                    - Color: Outlier status (True/False)
                    
                    #### Timing Gap Table
                    Shows detailed timing violations:
                    - Case ID
                    - Current/Previous Activities
                    - Gap Duration
                    - Threshold Values
                    - Color coding:
                      - Red: Exceeds threshold
                      - Green: Within threshold
                    
                    ## Key Metrics Tracked
                    
                    1. **Timing Violations**
                       - Gap between activities
                       - Sequence position violations
                       - Resource handover delays
                    
                    2. **Activity Statistics**
                       - Average duration per activity
                       - Resource distribution
                       - Object type distribution
                    
                    3. **Outlier Metrics**
                       - Z-score per activity
                       - Violation rate
                       - Total events and violations
                    
                    ## Example Interpretation
                    
                    If an activity shows:
                    - Z-score > 3: Significant outlier
                    - Multiple timing gaps: Process bottleneck
                    - High violation rate: Potential process issue
                    
                    ## Understanding the Visualization
                    
                    1. **Scatter Plot Reading**
                       - Each point represents an activity
                       - Size indicates event frequency
                       - Higher Z-scores suggest more severe timing issues
                       - Color differentiates outliers from normal activities
                    
                    2. **Gap Analysis Table Reading**
                       - Red rows indicate critical timing violations
                       - Compare actual gaps against thresholds
                       - Look for patterns in specific cases or activities
                    
                    ## Technical Implementation Notes
                    
                    The detection uses a multi-level approach:
                    1. Event-level timing analysis
                    2. Activity-level pattern detection
                    3. Object-centric relationship validation
                    4. Cross-object timing correlation
                    
                    This ensures comprehensive coverage of timing patterns while maintaining object-centric process mining principles.
                    """)

                # Validate duration data exists
                duration_data = self.outliers.get('duration', {})
                if not duration_data:
                    st.warning("No time analysis data available")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        try:
                            self._display_time_analysis(duration_data)
                        except Exception as e:
                            logger.error(f"Error displaying time analysis: {str(e)}")
                            st.error("Error displaying time analysis visualizations")

                    with col2:
                        timing_metrics = {
                            'avg_duration': f"{np.mean([len(events) for events in self.case_events.values()]):.2f}",
                            'outlier_count': len(
                                [m for m in duration_data.values() if getattr(m, 'is_outlier', False)]),
                            'typical_duration': "2-4 hours"
                        }
                        st.markdown("### Understanding Time Patterns")
                        try:
                            explanation = self.get_explanation('time', timing_metrics)
                            st.write(explanation.get('summary', 'No summary available'))

                            if explanation.get('insights'):
                                st.markdown("#### Key Insights")
                                for insight in explanation['insights']:
                                    st.markdown(f"• {insight}")

                            if explanation.get('recommendations'):
                                st.markdown("#### Recommendations")
                                for rec in explanation['recommendations']:
                                    st.markdown(f"• {rec}")
                        except Exception as e:
                            logger.error(f"Error getting time explanation: {str(e)}")
                            st.error("Unable to generate time insights")

            # Case Analysis Tab
            with tabs[3]:
                logger.debug("Processing Case Analysis tab")

                with st.expander("Case Outlier Detection Logic", expanded=False):
                    st.markdown("""
                    ## Case Outlier Detection in Object-Centric Process Mining

                    ### 1. Data Structure & Analysis
                    The code analyzes case outliers using these key metrics:
                    - **Total Events**: Number of events per case
                    - **Activity Variety**: Unique activities in each case
                    - **Resource Variety**: Different resources involved
                    - **Object Type Variety**: Different object types per case
                    - **Case Duration**: Total duration in hours

                    ### 2. Detection Method
                    Cases are flagged as outliers based on:
                    - Composite z-score calculation across all metrics
                    - Threshold: z-score > 3 indicates outlier
                    - Multi-dimensional analysis including:
                        - Event frequency patterns
                        - Activity sequence variations
                        - Resource utilization patterns
                        - Object type interactions

                    ### 3. Visualization Components

                    #### Main Scatter Plot
                    - X-axis: Case IDs
                    - Y-axis: Z-scores
                    - Point Size: Total events in case
                    - Color: Outlier status (True/False)
                    - Hover data: Detailed case metrics

                    #### Case Details View
                    When selecting a specific case:
                    - Event sequence timeline
                    - Resource distribution
                    - Object type interactions
                    - Duration metrics

                    ### 4. Implementation Details
                    ```python
                    # Key method calls in order:
                    1. _detect_case_outliers()
                    2. _build_trace_index()
                    3. _display_case_analysis()
                    4. _display_case_details()
                    ```

                    ### 5. Outlier Classification
                    Cases are marked as outliers if they exhibit:
                    - Unusually high/low number of events
                    - Unexpected activity patterns
                    - Abnormal resource usage
                    - Complex object interactions
                    - Extreme duration values

                    ### 6. AI Enhancement
                    The analysis includes AI-powered insights:
                    - Pattern identification
                    - Root cause analysis
                    - Improvement recommendations
                    """)

                    st.info("""
                    **How to Use the Visualization:**
                    1. Examine the scatter plot for overall outlier distribution
                    2. Click on specific cases to view detailed analysis
                    3. Review AI insights for process understanding
                    4. Check object interactions for complexity analysis
                    """)

                # Validate case complexity data exists
                case_data = self.outliers.get('case_complexity', {})
                if not case_data:
                    st.warning("No case analysis data available")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        try:
                            self._display_case_analysis(case_data)
                        except Exception as e:
                            logger.error(f"Error displaying case analysis: {str(e)}")
                            st.error("Error displaying case analysis visualizations")

                    with col2:
                        case_metrics = {
                            'complex_cases': len([c for c in case_data.values()
                                                  if c.details['metrics'].get('total_events', 0) > 10]),
                            'timestamp_cases': len([c for c in case_data.values()
                                                    if c.details['temporal'].get('duration_hours', 0) > 24]),
                            'object_cases': len(case_data)
                        }
                        st.markdown("### Understanding Case Complexity")
                        try:
                            explanation = self.get_explanation('case', case_metrics)
                            st.markdown(explanation.get('summary', 'No summary available'))

                            if explanation.get('insights'):
                                st.markdown("#### Key Insights")
                                for insight in explanation['insights']:
                                    st.markdown(f"• {insight}")

                            if explanation.get('recommendations'):
                                st.markdown("#### Recommendations")
                                for rec in explanation['recommendations']:
                                    st.markdown(f"• {rec}")
                        except Exception as e:
                            logger.error(f"Error getting case explanation: {str(e)}")
                            st.error("Unable to generate case insights")

        except Exception as e:
            logger.error(f"Error in display_enhanced_analysis: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("Error analyzing process patterns. Please check logs for details.")
            if st.checkbox("Show detailed error"):
                st.code(traceback.format_exc())

    def _get_events_dataframe(self, event_ids: List[str]) -> pd.DataFrame:
        """Helper method to create DataFrame from event IDs"""
        events_data = []
        for event_id in event_ids:
            event = next((e for e in self.ocel_data['ocel:events'] if e['ocel:id'] == event_id), None)
            if event:
                events_data.append({
                    'event_id': event_id,
                    'timestamp': event['ocel:timestamp'],
                    'activity': event['ocel:activity'],
                    'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                    'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                    'objects': ', '.join(obj['id'] for obj in event.get('ocel:objects', []))
                })
        return pd.DataFrame(events_data)

    def _get_event_details(self, event_id: str) -> Dict:
        """Helper method to get event details from OCEL data"""
        event = next((e for e in self.ocel_data['ocel:events'] if e['ocel:id'] == event_id), None)
        if event:
            return {
                'Event ID': event_id,
                'Timestamp': event['ocel:timestamp'],
                'Activity': event['ocel:activity'],
                'Resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                'Case ID': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                'Objects': [obj['id'] for obj in event.get('ocel:objects', [])]
            }
        return {}

