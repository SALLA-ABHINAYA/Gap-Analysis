import traceback
import streamlit as st
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
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

            # self._process_outliers() #Note : _process_outliers is already getting called in _process_data above

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
        """Optimized case complexity outlier detection"""
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

                        # Get case specific events for this object type
                        case_events = case_data[case_data['object_types'].apply(lambda x: obj_type in x)]

                        failures['incomplete_cases'].append({
                            'case_id': case_id,
                            'object_type': obj_type,
                            'missing_activities': violation['missing_activities'],
                            'completed_activities': case_events['activity'].tolist(),
                            'events': case_events['event_id'].tolist(),
                            'last_event': {
                                'event_id': case_events.iloc[-1][
                                    'event_id'] if not case_events.empty else None,
                                'activity': case_events.iloc[-1][
                                    'activity'] if not case_events.empty else None,
                                'timestamp': case_events.iloc[-1][
                                    'timestamp'] if not case_events.empty else None,
                                'resource': case_events.iloc[-1][
                                    'resource'] if not case_events.empty else None
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

    def create_failure_pattern_visualization(self, failures: Dict) -> go.Figure:
        """Create enhanced visualization for all failure patterns"""
        # Count occurrences of each failure type
        failure_counts = {
            'Sequence Violations': len(failures.get('sequence_violations', [])),
            'Incomplete Cases': len(failures.get('incomplete_cases', [])),
            'Long Running Cases': len(failures.get('long_running', [])),
            'Resource Switches': len(failures.get('resource_switches', [])),
            'Rework Activities': len(failures.get('rework_activities', []))
        }

        # Create bar chart
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
            showlegend=False,
            height=400,
            template='plotly_dark'
        )

        return fig

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

    def _format_ai_response(self, response_text: str) -> Dict:
        """Helper method to format AI response with enhanced logging"""
        logger.info("Formatting AI response")
        try:
            # First try to parse as JSON if response happens to be JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.debug("Response is not JSON format, formatting as structured text")

                # Format text response into sections
                sections = response_text.split('\n')
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

                logger.debug(f"Formatted response: {formatted_response}")
                return formatted_response

        except Exception as e:
            logger.error(f"Error formatting AI response: {str(e)}")
            logger.error(f"Original response: {response_text}")
            return {
                'summary': 'Error formatting response',
                'insights': [],
                'recommendations': []
            }

    def get_explanation(self, tab_type: str, metrics: Dict) -> Dict:
        """Get AI explanation with enhanced logging"""
        logger.info(f"Generating explanation for tab: {tab_type}")
        logger.info(f"Input metrics: {metrics}")

        try:
            # Map tab types to analysis contexts
            context_map = {
                'resource': {
                    'title': 'Resource Workload Analysis',
                    'metrics': [
                        f"Market Maker B workload: {metrics.get('market_maker_b', 'N/A')} cases",
                        f"Client Desk D workload: {metrics.get('client_desk_d', 'N/A')} cases",
                        f"Options Desk A workload: {metrics.get('options_desk_a', 'N/A')} cases"
                    ]
                },
                'case': {
                    'title': 'Case Complexity Analysis',
                    'metrics': [
                        f"Complex cases: {metrics.get('complex_cases', 0)}",
                        f"Timing issues: {metrics.get('timestamp_cases', 0)}",
                        f"Multi-object cases: {metrics.get('object_cases', 0)}"
                    ]
                },
                'time': {
                    'title': 'Time Analysis',
                    'metrics': [
                        f"Average duration: {metrics.get('avg_duration', 'N/A')}",
                        f"Duration outliers: {metrics.get('outlier_count', 0)}",
                        f"Typical duration: {metrics.get('typical_duration', 'N/A')}"
                    ]
                },
                'failure': {
                    'title': 'Failure Pattern Analysis',
                    'metrics': [
                        f"Sequence violations: {metrics.get('sequence_violations', 0)}",
                        f"Incomplete cases: {metrics.get('incomplete_cases', 0)}",
                        f"Long running cases: {metrics.get('long_running', 0)}"
                    ]
                }
            }

            context = context_map.get(tab_type)
            if not context:
                logger.error(f"Unknown tab type: {tab_type}")
                return {"error": f"Unknown analysis type: {tab_type}"}

            logger.debug(f"Using context: {context}")

            prompt = f"""
            Analyze {context['title']}:
            Metrics:
            {chr(10).join(f'- {metric}' for metric in context['metrics'])}

            Provide analysis in following format:
            1. Brief summary of the situation
            2. Key insights about patterns and anomalies
            3. Specific recommendations for improvement

            Keep each section concise and actionable.
            """

            logger.debug(f"Generated prompt: {prompt}")

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are a process mining expert explaining patterns to business users."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )

            logger.debug("Received response from OpenAI")
            response_text = response.choices[0].message.content

            # Format and structure the response
            formatted_response = self._format_ai_response(response_text)
            logger.info("Successfully generated and formatted explanation")
            return formatted_response

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": "Unable to generate explanation",
                "details": str(e)
            }

    def display_enhanced_analysis(self):
        """Display enhanced analysis with comprehensive outlier tracing"""
        try:
            # Create tabs for different analyses
            tabs = st.tabs(["Resource Outlier", "Time Outlier", "Case Outlier       ", "Failure Patterns"])

            # Resource Analysis Tab
            with tabs[0]:
                logger.debug("Processing Resource Analysis tab")
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Create resource workload visualization
                    workload_data = []
                    resource_details = {}

                    for resource, metrics in self.outliers['resource_load'].items():
                        workload_data.append({
                            'Resource': resource,
                            'Z-Score': metrics.z_score,
                            'Is Outlier': metrics.is_outlier,
                            'Total Events': metrics.details['metrics']['total_events'],
                            'Cases': len(metrics.details['events']['cases']),
                            'Activities': len(metrics.details['events']['activities'])
                        })

                        # Store all details for traceability
                        resource_details[resource] = {
                            'metrics': metrics.details['metrics'],
                            'events': metrics.details['events'],
                            'patterns': metrics.details.get('outlier_patterns', {})
                        }

                    # Create resource plot
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
                            details = resource_details[selected_resource]
                            st.write("### Resource Details")

                            # Show if resource is an outlier
                            if workload_data[
                                next(i for i, x in enumerate(workload_data) if x['Resource'] == selected_resource)][
                                'Is Outlier']:
                                st.warning("⚠️ This resource is identified as an outlier")

                            # Show metrics
                            cols = st.columns(3)
                            cols[0].metric("Total Events", details['metrics']['total_events'])
                            cols[1].metric("Cases", len(details['events']['cases']))
                            cols[2].metric("Activities", len(details['events']['activities']))

                            # Show activity distribution
                            st.write("### Activity Distribution")
                            events_df = self._get_events_dataframe(details['events']['all_events'])
                            if not events_df.empty:
                                activity_counts = events_df['activity'].value_counts()
                                activity_fig = px.bar(
                                    x=activity_counts.index,
                                    y=activity_counts.values,
                                    title=f'Activities Performed by {selected_resource}',
                                    labels={'x': 'Activity', 'y': 'Count'}
                                )
                                st.plotly_chart(activity_fig)

                            # Show case involvement timeline
                            st.write("### Case Timeline")
                            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
                            timeline_fig = px.scatter(
                                events_df,
                                x='timestamp',
                                y='case_id',
                                color='activity',
                                title=f'Case Involvement Timeline - {selected_resource}',
                                labels={'timestamp': 'Time', 'case_id': 'Case', 'activity': 'Activity'}
                            )
                            timeline_fig.update_layout(height=400)
                            st.plotly_chart(timeline_fig)

                            # Show objects handled
                            st.write("### Objects Handled")
                            object_series = pd.Series(
                                [obj.strip() for obj in events_df['objects'].str.split(',').explode()])
                            object_counts = object_series.value_counts()
                            object_fig = px.pie(
                                values=object_counts.values,
                                names=object_counts.index,
                                title=f'Object Types Handled by {selected_resource}'
                            )
                            st.plotly_chart(object_fig)

                            # Show detailed event table
                            st.write("### Event Details")
                            st.dataframe(events_df.sort_values('timestamp'))

                with col2:
                    workload_metrics = {
                        'market_maker_b': len(self.resource_events.get('Market Maker B', [])),
                        'client_desk_d': len(self.resource_events.get('Client Desk D', [])),
                        'options_desk_a': len(self.resource_events.get('Options Desk A', []))
                    }
                    st.markdown("### Understanding Resource Distribution")
                    explanation = self.get_explanation('resource', workload_metrics)
                    st.write(explanation.get('summary', ''))

                    if explanation.get('insights'):
                        st.write("#### Key Insights")
                        for insight in explanation['insights']:
                            st.write(f"• {insight}")

                    if explanation.get('recommendations'):
                        st.write("#### Recommendations")
                        for rec in explanation['recommendations']:
                            st.write(f"• {rec}")

            # Inside Time Analysis Tab section of display_enhanced_analysis method
            with tabs[1]:  # Time Analysis Tab
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Keep existing duration outlier visualization
                    duration_data = []
                    duration_outliers = {}

                    for activity, metrics in self.outliers['duration'].items():
                        duration_data.append({
                            'Activity': activity,
                            'Z-Score': metrics.z_score,
                            'Is Outlier': metrics.is_outlier,
                            'Total Events': metrics.details['total_events'],
                            'Violation Count': len(metrics.details.get('outlier_events', {}).get('timing_gap', [])),
                        })

                        if metrics.is_outlier:
                            duration_outliers[activity] = metrics.details

                    # Original scatter plot visualization
                    fig = px.scatter(
                        pd.DataFrame(duration_data),
                        x='Activity',
                        y='Z-Score',
                        size='Total Events',
                        color='Is Outlier',
                        title='Activity Duration Distribution',
                        custom_data=['Violation Count']
                    )

                    fig.update_traces(
                        hovertemplate=(
                                "<b>%{x}</b><br>" +
                                "Z-Score: %{y:.2f}<br>" +
                                "Total Events: %{marker.size}<br>" +
                                "Violations: %{customdata[0]}<br>" +
                                "<extra></extra>"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Add new timing gap analysis section
                    st.subheader("Timing Gap Details")
                    timing_gaps = []
                    for activity, metrics in self.outliers['duration'].items():
                        for violation in metrics.details.get('outlier_events', {}).get('timing_gap', []):
                            # Get previous event details
                            prev_event = self._get_event_details(violation['details']['previous_event'])

                            timing_gaps.append({
                                'Current Activity': activity,
                                'Previous Activity': prev_event.get('Activity', 'Unknown'),
                                'Gap (Minutes)': violation['details']['time_gap_minutes'],
                                'Threshold': violation['details']['threshold_minutes'],
                                'Case ID': violation['case_id'],
                                'Object ID': violation['object_id']
                            })

                    if timing_gaps:
                        df = pd.DataFrame(timing_gaps)

                        # Create timing gap table
                        gap_table = go.Figure(data=[go.Table(
                            header=dict(
                                values=['Case ID', 'Current Activity', 'Previous Activity',
                                        'Gap (Minutes)', 'Threshold'],
                                fill_color='paleturquoise',
                                align='left'
                            ),
                            cells=dict(
                                values=[
                                    df['Case ID'],
                                    df['Current Activity'],
                                    df['Previous Activity'],
                                    df['Gap (Minutes)'].round(2),
                                    df['Threshold']
                                ],
                                fill_color=[['pink' if gap > thresh else 'lightgreen'
                                             for gap, thresh in zip(df['Gap (Minutes)'], df['Threshold'])]],
                                align='left'
                            )
                        )])

                        gap_table.update_layout(
                            title="Detailed Timing Gap Analysis",
                            height=400
                        )
                        st.plotly_chart(gap_table, use_container_width=True)

                with col2:
                    # Keep existing explanation section
                    timing_metrics = {
                        'avg_duration': f"{np.mean([len(events) for events in self.case_events.values()]):.2f}",
                        'outlier_count': len([m for m in self.outliers['duration'].values() if m.is_outlier]),
                        'typical_duration': "2-4 hours"
                    }
                    st.markdown("### Understanding Time Patterns")
                    explanation = self.get_explanation('time', timing_metrics)
                    st.write(explanation.get('summary', ''))

                    if explanation.get('insights'):
                        st.write("#### Key Insights")
                        for insight in explanation['insights']:
                            st.write(f"• {insight}")

                    if explanation.get('recommendations'):
                        st.write("#### Recommendations")
                        for rec in explanation['recommendations']:
                            st.write(f"• {rec}")

            # Case Analysis Tab
            with tabs[2]:
                logger.debug("Processing Case Analysis tab")
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Create case outlier visualization
                    case_data = []
                    case_details = {}  # Store all cases data

                    for case_id, metrics in self.outliers['case_complexity'].items():
                        case_data.append({
                            'Case': case_id,
                            'Z-Score': metrics.z_score,
                            'Is Outlier': metrics.is_outlier,
                            'Total Events': metrics.details['metrics']['total_events'],
                            'Activity Count': metrics.details['metrics']['activity_variety']
                        })

                        # Store details for all cases
                        case_details[case_id] = metrics.details

                    # Create case plot
                    fig = px.scatter(
                        pd.DataFrame(case_data),
                        x='Case',
                        y='Z-Score',
                        size='Total Events',
                        color='Is Outlier',
                        title='Case Complexity Distribution',
                        custom_data=['Activity Count']
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

                    # Show case details for any selected case
                    if case_details:
                        selected_case = st.selectbox(
                            "Select case for details",
                            options=list(case_details.keys())
                        )
                        if selected_case:
                            details = case_details[selected_case]
                            st.write("### Case Details")

                            # Add outlier indicator
                            if case_data[next(i for i, x in enumerate(case_data) if x['Case'] == selected_case)][
                                'Is Outlier']:
                                st.warning("⚠️ This case is identified as an outlier")

                            # Show metrics
                            cols = st.columns(3)
                            cols[0].metric("Total Events", details['metrics']['total_events'])
                            cols[1].metric("Activities", details['metrics']['activity_variety'])
                            cols[2].metric("Duration (hrs)", f"{details['temporal']['duration_hours']:.1f}")

                            # Show timeline
                            events_df = self._get_events_dataframe(details['events']['all_events'])
                            if not events_df.empty:
                                events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
                                timeline_fig = px.scatter(
                                    events_df,
                                    x='timestamp',
                                    y='activity',
                                    color='resource',
                                    title=f'Case Timeline - {selected_case}',
                                    labels={'timestamp': 'Time', 'activity': 'Activity', 'resource': 'Resource'}
                                )
                                timeline_fig.update_layout(
                                    height=400,
                                    showlegend=True,
                                    hovermode='closest'
                                )
                                st.plotly_chart(timeline_fig)

                                # Show event details table
                                st.write("### Event Details")
                                st.dataframe(events_df.style.highlight_max(subset=['timestamp']))

                with col2:
                    case_metrics = {
                        'complex_cases': len([c for c in self.outliers['case_complexity'].values() if
                                              c.details['metrics']['total_events'] > 10]),
                        'timestamp_cases': len([c for c in self.outliers['case_complexity'].values() if
                                                c.details['temporal']['duration_hours'] > 24]),
                        'object_cases': len(self.outliers['case_complexity'])
                    }
                    st.markdown("### Understanding Case Complexity")
                    explanation = self.get_explanation('case', case_metrics)
                    st.write(explanation.get('summary', ''))

                    if explanation.get('insights'):
                        st.write("#### Key Insights")
                        for insight in explanation['insights']:
                            st.write(f"• {insight}")

                    if explanation.get('recommendations'):
                        st.write("#### Recommendations")
                        for rec in explanation['recommendations']:
                            st.write(f"• {rec}")

            # Failure Patterns Tab
            with tabs[3]:
                logger.debug("Processing Failure Patterns tab")
                col1, col2 = st.columns([2, 1])

                with col1:
                    failure_counts = {
                        'Sequence Violations': len(self.outliers['failures']['sequence_violations']),
                        'Incomplete Cases': len(self.outliers['failures']['incomplete_cases']),
                        'Long Running': len(self.outliers['failures']['long_running']),
                        'Resource Switches': len(self.outliers['failures']['resource_switches']),
                        'Rework Activities': len(self.outliers['failures']['rework_activities'])
                    }

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
                    for pattern, failures in self.outliers['failures'].items():
                        if failures:
                            with st.expander(f"{pattern.replace('_', ' ').title()} ({len(failures)})"):
                                failure_df = pd.DataFrame(failures)
                                st.dataframe(failure_df)

                with col2:
                    failure_metrics = {
                        'sequence_violations': len(self.outliers['failures']['sequence_violations']),
                        'incomplete_cases': len(self.outliers['failures']['incomplete_cases']),
                        'long_running': len(self.outliers['failures']['long_running'])
                    }
                    st.markdown("### Understanding Failure Patterns")
                    explanation = self.get_explanation('failure', failure_metrics)
                    st.write(explanation.get('summary', ''))

                    if explanation.get('insights'):
                        st.write("#### Key Insights")
                        for insight in explanation['insights']:
                            st.write(f"• {insight}")

                    if explanation.get('recommendations'):
                        st.write("#### Recommendations")
                        for rec in explanation['recommendations']:
                            st.write(f"• {rec}")

        except Exception as e:
            logger.error(f"Error in display_enhanced_analysis: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("Error analyzing process patterns. Check logs for details.")

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

