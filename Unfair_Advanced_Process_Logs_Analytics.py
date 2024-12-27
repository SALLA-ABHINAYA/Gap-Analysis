# Unfair_Advanced_Process_Logs_Analytics.py
import streamlit as st
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')


class UnfairOCELAnalyzer:
    """Optimized analyzer for identifying unfairness in process mining"""

    def __init__(self, ocel_path: str):
        """Initialize analyzer with OCEL file path"""
        try:
            # Load OCEL data
            with open(ocel_path, 'r', encoding='utf-8') as f:
                self.ocel_data = json.load(f)

            if not isinstance(self.ocel_data, dict):
                raise ValueError(f"Invalid OCEL data type: {type(self.ocel_data)}")

            if 'ocel:events' not in self.ocel_data:
                raise ValueError("Missing ocel:events in data")

            # Initialize caches
            self._metrics_cache = {}
            self._df_cache = {}
            self._trace_cache = {}  # New cache for traces

            # Process data efficiently
            self._process_data()

        except Exception as e:
            st.error(f"Error loading OCEL file: {str(e)}")
            raise

    def _process_data(self):
        """Process OCEL data into efficient DataFrame format"""
        try:
            # Process events in batch
            events_data = []
            relationships_data = []
            trace_data = []  # New trace data storage

            for event in self.ocel_data['ocel:events']:
                event_base = {
                    'event_id': event['ocel:id'],
                    'timestamp': pd.to_datetime(event['ocel:timestamp']),
                    'activity': event['ocel:activity'],
                    'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                    'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown')
                }

                # Store event details for tracing
                trace_data.append({
                    **event_base,
                    'objects': [obj['id'] for obj in event.get('ocel:objects', [])],
                    'raw_event': event  # Store complete event for traceability
                })

                events_data.append(event_base)

                # Process relationships
                for obj in event.get('ocel:objects', []):
                    relationships_data.append({
                        **event_base,
                        'object_id': obj['id'],
                        'object_type': obj.get('type', 'Unknown')
                    })

            # Create DataFrames with optimized dtypes
            if events_data:
                self.events_df = pd.DataFrame(events_data)
                self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'])
                self.events_df.set_index('event_id', inplace=True)
            else:
                raise ValueError("No events data found in OCEL file")

            if relationships_data:
                self.relationships_df = pd.DataFrame(relationships_data)
                self.relationships_df['timestamp'] = pd.to_datetime(self.relationships_df['timestamp'])
                self.relationships_df.sort_values(['case_id', 'timestamp'], inplace=True)
            else:
                raise ValueError("No relationships data found in OCEL file")

            # Create trace DataFrame and build indices
            self.trace_df = pd.DataFrame(trace_data)
            self._build_trace_index()

        except Exception as e:
            st.error(f"Error processing OCEL data: {str(e)}")
            raise

    def _build_trace_index(self):
        """Build indices for quick trace lookups"""
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

    def get_trace_for_metric(self, metric_type: str, identifier: str) -> List[Dict]:
        """Get traced events for a specific metric"""
        events = []

        if metric_type == 'resource':
            events = self.resource_events.get(identifier, [])
        elif metric_type == 'case':
            events = self.case_events.get(identifier, [])
        elif metric_type == 'activity':
            events = self.activity_events.get(identifier, [])
        elif metric_type == 'handover':
            source, target = identifier.split('->')
            source_events = set(self.resource_events.get(source, []))
            target_events = set(self.resource_events.get(target, []))
            events = list(source_events.union(target_events))

        return [
            self.trace_df[self.trace_df['event_id'] == event_id].to_dict('records')[0]
            for event_id in events
        ]

    @lru_cache(maxsize=32)
    def _calculate_resource_metrics(self) -> Dict:
        """Calculate resource metrics with optimized operations"""
        metrics = {}

        try:
            resource_stats = self.relationships_df.groupby('resource').agg({
                'case_id': 'nunique',
                'object_id': 'count'
            })

            total_cases = self.relationships_df['case_id'].nunique()
            expected_cases = total_cases / len(resource_stats) if len(resource_stats) > 0 else 0

            for resource, stats in resource_stats.iterrows():
                bias_score = (stats['case_id'] - expected_cases) / expected_cases if expected_cases > 0 else 0

                # Get traced events for this resource
                traced_events = self.get_trace_for_metric('resource', resource)

                metrics[resource] = {
                    'cases': int(stats['case_id']),
                    'bias_score': float(bias_score),
                    'percentage': float((stats['case_id'] / total_cases) * 100) if total_cases > 0 else 0,
                    'supporting_events': traced_events  # Add tracing information
                }

            return metrics
        except Exception as e:
            st.error(f"Error calculating resource metrics: {str(e)}")
            return {}

    @lru_cache(maxsize=32)
    def _calculate_time_metrics(self) -> Dict:
        """Calculate time-based metrics efficiently"""
        metrics = {}

        try:
            # Calculate durations using vectorized operations
            duration_df = self.relationships_df.groupby(['object_id', 'resource']).agg({
                'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600
            }).reset_index()

            for resource in duration_df['resource'].unique():
                resource_durations = duration_df[duration_df['resource'] == resource]['timestamp']

                if len(resource_durations) > 0:
                    metrics[resource] = {
                        'mean_time': float(resource_durations.mean()),
                        'median_time': float(resource_durations.median()),
                        'std_dev': float(resource_durations.std()),
                        'min_time': float(resource_durations.min()),
                        'max_time': float(resource_durations.max())
                    }

            return metrics
        except Exception as e:
            st.error(f"Error calculating time metrics: {str(e)}")
            return {}

    @lru_cache(maxsize=32)
    def _calculate_case_metrics(self) -> Dict:
        """Calculate case-related metrics efficiently"""
        metrics = {}

        try:
            # Calculate case durations
            case_duration_df = self.relationships_df.groupby(['case_id', 'activity']).agg({
                'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600
            }).reset_index()

            overall_mean = case_duration_df['timestamp'].mean() if not case_duration_df.empty else 0

            for activity in case_duration_df['activity'].unique():
                activity_durations = case_duration_df[case_duration_df['activity'] == activity]['timestamp']

                if len(activity_durations) > 0:
                    mean_duration = activity_durations.mean()
                    metrics[activity] = {
                        'mean_duration': float(mean_duration),
                        'bias_score': float((mean_duration - overall_mean) / overall_mean if overall_mean > 0 else 0),
                        'total_cases': int(len(activity_durations))
                    }

            return metrics
        except Exception as e:
            st.error(f"Error calculating case metrics: {str(e)}")
            return {}

    @lru_cache(maxsize=32)
    def _calculate_handover_metrics(self) -> Dict:
        """Calculate handover metrics efficiently"""
        metrics = {}

        try:
            # Calculate handovers using shift operation
            df_temp = self.relationships_df.sort_values(['object_id', 'timestamp']).copy()
            df_temp['next_resource'] = df_temp.groupby('object_id')['resource'].shift(-1)

            handover_pairs = df_temp.dropna(subset=['next_resource']).groupby(
                ['resource', 'next_resource']).size()

            total_handovers = handover_pairs.sum() if not handover_pairs.empty else 0

            if total_handovers > 0:
                for (source, target), count in handover_pairs.items():
                    key = f"{source}->{target}"
                    metrics[key] = {
                        'count': int(count),
                        'percentage': float((count / total_handovers) * 100)
                    }

            return metrics
        except Exception as e:
            st.error(f"Error calculating handover metrics: {str(e)}")
            return {}

    def get_analysis_plots(self) -> Tuple[Dict[str, plt.Figure], Dict[str, Dict]]:
        """Generate all analysis plots and metrics efficiently"""
        try:
            # Calculate metrics first (using cached results)
            metrics = {
                'resource': self._calculate_resource_metrics(),
                'time': self._calculate_time_metrics(),
                'case': self._calculate_case_metrics(),
                'handover': self._calculate_handover_metrics()
            }

            # Generate plots
            plots = {
                'resource_discrimination': self._create_resource_plot(metrics['resource']),
                'time_bias': self._create_time_plot(metrics['time']),
                'case_priority': self._create_case_plot(metrics['case']),
                'handover': self._create_handover_plot(metrics['handover'])
            }

            return plots, metrics

        except Exception as e:
            st.error(f"Error generating analysis plots: {str(e)}")
            return {}, {}

    def _create_resource_plot(self, metrics: Dict) -> plt.Figure:
        """Create resource discrimination plot"""
        fig, ax = plt.subplots(figsize=(12, 6))

        try:
            if not metrics:
                ax.text(0.5, 0.5, 'No resource data available',
                        ha='center', va='center')
                return fig

            resources = list(metrics.keys())
            bias_scores = [m['bias_score'] for m in metrics.values()]
            colors = ['red' if x > 0.2 else 'orange' if x > 0 else 'green' for x in bias_scores]

            ax.bar(resources, bias_scores, color=colors)
            ax.set_xticklabels(resources, rotation=45, ha='right')
            ax.set_title('Resource Discrimination Analysis')
            ax.set_ylabel('Bias Score')
            plt.tight_layout()

        except Exception as e:
            st.error(f"Error creating resource plot: {str(e)}")
            ax.text(0.5, 0.5, 'Error generating plot',
                    ha='center', va='center')

        return fig

    def generate_trace_report(self, metric_type: str, identifier: str) -> str:
        """Generate a trace report for a specific metric"""
        events = self.get_trace_for_metric(metric_type, identifier)
        report = [f"Trace Report for {metric_type}: {identifier}\n"]
        report.append("Supporting Events:")

        for event in events:
            report.append(f"- Event ID: {event['event_id']}")
            report.append(f"  Activity: {event['activity']}")
            report.append(f"  Timestamp: {event['timestamp']}")
            report.append(f"  Resource: {event['resource']}")
            report.append(f"  Case ID: {event['case_id']}\n")

        return "\n".join(report)

    def _create_time_plot(self, metrics: Dict) -> plt.Figure:
        """Create time bias plot"""
        fig, ax = plt.subplots(figsize=(12, 6))

        try:
            if not metrics:
                ax.text(0.5, 0.5, 'No time data available',
                        ha='center', va='center')
                return fig

            resources = list(metrics.keys())
            means = [m['mean_time'] for m in metrics.values()]
            stds = [m['std_dev'] for m in metrics.values()]

            ax.bar(resources, means, yerr=stds)
            ax.set_xticklabels(resources, rotation=45, ha='right')
            ax.set_title('Processing Time Distribution')
            ax.set_ylabel('Hours')
            plt.tight_layout()

        except Exception as e:
            st.error(f"Error creating time plot: {str(e)}")
            ax.text(0.5, 0.5, 'Error generating plot',
                    ha='center', va='center')

        return fig

    def _create_case_plot(self, metrics: Dict) -> plt.Figure:
        """Create case priority plot"""
        fig, ax = plt.subplots(figsize=(12, 6))

        try:
            if not metrics:
                ax.text(0.5, 0.5, 'No case data available',
                        ha='center', va='center')
                return fig

            activities = list(metrics.keys())
            bias_scores = [m['bias_score'] for m in metrics.values()]
            colors = ['red' if x > 0.2 else 'orange' if x > 0 else 'green' for x in bias_scores]

            ax.bar(activities, bias_scores, color=colors)
            ax.set_xticklabels(activities, rotation=45, ha='right')
            ax.set_title('Case Priority Analysis')
            ax.set_ylabel('Bias Score')
            plt.tight_layout()

        except Exception as e:
            st.error(f"Error creating case plot: {str(e)}")
            ax.text(0.5, 0.5, 'Error generating plot',
                    ha='center', va='center')

        return fig

    def _create_handover_plot(self, metrics: Dict) -> plt.Figure:
        """Create handover pattern plot"""
        fig, ax = plt.subplots(figsize=(12, 10))

        try:
            if not metrics:
                ax.text(0.5, 0.5, 'No handover data available',
                        ha='center', va='center')
                return fig

            # Create matrix for heatmap
            resources = sorted(set(k.split('->')[0] for k in metrics.keys()))
            matrix = np.zeros((len(resources), len(resources)))

            for handover, data in metrics.items():
                source, target = handover.split('->')
                i = resources.index(source)
                j = resources.index(target)
                matrix[i, j] = data['percentage']

            sns.heatmap(matrix,
                        xticklabels=resources,
                        yticklabels=resources,
                        cmap='RdYlGn_r',
                        annot=True,
                        fmt='.2%',
                        ax=ax)
            ax.set_title('Handover Pattern Ansealysis')
            plt.tight_layout()

        except Exception as e:
            st.error(f"Error creating handover plot: {str(e)}")
            ax.text(0.5, 0.5, 'Error generating plot',
                    ha='center', va='center')

        return fig