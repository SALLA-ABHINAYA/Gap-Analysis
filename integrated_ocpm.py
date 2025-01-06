import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pm4py
from dataclasses import dataclass
from collections import defaultdict
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json
from pathlib import Path
import traceback


@dataclass
class ObjectType:
    """Represents an object type in the OCPM model"""
    name: str
    activities: List[str]
    attributes: List[str]
    relationships: List[str]


class IntegratedOCPMAnalyzer:
    """Enhanced OCPM Analyzer with integrated unfair analysis"""

    def __init__(self, event_log: pd.DataFrame):
        self.event_log = self._preprocess_event_log(event_log)
        self.object_types = self._initialize_object_types()
        self.object_relationships = defaultdict(list)
        self.activity_object_mapping = self._create_activity_mapping()
        self.ocel_data = None
        self.unfair_results = None

    def _preprocess_event_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess event log ensuring consistent format"""
        column_mappings = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp',
            'Case_': 'case_id',
            'Activity': 'activity',
            'Timestamp': 'timestamp',
            'case': 'case_id',
            'event': 'activity',
            'start_timestamp': 'timestamp'
        }

        df = df.copy()

        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        required_columns = ['case_id', 'activity', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def _initialize_object_types(self) -> Dict[str, ObjectType]:
        """Initialize predefined object types with enhanced attributes"""
        return {
            'Trade': ObjectType(
                name='Trade',
                activities=['Trade Initiated', 'Trade Executed', 'Trade Allocated', 'Trade Settled'],
                attributes=['currency_pair', 'notional_amount', 'trade_type'],
                relationships=['Market', 'Risk', 'Settlement']
            ),
            'Market': ObjectType(
                name='Market',
                activities=['Quote Requested', 'Quote Provided', 'Market Analysis'],
                attributes=['market_condition', 'volatility'],
                relationships=['Trade']
            ),
            'Risk': ObjectType(
                name='Risk',
                activities=['Risk Assessment', 'Limit Check', 'Exposure Analysis'],
                attributes=['risk_score', 'exposure_level'],
                relationships=['Trade', 'Settlement']
            ),
            'Settlement': ObjectType(
                name='Settlement',
                activities=['Settlement Processing', 'Position Reconciliation'],
                attributes=['settlement_amount', 'settlement_status'],
                relationships=['Trade', 'Risk']
            )
        }

    def _create_activity_mapping(self) -> Dict[str, List[str]]:
        """Create enhanced mapping of activities to object types"""
        mapping = defaultdict(list)
        for obj_type, obj_info in self.object_types.items():
            for activity in obj_info.activities:
                mapping[activity].append(obj_type)
        return mapping

    def convert_to_ocel(self) -> Dict:
        """Convert regular event log to OCEL format with enhanced attributes"""
        events = []
        objects = defaultdict(dict)
        object_types = set()

        for idx, row in self.event_log.iterrows():
            event_id = f"e{idx}"
            activity = row['activity']
            related_objects = self.activity_object_mapping.get(activity, ['Trade'])

            event = {
                "ocel:eid": event_id,
                "ocel:activity": activity,
                "ocel:timestamp": row['timestamp'].isoformat(),
                "ocel:omap": [],
                "ocel:vmap": {
                    "resource": row.get('resource', 'Unknown'),
                    "cost": float(row.get('cost', 0))
                }
            }

            for obj_type in related_objects:
                object_id = f"{obj_type.lower()}_{row['case_id']}"
                event["ocel:omap"].append(object_id)

                if object_id not in objects[obj_type]:
                    objects[obj_type][object_id] = {
                        "ocel:type": obj_type,
                        "ocel:ovmap": {
                            attr: row.get(f"{obj_type.lower()}_{attr}", None)
                            for attr in self.object_types[obj_type].attributes
                        }
                    }

                object_types.add(obj_type)

            events.append(event)

        self.ocel_data = {
            "ocel:global-event": {"ocel:activity": list(set(self.event_log['activity']))},
            "ocel:global-object": {"ocel:type": list(object_types)},
            "ocel:events": events,
            "ocel:objects": {
                obj_type: list(type_objects.values())
                for obj_type, type_objects in objects.items()
            }
        }

        return self.ocel_data

    def perform_unfair_analysis(self) -> Dict:
        """Perform unfair analysis on OCEL data"""
        if not self.ocel_data:
            self.convert_to_ocel()

        unfair_metrics = {
            'object_interaction_bias': self._calculate_interaction_bias(),
            'temporal_fairness': self._analyze_temporal_fairness(),
            'resource_distribution': self._analyze_resource_distribution(),
            'processing_time_variance': self._analyze_processing_times()
        }

        self.unfair_results = unfair_metrics
        return unfair_metrics

    def _calculate_interaction_bias(self) -> Dict:
        """Calculate bias in object interactions"""
        interactions = defaultdict(int)
        total_interactions = 0

        for event in self.ocel_data['ocel:events']:
            objects = event['ocel:omap']
            for obj1 in objects:
                for obj2 in objects:
                    if obj1 < obj2:
                        interactions[f"{obj1}-{obj2}"] += 1
                        total_interactions += 1

        return {
            'interaction_counts': dict(interactions),
            'bias_score': self._calculate_gini_coefficient(list(interactions.values()))
        }

    def _analyze_temporal_fairness(self) -> Dict:
        """Analyze temporal aspects of fairness"""
        processing_times = defaultdict(list)

        for event in self.ocel_data['ocel:events']:
            activity = event['ocel:activity']
            timestamp = datetime.fromisoformat(event['ocel:timestamp'])
            processing_times[activity].append(timestamp)

        fairness_scores = {}
        for activity, timestamps in processing_times.items():
            if len(timestamps) > 1:
                time_diffs = [(timestamps[i + 1] - timestamps[i]).total_seconds()
                              for i in range(len(timestamps) - 1)]
                fairness_scores[activity] = {
                    'mean_processing_time': np.mean(time_diffs),
                    'std_processing_time': np.std(time_diffs),
                    'fairness_score': 1 - (np.std(time_diffs) / np.mean(time_diffs))
                    if np.mean(time_diffs) > 0 else 0
                }

        return fairness_scores

    def _analyze_resource_distribution(self) -> Dict:
        """Analyze distribution of resources"""
        resource_loads = defaultdict(int)
        resource_activities = defaultdict(set)

        for event in self.ocel_data['ocel:events']:
            resource = event['ocel:vmap'].get('resource', 'Unknown')
            resource_loads[resource] += 1
            resource_activities[resource].add(event['ocel:activity'])

        return {
            'resource_loads': dict(resource_loads),
            'resource_diversity': {
                resource: len(activities)
                for resource, activities in resource_activities.items()
            },
            'distribution_fairness': 1 - self._calculate_gini_coefficient(
                list(resource_loads.values())
            )
        }

    def _analyze_processing_times(self) -> Dict:
        """Analyze variance in processing times"""
        activity_times = defaultdict(list)

        for event in self.ocel_data['ocel:events']:
            activity = event['ocel:activity']
            timestamp = datetime.fromisoformat(event['ocel:timestamp'])
            activity_times[activity].append(timestamp)

        processing_variance = {}
        for activity, timestamps in activity_times.items():
            if len(timestamps) > 1:
                sorted_timestamps = sorted(timestamps)
                intervals = [(sorted_timestamps[i + 1] - sorted_timestamps[i]).total_seconds()
                             for i in range(len(sorted_timestamps) - 1)]
                processing_variance[activity] = {
                    'variance': np.var(intervals),
                    'fairness_score': 1 / (1 + np.var(intervals))
                }

        return processing_variance

    @staticmethod
    def _calculate_gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for fairness measurement"""
        if not values or len(values) < 2:
            return 0.0

        # Ensure float64 array type
        array = np.array(values, dtype=np.float64)

        # Handle negative values if any
        if np.amin(array) < 0:
            array = array - np.amin(array)

        # Add small constant to avoid division by zero
        array = array + 0.0000001

        # Sort array and calculate indices
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1, dtype=np.float64)
        n = array.shape[0]

        # Calculate Gini coefficient
        return float(((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))))


class IntegratedOCPMVisualizer:
    """Enhanced visualizer for integrated OCPM analysis"""

    @staticmethod
    def create_object_interaction_heatmap(interactions: Dict[Tuple[str, str], int]) -> go.Figure:
        """Create enhanced heatmap visualization"""
        object_types = sorted(list(set([obj.split('-')[0].split('_')[0]
                                        for obj in interactions['interaction_counts'].keys()])))
        matrix = np.zeros((len(object_types), len(object_types)))

        for pair, count in interactions['interaction_counts'].items():
            obj1, obj2 = pair.split('-')
            type1 = obj1.split('_')[0]
            type2 = obj2.split('_')[0]
            i, j = object_types.index(type1), object_types.index(type2)
            matrix[i][j] = count
            matrix[j][i] = count

        return go.Figure(data=go.Heatmap(
            z=matrix,
            x=object_types,
            y=object_types,
            colorscale='Viridis',
            colorbar=dict(title='Interaction Count')
        ))

    @staticmethod
    def create_fairness_dashboard(unfair_results: Dict) -> List[go.Figure]:
        """Create comprehensive fairness analysis dashboard"""
        figures = []

        # Resource Distribution
        resource_data = unfair_results['resource_distribution']
        fig_resources = go.Figure(data=[
            go.Bar(
                x=list(resource_data['resource_loads'].keys()),
                y=list(resource_data['resource_loads'].values()),
                name='Resource Load'
            )
        ])
        fig_resources.update_layout(title='Resource Distribution Analysis')
        figures.append(fig_resources)

        # Temporal Fairness
        temporal_data = unfair_results['temporal_fairness']
        activities = list(temporal_data.keys())
        fairness_scores = [data['fairness_score'] for data in temporal_data.values()]

        fig_temporal = go.Figure(data=[
            go.Bar(
                x=activities,
                y=fairness_scores,
                name='Temporal Fairness'
            )
        ])
        fig_temporal.update_layout(title='Temporal Fairness Analysis')
        figures.append(fig_temporal)

        # Processing Time Variance
        processing_data = unfair_results['processing_time_variance']
        activities = list(processing_data.keys())
        fairness_scores = [data['fairness_score'] for data in processing_data.values()]

        fig_processing = go.Figure(data=[
            go.Bar(
                x=activities,
                y=fairness_scores,
                name='Processing Fairness'
            )
        ])
        fig_processing.update_layout(title='Processing Time Fairness')
        figures.append(fig_processing)

        return figures


def create_integrated_ocpm_ui():
    """Create integrated Streamlit UI for OCPM analysis"""
    st.title("Integrated Object-Centric Process Mining Analysis")

    uploaded_file = st.file_uploader("Upload Event Log (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read and process the event log
            df = pd.read_csv(uploaded_file, sep=';')
            analyzer = IntegratedOCPMAnalyzer(df)

            # Convert to OCEL and perform both analyses
            ocel_data = analyzer.convert_to_ocel()
            unfair_results = analyzer.perform_unfair_analysis()

            # Create tabs for different analyses
            tabs = st.tabs(["OCEL Analysis", "Fairness Analysis", "Visualizations"])

            # OCEL Analysis Tab
            with tabs[0]:
                st.subheader("OCEL Data Analysis")
                st.json(ocel_data)

                st.subheader("Object Type Statistics")
                obj_stats = {obj_type: len(objs) for obj_type, objs in ocel_data['ocel:objects'].items()}
                st.bar_chart(obj_stats)

            # Fairness Analysis Tab
            with tabs[1]:
                st.subheader("Fairness Analysis Results")

                # Display interaction bias
                st.write("### Interaction Bias Analysis")
                bias_score = unfair_results['object_interaction_bias']['bias_score']
                st.metric("Overall Interaction Bias Score", f"{bias_score:.3f}")

                # Display temporal fairness
                st.write("### Temporal Fairness Analysis")
                for activity, metrics in unfair_results['temporal_fairness'].items():
                    st.write(f"Activity: {activity}")
                    cols = st.columns(3)
                    cols[0].metric("Mean Processing Time", f"{metrics['mean_processing_time']:.2f}s")
                    cols[1].metric("Std Processing Time", f"{metrics['std_processing_time']:.2f}s")
                    cols[2].metric("Fairness Score", f"{metrics['fairness_score']:.3f}")

                # Display resource distribution
                st.write("### Resource Distribution Analysis")
                resource_fairness = unfair_results['resource_distribution']['distribution_fairness']
                st.metric("Resource Distribution Fairness", f"{resource_fairness:.3f}")

            # Visualizations Tab
            with tabs[2]:
                st.subheader("Interactive Visualizations")

                # Object Interaction Heatmap
                st.write("### Object Interaction Patterns")
                heatmap = IntegratedOCPMVisualizer.create_object_interaction_heatmap(
                    unfair_results['object_interaction_bias']
                )
                st.plotly_chart(heatmap)

                # Fairness Dashboard
                st.write("### Fairness Analysis Dashboard")
                fairness_figures = IntegratedOCPMVisualizer.create_fairness_dashboard(unfair_results)
                for fig in fairness_figures:
                    st.plotly_chart(fig)

            # Save results
            output_dir = Path("ocpm_output")
            output_dir.mkdir(exist_ok=True)

            # Save OCEL data
            with open(output_dir / "ocel_data.json", "w") as f:
                json.dump(ocel_data, f, indent=2, default=str)

            # Save unfairness analysis
            with open(output_dir / "unfairness_analysis.json", "w") as f:
                json.dump(unfair_results, f, indent=2, default=str)

            st.success("Analysis completed! Results saved in 'ocpm_output' directory.")

        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.write("Debug information:")
            st.code(traceback.format_exc())


if __name__ == "__main__":
    st.set_page_config(page_title="Integrated OCPM Analysis", layout="wide")
    create_integrated_ocpm_ui()