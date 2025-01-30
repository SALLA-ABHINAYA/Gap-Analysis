# ocpm_analysis.py

import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os

# Add this at the start of your script, before any other imports
os.environ["PATH"] += os.pathsep + r"C:\samadhi\technology\Graphviz\bin"

@dataclass
class ObjectType:
    """Represents an object type in the OCPM model"""
    name: str
    activities: List[str]
    attributes: List[str]
    relationships: List[str]


class OCPMAnalyzer:
    """Main class for Object-Centric Process Mining analysis"""

    def __init__(self, event_log: pd.DataFrame):
        # Log the original columns for debugging
        print(f"Original columns: {event_log.columns.tolist()}")

        self.event_log                  = self._preprocess_event_log(event_log)
        self.object_types               = self._initialize_object_types()
        self.object_relationships       = defaultdict(list)
        self.activity_object_mapping    = self._create_activity_mapping()


    def convert_to_ocel(self) -> Dict:
        
        """Convert OCPM data to OCEL format."""
        ocpm_df = self.convert_to_ocpm_format()

        ocel_data = {
            "ocel:version": "1.0",
            "ocel:ordering": "timestamp",
            "ocel:attribute-names": [],
            "ocel:events": [],
            "ocel:objects": [],
            "ocel:object-types": list(self.object_types.keys()),
            "ocel:global-log": {
                "ocel:attribute-names": []
            }
        }

        # Convert events
        events_map = {}  # To track unique events
        for _, row in ocpm_df.iterrows():
            event_id = f"{row['case_id']}_{row['activity']}"
            if event_id not in events_map:
                event = {
                    "ocel:id": event_id,
                    "ocel:timestamp": row['timestamp'].isoformat(),
                    "ocel:activity": row['activity'],
                    "ocel:type": "event",
                    "ocel:attributes": {
                        "resource": row['resource'],
                        "case_id": row['case_id'],
                        "object_type": row['object_type']
                    },
                    "ocel:objects": [{
                        "id": row['object_id'],
                        "type": row['object_type']
                    }]
                }
                ocel_data["ocel:events"].append(event)
                events_map[event_id] = True

        # Convert objects
        objects_map = {}  # To track unique objects
        for obj_type, obj_info in self.object_types.items():
            for activity in obj_info.activities:
                obj_id = f"{obj_type}_{activity}"
                if obj_id not in objects_map:
                    obj = {
                        "ocel:id": obj_id,
                        "ocel:type": obj_type,
                        "ocel:attributes": {
                            "activity": activity,
                            "attributes": obj_info.attributes
                        }
                    }
                    ocel_data["ocel:objects"].append(obj)
                    objects_map[obj_id] = True

        return ocel_data

    def save_ocel(self, output_path: str = "ocpm_output/process_data.json") -> str:
        """Save OCPM data in OCEL format."""
        import json
        from pathlib import Path

        try:
            # Generate OCEL data
            ocel_data = self.convert_to_ocel()

            # Debug output
            st.write("Generated OCEL data structure:")
            st.write("Top-level keys:", list(ocel_data.keys()))
            st.write("Number of events:", len(ocel_data.get('ocel:events', [])))

            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save OCEL file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ocel_data, f, indent=2, ensure_ascii=False)

            # Verify the file
            with open(output_path, 'r', encoding='utf-8') as f:
                verification = json.load(f)
                if not verification.get('ocel:events'):
                    st.error("Generated OCEL file has no events")
                    raise ValueError("Generated OCEL file has no events")
                else:
                    st.success(f"Successfully saved {len(verification['ocel:events'])} events")

            return str(output_path)

        except Exception as e:
            st.error(f"Error saving OCEL file: {str(e)}")
            raise

    def _preprocess_event_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess event log to ensure consistent format"""
        # Print incoming dataframe info for debugging
        print("Input DataFrame Info:")
        print(df.info())

        # Define all possible column mappings
        column_mappings = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp',
            'Case_': 'case_id',  # For synthetic data format
            'Activity': 'activity',
            'Timestamp': 'timestamp',
            'case': 'case_id',
            'event': 'activity',
            'start_timestamp': 'timestamp',
            'case_id': 'case_id',  # Already correct format
            'activity': 'activity',  # Already correct format
            'timestamp': 'timestamp'  # Already correct format
        }

        # Create a copy to avoid modifying the original
        df = df.copy()

        # Try to identify and rename columns
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Special handling for case_id if it has a numeric prefix
        case_id_columns = [col for col in df.columns if 'case' in col.lower()]
        if case_id_columns and 'case_id' not in df.columns:
            df = df.rename(columns={case_id_columns[0]: 'case_id'})

        # Special handling for synthetic data format
        if 'case_id' not in df.columns and 'case:concept:name' in df.columns:
            df['case_id'] = df['case:concept:name']
        if 'activity' not in df.columns and 'concept:name' in df.columns:
            df['activity'] = df['concept:name']
        if 'timestamp' not in df.columns and 'time:timestamp' in df.columns:
            df['timestamp'] = df['time:timestamp']

        # Check required columns
        required_columns = ['case_id', 'activity', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print("Available columns:", df.columns.tolist())
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Print final dataframe info for verification
        print("\nProcessed DataFrame Info:")
        print(df.info())
        return df

    def _initialize_object_types(self) -> Dict[str, ObjectType]:
        """Initialize predefined object types for FX trading"""
        return {
            'Trade': ObjectType(
                name='Trade',
                activities=[
                    'Trade Initiated', 'Trade Executed', 'Trade Allocated',
                    'Trade Settled', 'Trade Canceled'  # Match synthetic data activities
                ],
                attributes=['currency_pair', 'notional_amount'],
                relationships=['Market', 'Risk', 'Settlement']
            ),
            'Market': ObjectType(
                name='Market',
                activities=[
                    'Trade Executed', 'Quote Requested', 'Quote Provided'
                ],
                attributes=['currency_pair'],
                relationships=['Trade']
            ),
            'Risk': ObjectType(
                name='Risk',
                activities=[
                    'Trade Allocated', 'Risk Assessment'
                ],
                attributes=['risk_score'],
                relationships=['Trade', 'Settlement']
            ),
            'Settlement': ObjectType(
                name='Settlement',
                activities=[
                    'Trade Settled', 'Position Reconciliation'
                ],
                attributes=['settlement_amount'],
                relationships=['Trade', 'Risk']
            )
        }

    def _create_activity_mapping(self) -> Dict[str, List[str]]:
        """Create mapping of activities to object types"""
        mapping = defaultdict(list)
        for obj_type, obj_info in self.object_types.items():
            for activity in obj_info.activities:
                mapping[activity].append(obj_type)
        return mapping

    def convert_to_ocpm_format(self) -> pd.DataFrame:
        """Convert regular event log to OCPM format"""
        ocpm_events = []

        for _, row in self.event_log.iterrows():
            activity = row['activity']
            related_objects = self.activity_object_mapping.get(activity, ['Trade'])

            for obj_type in related_objects:
                event = {
                    'case_id': row['case_id'],
                    'activity': activity,
                    'timestamp': row['timestamp'],
                    'object_type': obj_type,
                    'object_id': f"{obj_type}_{row['case_id']}",
                    'resource': row.get('resource', 'Unknown'),
                }

                # Add object-specific attributes
                for attr in self.object_types[obj_type].attributes:
                    if attr in row:
                        event[f"{obj_type.lower()}_{attr}"] = row[attr]

                ocpm_events.append(event)

                # Record object relationships
                for related_obj in self.object_types[obj_type].relationships:
                    self.object_relationships[obj_type].append(related_obj)

        return pd.DataFrame(ocpm_events)

    def analyze_object_interactions(self) -> Dict:
        """Analyze interactions between different object types"""
        interactions = defaultdict(int)

        for obj_type, related_objects in self.object_relationships.items():
            for related_obj in related_objects:
                key = tuple(sorted([obj_type, related_obj]))
                interactions[key] += 1

        return dict(interactions)

    def calculate_object_metrics(self) -> Dict:
        """Calculate comprehensive metrics for each object type"""
        metrics = {}
        ocpm_df = self.convert_to_ocpm_format()

        for obj_type in self.object_types.keys():
            obj_data = ocpm_df[ocpm_df['object_type'] == obj_type]

            metrics[obj_type] = {
                'total_instances': len(obj_data['object_id'].unique()),
                'total_activities': len(obj_data['activity'].unique()),
                'avg_activities_per_instance': len(obj_data) / len(obj_data['object_id'].unique()),
                'top_activities': obj_data['activity'].value_counts().head(3).to_dict(),
                'interaction_count': len(self.object_relationships[obj_type])
            }

        return metrics

    def generate_object_lifecycle_graph(self, object_type: str) -> nx.DiGraph:
        """Generate lifecycle graph for specific object type"""
        ocpm_df = self.convert_to_ocpm_format()
        obj_data = ocpm_df[ocpm_df['object_type'] == object_type]

        G = nx.DiGraph()

        # Add nodes and edges based on activity sequence
        for _, group in obj_data.groupby('object_id'):
            activities = group['activity'].tolist()
            for i in range(len(activities) - 1):
                G.add_edge(activities[i], activities[i + 1])

        return G


class OCPMVisualizer:
    """Handles visualization of OCPM analysis results"""

    @staticmethod
    def create_object_interaction_heatmap(interactions: Dict[Tuple[str, str], int]) -> go.Figure:
        """Create heatmap of object interactions"""
        object_types = sorted(list(set([obj for pair in interactions.keys() for obj in pair])))
        matrix = np.zeros((len(object_types), len(object_types)))

        for (obj1, obj2), count in interactions.items():
            i, j = object_types.index(obj1), object_types.index(obj2)
            matrix[i][j] = count
            matrix[j][i] = count

        return go.Figure(data=go.Heatmap(
            z=matrix,
            x=object_types,
            y=object_types,
            colorscale='Viridis'
        ))

    @staticmethod
    def create_object_metrics_dashboard(metrics: Dict) -> List[go.Figure]:
        """Create dashboard visualizations for object metrics"""
        figures = []

        # Activity Distribution
        activity_counts = {obj: data['total_activities']
                           for obj, data in metrics.items()}

        figures.append(go.Figure(data=[
            go.Bar(x=list(activity_counts.keys()),
                   y=list(activity_counts.values()),
                   name='Activities per Object Type')
        ]))

        # Instance Distribution
        instance_counts = {obj: data['total_instances']
                           for obj, data in metrics.items()}

        figures.append(go.Figure(data=[
            go.Pie(labels=list(instance_counts.keys()),
                   values=list(instance_counts.values()),
                   name='Object Instances Distribution')
        ]))

        return figures

"""Modified create_ocpm_ui to generate OCEL files."""


def create_ocpm_ui():
    """Create Streamlit UI components for OCPM analysis"""
    st.subheader("Object-Centric Process Analytics Analysis")

    # Create output directory if it doesn't exist
    os.makedirs("ocpm_output", exist_ok=True)

    uploaded_file = st.file_uploader("Upload Event Log (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read the CSV file with multiple possible separators
            try:
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                try:
                    df = pd.read_csv(uploaded_file, sep=',')
                except:
                    df = pd.read_csv(uploaded_file, sep='\t')

            # Initialize analyzer
            analyzer = OCPMAnalyzer(df)

            # Generate and save OCEL file
            ocel_path = analyzer.save_ocel()
            st.session_state['ocel_path'] = ocel_path
            st.success(f"OCEL file generated: {ocel_path}")

            # Create tabs for different analyses
            tabs = st.tabs(["Object Interactions", "Object Metrics", "Object Lifecycles"])

            # Object Interactions Tab
            with tabs[0]:
                st.subheader("Object Type Interactions")
                interactions = analyzer.analyze_object_interactions()
                fig = OCPMVisualizer.create_object_interaction_heatmap(interactions)
                st.plotly_chart(fig)

            # Object Metrics Tab
            with tabs[1]:
                st.subheader("Object Type Metrics")
                metrics = analyzer.calculate_object_metrics()
                figures = OCPMVisualizer.create_object_metrics_dashboard(metrics)
                for fig in figures:
                    st.plotly_chart(fig)

            # Object Lifecycles Tab
            with tabs[2]:
                st.subheader("Object Lifecycles")
                selected_object = st.selectbox(
                    "Select Object Type",
                    list(analyzer.object_types.keys())
                )

                if selected_object:
                    lifecycle_graph = analyzer.generate_object_lifecycle_graph(selected_object)
                    dot_graph = nx.nx_pydot.to_pydot(lifecycle_graph)
                    st.graphviz_chart(dot_graph.to_string())

        except Exception as e:
            st.error(f"Error in OCPM analysis: {str(e)}")
            st.write("Available columns in uploaded file:",
                     df.columns.tolist() if 'df' in locals() else 'No data loaded')
            st.write("Please ensure your CSV file has the required columns and data format")