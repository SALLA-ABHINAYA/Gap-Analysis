# ocpm_analysis.py
import traceback

import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import json

from openai import AzureOpenAI

from azure_processor import process_log_file

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
        """
        Preprocess event log to ensure consistent format.
        Handles both traditional process mining formats and custom business formats.
        Required columns: case_id (or Case id), timestamp (or Timestamp)
        """
        print("Input DataFrame Info:")
        print(df.info())

        # Create a copy to avoid modifying the original
        df = df.copy()

        # Define all possible column mappings - both traditional and new format
        column_mappings = {
            # Traditional process mining formats
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp',
            'Case_': 'case_id',
            'Activity': 'activity',
            'Timestamp': 'timestamp',
            'case': 'case_id',
            'event': 'activity',
            'start_timestamp': 'timestamp',
            
            # New business format
            'Case id': 'case_id',
            'Activity name': 'activity',
            
            # Already correct format
            'case_id': 'case_id',
            'activity': 'activity',
            'timestamp': 'timestamp',
            
            # Additional business columns
            'Resource': 'resource',
            'User type': 'user_type',
            'Resource name': 'resource_name',
            'Product name': 'product_name',
            'Product description': 'product_description',
            'Order value': 'order_value',
            'Business unit': 'business_unit',
            'Customer name': 'customer_name',
            'Customer payment history': 'payment_history'
        }

        # Try to identify and rename columns
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Special handling for case_id if it has a numeric prefix
        if 'case_id' not in df.columns:
            case_id_columns = [col for col in df.columns if 'case' in col.lower()]
            if case_id_columns:
                df = df.rename(columns={case_id_columns[0]: 'case_id'})

        # Special handling for synthetic data format
        if 'case_id' not in df.columns and 'case:concept:name' in df.columns:
            df['case_id'] = df['case:concept:name']
        if 'activity' not in df.columns and 'concept:name' in df.columns:
            df['activity'] = df['concept:name']
        if 'timestamp' not in df.columns and 'time:timestamp' in df.columns:
            df['timestamp'] = df['time:timestamp']

        # Check required columns
        required_columns = ['case_id', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print("Available columns:", df.columns.tolist())
            raise ValueError(f"Missing required columns: {missing_columns}")

        # If activity column is not present, create it from Activity name or set to default
        if 'activity' not in df.columns:
            if 'Activity name' in df.columns:
                df['activity'] = df['Activity name']
            else:
                df['activity'] = 'Unknown Activity'

        # Try multiple timestamp formats
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            timestamp_formats = [
                '%d/%m/%Y %H:%M',  # New business format
                '%Y-%m-%d %H:%M:%S',  # Traditional format
                '%Y/%m/%d %H:%M:%S',
                '%d-%m-%Y %H:%M:%S'
            ]
            
            for timestamp_format in timestamp_formats:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=timestamp_format)
                    break
                except ValueError:
                    continue
            
            # If none of the specific formats work, try pandas default parser
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Handle numeric fields
        if 'order_value' in df.columns:
            df['order_value'] = pd.to_numeric(df['order_value'], errors='coerce')

        # Set default values for optional columns
        default_values = {
            'activity': 'Unknown Activity',
            'resource': 'Unknown Resource',
            'user_type': 'Unknown',
            'resource_name': 'Unknown',
            'product_name': 'Unknown Product',
            'product_description': 'No Description',
            'business_unit': 'Unknown Unit',
            'customer_name': '',
            'payment_history': 'unknown'
        }

        # Only fill default values for columns that exist
        for col, default_val in default_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)

        print("\nProcessed DataFrame Info:")
        print(df.info())
        return df

    def generate_timing_thresholds(self) -> Dict:
        """
        Generates timing thresholds based on OCEL object model.
        This method should be called after process_log_file has generated output_ocel.json
        """
        try:
            # First load the existing OCEL model
            with open('ocpm_output/output_ocel.json', 'r') as f:
                ocel_model = json.load(f)

            # Initialize OpenAI client
            client = AzureOpenAI(
                api_key="5GLXXNjNjhjRKunOEVm8v7HIk335V4E9myCFNdFvpUmuezUG3hzbJQQJ99BAACYeBjFXJ3w3AAABACOGBfoy",
                api_version="2024-02-01",
                azure_endpoint="https://smartcall.openai.azure.com/"
            )

            # Create prompt for timing threshold generation
            prompt = f"""
            Based on this OCEL object model, generate appropriate timing thresholds for a financial trading system:

            {json.dumps(ocel_model, indent=2)}

            Generate timing thresholds following these rules:
            1. Each object type needs:
               - Total maximum duration for all activities (in hours)
               - Default maximum gap between consecutive activities (in hours)
               - Activity-specific thresholds for processing time and gaps

            2. Consider these factors:
               - Trading activities should have shorter durations (typically < 1 hour)
               - Risk and compliance activities can have longer durations
               - Market data activities should be near real-time
               - Settlement activities can span multiple hours

            Return only valid JSON following this exact structure:
            {{
              "ObjectType": {{
                "total_duration_hours": number,
                "default_gap_hours": number,
                "activity_thresholds": {{
                  "activity_name": {{
                    "max_duration_hours": number,
                    "max_gap_after_hours": number
                  }}
                }}
              }}
            }}
            """

            # Get threshold recommendations from OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in financial trading processes and regulatory compliance. Provide timing thresholds that reflect real-world trading operations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            # Parse and validate the response
            thresholds = json.loads(response.choices[0].message.content)

            # Validate the structure matches our requirements
            for obj_type, config in thresholds.items():
                required_keys = {'total_duration_hours', 'default_gap_hours', 'activity_thresholds'}
                if not all(key in config for key in required_keys):
                    raise ValueError(f"Missing required keys in threshold config for {obj_type}")

            # Save the thresholds
            os.makedirs('ocpm_output', exist_ok=True)
            with open('ocpm_output/output_ocel_threshold.json', 'w') as f:
                json.dump(thresholds, f, indent=2)

            return thresholds

        except Exception as e:
            print(f"Error generating timing thresholds: {str(e)}")
            print(traceback.format_exc())
            raise


    def _initialize_object_types(self, use_azure: bool = True) -> Dict[str, ObjectType]:
        """Initialize object types, optionally using Azure OpenAI"""
        try:
            if use_azure:
                # Process directly with Azure
                azure_types = process_log_file(df=self.event_log,chunk_size=1000)

                # Generate timing thresholds based on the object model
                self.generate_timing_thresholds()
                
                return azure_types
                    
            return self._get_default_object_types()
            
        except Exception as e:
            print(f"Error in object type initialization: {str(e)}")
            return self._get_default_object_types()

    def _get_default_object_types(self) -> Dict[str, ObjectType]:
        """Return default object types"""
        print("Using default object types")
        return {
            'Trade': ObjectType(
                name='Trade',
                activities=[
                    'Trade Initiated', 'Trade Executed', 'Trade Allocated',
                    'Trade Settled', 'Trade Canceled'
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

        default_object_type = next(iter(self.object_types.keys())) if self.object_types else None
        if not default_object_type:
            raise ValueError("No object types available for conversion")

        print(f"Using default object type: {default_object_type}")

        for _, row in self.event_log.iterrows():
            activity = row['activity']
            related_objects = self.activity_object_mapping.get(activity, [default_object_type])

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

            unique_objects = len(obj_data['object_id'].unique())
            avg_activities = len(obj_data) / unique_objects if unique_objects > 0 else 1

            metrics[obj_type] = {
                'total_instances': len(obj_data['object_id'].unique()),
                'total_activities': len(obj_data['activity'].unique()),
                'avg_activities_per_instance': avg_activities,
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


def read_csv_with_detection(file):
    """Read CSV file with automatic separator detection"""
    # First, let's peek at the file content
    try:
        # Read first few lines to detect the format
        file_content = file.read(1024).decode('utf-8')
        file.seek(0)  # Reset file pointer
        
        # Print first few lines for debugging
        print("First few lines of file:")
        print(file_content[:200])
        
        # Count potential separators
        separators = {
            ',': file_content.count(','),
            ';': file_content.count(';'),
            '\t': file_content.count('\t'),
            '|': file_content.count('|')
        }
        
        print("Detected separator counts:", separators)
        
        # Try separators in order of frequency
        for sep, count in sorted(separators.items(), key=lambda x: x[1], reverse=True):
            try:
                print(f"Trying separator: '{sep}'")
                df = pd.read_csv(file, sep=sep)
                print(f"Successfully read with separator: '{sep}'")
                print("Columns found:", df.columns.tolist())
                return df
            except Exception as e:
                print(f"Failed with separator '{sep}': {str(e)}")
                file.seek(0)  # Reset file pointer for next attempt
        
        # If no separator worked, try pandas' automatic detection
        print("Trying pandas automatic detection...")
        file.seek(0)
        df = pd.read_csv(file, engine='python')
        print("Successfully read with automatic detection")
        print("Columns found:", df.columns.tolist())
        return df
        
    except Exception as e:
        raise Exception(f"Failed to read CSV file: {str(e)}")


def create_ocpm_ui():
    """Create Streamlit UI components for OCPM analysis"""
    st.subheader("Object-Centric Process Analytics Analysis")

    # Create output directory if it doesn't exist
    os.makedirs("ocpm_output", exist_ok=True)

    uploaded_file = st.file_uploader("Upload Event Log (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            df = read_csv_with_detection(uploaded_file)

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