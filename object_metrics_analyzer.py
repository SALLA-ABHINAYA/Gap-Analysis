import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import json
import streamlit as st


class DetailedOCELAnalyzer:
    """Advanced OCEL log analyzer with detailed metrics"""

    def __init__(self, ocel_path: str):
        """Initialize with OCEL file path"""
        self.ocel_data = self._load_ocel(ocel_path)
        self.events = pd.DataFrame(self.ocel_data['ocel:events'])
        self.object_types = self.ocel_data['ocel:object-types']

    def _load_ocel(self, path: str) -> Dict:
        """Load and validate OCEL JSON"""
        with open(path, 'r') as f:
            data = json.load(f)

        required_keys = ['ocel:events', 'ocel:object-types']
        if not all(key in data for key in required_keys):
            raise ValueError("Invalid OCEL format")

        return data

    def analyze_interaction_patterns(self) -> Tuple[go.Figure, Dict]:
        """Analyze detailed interaction patterns between object types"""
        # Initialize interaction matrix
        matrix = np.zeros((len(self.object_types), len(self.object_types)))

        # Track detailed patterns
        patterns = {
            (t1, t2): {
                'count': 0,
                'examples': [],
                'timing': []
            }
            for t1 in self.object_types
            for t2 in self.object_types
        }

        # Process events
        for _, event in self.events.iterrows():
            objects = event.get('ocel:objects', [])
            if not objects:
                continue

            # Analyze object interactions
            types = [obj['type'] for obj in objects]
            for t1 in types:
                for t2 in types:
                    if t1 != t2:
                        i = self.object_types.index(t1)
                        j = self.object_types.index(t2)
                        matrix[i][j] += 1

                        # Record pattern details
                        patterns[(t1, t2)]['count'] += 1
                        patterns[(t1, t2)]['examples'].append({
                            'event_id': event['ocel:id'],
                            'timestamp': event['ocel:timestamp'],
                            'activity': event['ocel:activity']
                        })

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=self.object_types,
            y=self.object_types,
            colorscale='YlOrRd',
            colorbar=dict(title='Interaction Count')
        ))

        fig.update_layout(
            title='Detailed Object Type Interactions',
            xaxis_title='Target Object Type',
            yaxis_title='Source Object Type'
        )

        return fig, patterns

    def analyze_object_lifecycle(self) -> Dict:
        """Analyze object lifecycle patterns"""
        lifecycles = {t: [] for t in self.object_types}

        for _, event in self.events.iterrows():
            objects = event.get('ocel:objects', [])
            for obj in objects:
                lifecycles[obj['type']].append({
                    'timestamp': event['ocel:timestamp'],
                    'activity': event['ocel:activity'],
                    'attributes': event.get('ocel:attributes', {})
                })

        return lifecycles

    def generate_insights(self) -> List[str]:
        """Generate detailed process insights"""
        insights = []

        # Analyze event patterns
        activity_counts = self.events['ocel:activity'].value_counts()
        most_common = activity_counts.index[0]
        insights.append(f"Most frequent activity: {most_common}")

        # Analyze timing patterns
        self.events['ocel:timestamp'] = pd.to_datetime(self.events['ocel:timestamp'])
        time_diffs = self.events['ocel:timestamp'].diff()
        avg_time = time_diffs.mean()
        insights.append(f"Average time between events: {avg_time}")

        # Analyze object patterns
        obj_counts = {t: 0 for t in self.object_types}
        for _, event in self.events.iterrows():
            for obj in event.get('ocel:objects', []):
                obj_counts[obj['type']] += 1

        dominant_type = max(obj_counts.items(), key=lambda x: x[1])[0]
        insights.append(f"Dominant object type: {dominant_type}")

        return insights

    def display_analysis(self):
        """Display comprehensive analysis in Streamlit"""
        st.title("Detailed OCEL Analysis")

        # Display interaction patterns
        st.header("Object Interaction Patterns")
        fig, patterns = self.analyze_interaction_patterns()
        st.plotly_chart(fig)

        # Display example patterns
        st.subheader("Example Interaction Patterns")
        for (t1, t2), data in patterns.items():
            if data['count'] > 0:
                st.write(f"{t1} → {t2}:")
                st.write(f"Count: {data['count']}")
                if data['examples']:
                    st.write("Example event:", data['examples'][0])

        # Display lifecycle analysis
        st.header("Object Lifecycles")
        lifecycles = self.analyze_object_lifecycle()
        for obj_type, events in lifecycles.items():
            if events:
                st.subheader(f"{obj_type} Lifecycle")
                st.write(f"Total events: {len(events)}")
                st.write("Sample activities:",
                         [e['activity'] for e in events[:3]])

        # Display insights
        st.header("Process Insights")
        for insight in self.generate_insights():
            st.write(f"• {insight}")


# Usage
if __name__ == "__main__":
    analyzer = DetailedOCELAnalyzer("ocpm_output/process_data.json")
    analyzer.display_analysis()