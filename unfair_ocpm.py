import json
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class UnfairOCELAnalyzer:
    """
        Analyzer for identifying unfairness and discrimination in process mining
        Focuses on key objectives:
        1. Bias Detection: Identifying systematic discrimination in process execution
        2. Resource Fairness: Analyzing unfair workload and resource allocation
        3. Time Discrimination: Detecting unfair processing time variations
        4. Case Treatment Bias: Identifying discriminatory case handling
        """

    def __init__(self, json_path: str):
        """Initialize the analyzer with OCEL JSON file"""
        try:
            with open(json_path, 'r') as f:
                self.ocel_data = json.load(f)

            print("Available keys in JSON:", self.ocel_data.keys())
            self.events_df = self._process_events()
            self.objects_df = self._process_objects()
            self.relationships_df = self._process_relationships()

        except Exception as e:
            print(f"Error initializing UnfairOCELAnalyzer: {str(e)}")
            raise


    def _process_events(self):
        """Process events from list format with correct field names"""
        events_list = []
        for event in self.ocel_data['events']:
            # Extract objects from relationships
            object_ids = []
            if 'relationships' in event:
                object_ids = [rel['objectId'] for rel in event['relationships']]

            # Get resource from attributes if present
            resource = None
            if 'attributes' in event:
                for attr in event['attributes']:
                    if attr.get('name') == 'resource':
                        resource = attr.get('value')
                        break

            event_data = {
                'event_id': event.get('id'),
                'timestamp': pd.to_datetime(event.get('time')),
                'event_type': event.get('type'),
                'resource': resource,
                'object_ids': object_ids
            }
            events_list.append(event_data)

        return pd.DataFrame(events_list)

    def _process_objects(self):
        """Process objects from list format with correct field names"""
        objects_list = []
        for obj in self.ocel_data['objects']:
            obj_data = {
                'object_id': obj.get('id'),
                'object_type': obj.get('type')
            }
            # Process attributes
            if 'attributes' in obj:
                for attr in obj['attributes']:
                    name = attr.get('name')
                    value = attr.get('value')
                    obj_data[f'attr_{name}'] = value

            objects_list.append(obj_data)

        return pd.DataFrame(objects_list)

    def _process_relationships(self):
        """Create event-object relationships with qualifiers"""
        relationships_list = []

        for event in self.ocel_data['events']:
            event_id = event.get('id')
            timestamp = pd.to_datetime(event.get('time'))
            event_type = event.get('type')

            # Get resource from attributes
            resource = None
            if 'attributes' in event:
                for attr in event['attributes']:
                    if attr.get('name') == 'resource':
                        resource = attr.get('value')
                        break

            if 'relationships' in event:
                for rel in event['relationships']:
                    relationship = {
                        'event_id': event_id,
                        'object_id': rel['objectId'],
                        'qualifier': rel.get('qualifier'),
                        'timestamp': timestamp,
                        'event_type': event_type,
                        'resource': resource
                    }
                    relationships_list.append(relationship)

        return pd.DataFrame(relationships_list)


    def analyze_process_fairness(self, output_dir: str = 'fairness_analysis'):
        """Analyze and visualize fairness in process execution"""
        Path(output_dir).mkdir(exist_ok=True)

        print("\nUnfair OCEL Analysis Report")
        print("==========================")

        # 1. Resource Discrimination Analysis
        self._analyze_resource_discrimination(output_dir)

        # 2. Processing Time Bias Analysis
        self._analyze_time_bias(output_dir)

        # 3. Case Priority Discrimination
        self._analyze_case_discrimination(output_dir)

        # 4. Handover Bias Analysis
        self._analyze_handover_bias(output_dir)

        self.generate_process_model(output_dir)

    def _analyze_resource_discrimination(self, output_dir: str):
        """Analyze discrimination in resource allocation and workload"""
        print("\n1. Resource Discrimination Analysis:")
        print("-----------------------------------")

        # Calculate workload distribution
        resource_cases = self.relationships_df['resource'].value_counts()
        total_cases = len(self.relationships_df['object_id'].unique())
        expected_cases = total_cases / len(resource_cases)  # Expected fair distribution

        # Calculate discrimination metrics
        discrimination_scores = {}
        for resource, count in resource_cases.items():
            bias_score = (count - expected_cases) / expected_cases
            discrimination_scores[resource] = {
                'cases': count,
                'percentage': (count / total_cases) * 100,
                'bias_score': bias_score,
                'deviation_from_fair': abs(bias_score)
            }
            print(f"\n{resource}:")
            print(f"  Cases: {count} ({(count / total_cases) * 100:.1f}%)")
            print(f"  Bias Score: {bias_score:.3f}")
            print(f"  Deviation from Fair Distribution: {abs(bias_score) * 100:.1f}%")

        # Visualize resource discrimination
        plt.figure(figsize=(12, 6))
        bias_scores = [d['bias_score'] for d in discrimination_scores.values()]
        resources = list(discrimination_scores.keys())

        plt.bar(resources, bias_scores,
                color=['red' if x > 0.2 else 'orange' if x > 0 else 'green' for x in bias_scores])
        plt.xticks(rotation=45, ha='right')
        plt.title('Resource Discrimination Analysis')
        plt.ylabel('Bias Score (>0.2 indicates significant bias)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/resource_discrimination.png')
        plt.close()

    def _analyze_time_bias(self, output_dir: str):
        """Analyze bias in processing times"""
        print("\n2. Processing Time Bias Analysis:")
        print("--------------------------------")

        processing_times = defaultdict(list)
        for obj_id in self.relationships_df['object_id'].unique():
            obj_events = self.relationships_df[self.relationships_df['object_id'] == obj_id]
            if len(obj_events) > 1:
                duration = (obj_events['timestamp'].max() -
                            obj_events['timestamp'].min()).total_seconds() / 3600
                resources = obj_events['resource'].unique()
                for resource in resources:
                    if pd.notna(resource):
                        processing_times[resource].append(duration)

        # Calculate time bias metrics
        overall_mean = np.mean([t for times in processing_times.values() for t in times])
        time_bias_scores = {}

        plt.figure(figsize=(12, 6))
        box_data = []
        labels = []

        for resource, times in processing_times.items():
            if times:
                mean_time = np.mean(times)
                bias_score = (mean_time - overall_mean) / overall_mean
                time_bias_scores[resource] = {
                    'mean_time': mean_time,
                    'bias_score': bias_score,
                    'std_dev': np.std(times)
                }

                print(f"\n{resource}:")
                print(f"  Mean Processing Time: {mean_time:.2f} hours")
                print(f"  Time Bias Score: {bias_score:.3f}")
                print(f"  Standard Deviation: {np.std(times):.2f} hours")

                box_data.append(times)
                labels.append(resource)

        # Visualize time bias
        plt.boxplot(box_data, labels=labels)
        plt.xticks(rotation=45, ha='right')
        plt.title('Processing Time Distribution by Resource')
        plt.ylabel('Processing Time (hours)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_bias.png')
        plt.close()

    def generate_process_model(self, output_dir: str):
        """Generate process model with bias indicators"""
        print("\n5. Process Model with Bias Indicators:")
        print("------------------------------------")

        # Get event types and their sequences
        event_sequences = []
        for obj_id in self.relationships_df['object_id'].unique():
            events = self.relationships_df[
                self.relationships_df['object_id'] == obj_id
                ].sort_values('timestamp')

            if not events.empty:
                sequence = events['event_type'].tolist()
                event_sequences.append(sequence)

        # Calculate bias levels for each activity
        activity_metrics = defaultdict(list)
        for _, row in self.relationships_df.iterrows():
            activity_metrics[row['event_type']].append({
                'processing_time': 0,  # Calculate from timestamps
                'resource': row['resource']
            })

        # Generate Mermaid diagram
        diagram = [
            "flowchart LR",
            "    classDef highBias fill:#ef4444,color:white,stroke:#be123c",
            "    classDef medBias fill:#f97316,color:white,stroke:#c2410c",
            "    classDef lowBias fill:#22c55e,color:white,stroke:#15803d",
            "    classDef process fill:#ffffff,stroke:#666"
        ]

        # Add nodes and edges
        activities = sorted(set(
            event_type for seq in event_sequences for event_type in seq
        ))

        for activity in activities:
            node_id = activity.replace(" ", "_")
            diagram.append(f'    {node_id}["{activity}"]')

        # Add connections
        for seq in event_sequences:
            for i in range(len(seq) - 1):
                from_node = seq[i].replace(" ", "_")
                to_node = seq[i + 1].replace(" ", "_")
                diagram.append(f"    {from_node} --> {to_node}")

        # Apply bias classes
        for activity in activities:
            node_id = activity.replace(" ", "_")
            metrics = activity_metrics[activity]

            # Calculate bias score
            if metrics:
                resources = [m['resource'] for m in metrics]
                most_common_resource = max(set(resources), key=resources.count)
                resource_bias = self._get_resource_bias_score(most_common_resource)

                if abs(resource_bias) > 0.5:
                    diagram.append(f"    class {node_id} highBias")
                elif abs(resource_bias) > 0.25:
                    diagram.append(f"    class {node_id} medBias")
                else:
                    diagram.append(f"    class {node_id} lowBias")

        # Write diagram to file
        mermaid_file = Path(output_dir) / "process_model.mmd"
        mermaid_file.write_text("\n".join(diagram))

        print(f"\nProcess model with bias indicators saved to: {mermaid_file}")
        print("\nColor coding:")
        print("  Red: High bias (>0.5)")
        print("  Orange: Medium bias (0.25-0.5)")
        print("  Green: Low bias (<0.25)")

    def _get_resource_bias_score(self, resource):
        """Calculate bias score for a resource"""
        resource_cases = self.relationships_df['resource'].value_counts()
        total_cases = len(self.relationships_df['object_id'].unique())
        expected_cases = total_cases / len(resource_cases)

        if resource in resource_cases:
            actual_cases = resource_cases[resource]
            return (actual_cases - expected_cases) / expected_cases
        return 0

    def _analyze_case_discrimination(self, output_dir: str):
        """Analyze discrimination in case handling"""
        print("\n3. Case Priority Discrimination Analysis:")
        print("---------------------------------------")

        waiting_times = defaultdict(list)
        for obj_id in self.relationships_df['object_id'].unique():
            obj_events = self.relationships_df[
                self.relationships_df['object_id'] == obj_id
                ].sort_values('timestamp')

            if len(obj_events) > 1:
                obj_type = self.objects_df[
                    self.objects_df['object_id'] == obj_id
                    ]['object_type'].iloc[0]

                timestamps = obj_events['timestamp'].tolist()
                for i in range(len(timestamps) - 1):
                    wait_time = (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
                    waiting_times[obj_type].append(wait_time)

        # Calculate case discrimination metrics
        overall_mean_wait = np.mean([t for times in waiting_times.values() for t in times])

        plt.figure(figsize=(12, 6))
        case_bias_data = []
        case_types = []

        for case_type, times in waiting_times.items():
            if times:
                mean_wait = np.mean(times)
                bias_score = (mean_wait - overall_mean_wait) / overall_mean_wait

                print(f"\n{case_type}:")
                print(f"  Mean Waiting Time: {mean_wait:.2f} hours")
                print(f"  Priority Bias Score: {bias_score:.3f}")
                print(f"  Cases: {len(times)}")

                case_bias_data.append(bias_score)
                case_types.append(case_type)

        # Visualize case discrimination
        plt.bar(case_types, case_bias_data,
                color=['red' if x > 0.2 else 'orange' if x > 0 else 'green' for x in case_bias_data])
        plt.xticks(rotation=45, ha='right')
        plt.title('Case Priority Discrimination Analysis')
        plt.ylabel('Priority Bias Score (>0.2 indicates discrimination)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/case_discrimination.png')
        plt.close()

    def _analyze_handover_bias(self, output_dir: str):
        """Analyze bias in work handover patterns"""
        print("\n4. Handover Bias Analysis:")
        print("-------------------------")

        handovers = defaultdict(lambda: defaultdict(int))
        total_handovers = 0

        for obj_id in self.relationships_df['object_id'].unique():
            obj_events = self.relationships_df[
                self.relationships_df['object_id'] == obj_id
                ].sort_values('timestamp')

            if len(obj_events) > 1:
                resources = obj_events['resource'].tolist()
                for i in range(len(resources) - 1):
                    if pd.notna(resources[i]) and pd.notna(resources[i + 1]):
                        handovers[resources[i]][resources[i + 1]] += 1
                        total_handovers += 1

        # Create handover matrix for visualization
        resources = sorted(set(r for h in handovers.values() for r in h.keys()))
        matrix_data = np.zeros((len(resources), len(resources)))

        for i, from_resource in enumerate(resources):
            for j, to_resource in enumerate(resources):
                count = handovers[from_resource][to_resource]
                matrix_data[i, j] = count / total_handovers if total_handovers > 0 else 0

        # Visualize handover bias
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix_data,
                    xticklabels=resources,
                    yticklabels=resources,
                    cmap='RdYlGn_r',  # Red for high bias, green for low bias
                    annot=True,
                    fmt='.2%')
        plt.title('Handover Pattern Bias Analysis')
        plt.xlabel('To Resource')
        plt.ylabel('From Resource')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/handover_bias.png')
        plt.close()

        # Print significant handover biases
        print("\nSignificant Handover Biases:")
        expected_handover = 1 / (len(resources) * (len(resources) - 1))

        for from_resource in resources:
            for to_resource in resources:
                if from_resource != to_resource:
                    actual = handovers[from_resource][to_resource] / total_handovers if total_handovers > 0 else 0
                    bias = (actual - expected_handover) / expected_handover
                    if abs(bias) > 0.5:  # Show significant biases
                        print(f"\n{from_resource} → {to_resource}:")
                        print(f"  Actual: {actual:.1%}")
                        print(f"  Expected: {expected_handover:.1%}")
                        print(f"  Bias Score: {bias:.2f}")

if __name__ == "__main__":
    try:
        analyzer = UnfairOCELAnalyzer("ocel2-p2p.json")
        analyzer.analyze_process_fairness()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")