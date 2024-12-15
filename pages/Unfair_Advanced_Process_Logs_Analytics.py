import streamlit as st
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

    def __init__(self, ocel_data):
        """Initialize analyzer with OCEL data"""
        self.ocel_data = ocel_data
        self.events_df = self._process_events()
        self.objects_df = self._process_objects()
        self.relationships_df = self._process_relationships()


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

    def get_analysis_plots(self):
        """Generate all analysis plots and return them with metrics"""
        plots = {}
        metrics = {}

        # 1. Resource Discrimination Analysis
        fig_resource, resource_metrics = self._generate_resource_plot()
        plots['resource_discrimination'] = fig_resource
        metrics['resource'] = resource_metrics

        # 2. Time Bias Analysis
        fig_time, time_metrics = self._generate_time_plot()
        plots['time_bias'] = fig_time
        metrics['time'] = time_metrics

        # 3. Case Priority Analysis
        fig_case, case_metrics = self._generate_case_plot()
        plots['case_priority'] = fig_case
        metrics['case'] = case_metrics

        # 4. Handover Analysis
        fig_handover, handover_metrics = self._generate_handover_plot()
        plots['handover'] = fig_handover
        metrics['handover'] = handover_metrics

        return plots, metrics

    def _generate_resource_plot(self):
        """Generate resource discrimination plot and metrics"""
        fig, ax = plt.subplots(figsize=(12, 6))
        resource_cases = self.relationships_df['resource'].value_counts()
        total_cases = len(self.relationships_df['object_id'].unique())
        expected_cases = total_cases / len(resource_cases)

        bias_scores = []
        resources = []
        metrics = {}

        for resource, count in resource_cases.items():
            bias_score = (count - expected_cases) / expected_cases
            bias_scores.append(bias_score)
            resources.append(resource)
            metrics[resource] = {
                'cases': int(count),
                'bias_score': float(bias_score),
                'percentage': float((count / total_cases) * 100)
            }

        ax.bar(resources, bias_scores,
               color=['red' if x > 0.2 else 'orange' if x > 0 else 'green' for x in bias_scores])
        ax.set_xticklabels(resources, rotation=45, ha='right')
        ax.set_title('Resource Discrimination Analysis')
        ax.set_ylabel('Bias Score (>0.2 indicates significant bias)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig, metrics

    def _generate_time_plot(self):
        """Generate time bias plot and metrics"""
        processing_times = defaultdict(list)
        metrics = {}

        for obj_id in self.relationships_df['object_id'].unique():
            obj_events = self.relationships_df[self.relationships_df['object_id'] == obj_id]
            if len(obj_events) > 1:
                duration = (obj_events['timestamp'].max() -
                            obj_events['timestamp'].min()).total_seconds() / 3600
                resources = obj_events['resource'].unique()
                for resource in resources:
                    if pd.notna(resource):
                        processing_times[resource].append(duration)

        fig, ax = plt.subplots(figsize=(12, 6))
        box_data = [times for times in processing_times.values()]

        ax.boxplot(box_data, labels=processing_times.keys())
        ax.set_xticklabels(processing_times.keys(), rotation=45, ha='right')
        ax.set_title('Processing Time Distribution by Resource')
        ax.set_ylabel('Processing Time (hours)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Calculate metrics
        for resource, times in processing_times.items():
            metrics[resource] = {
                'mean_time': float(np.mean(times)),
                'median_time': float(np.median(times)),
                'std_dev': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times))
            }

        return fig, metrics

    def _generate_case_plot(self):
        """Generate case priority plot and metrics"""
        waiting_times = defaultdict(list)
        metrics = {}

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

        # Calculate metrics and prepare plot
        fig, ax = plt.subplots(figsize=(12, 6))
        overall_mean = np.mean([t for times in waiting_times.values() for t in times])

        bias_scores = []
        types = []

        for case_type, times in waiting_times.items():
            if times:
                mean_time = np.mean(times)
                bias_score = (mean_time - overall_mean) / overall_mean
                bias_scores.append(bias_score)
                types.append(case_type)

                metrics[case_type] = {
                    'mean_wait': float(mean_time),
                    'bias_score': float(bias_score),
                    'std_dev': float(np.std(times)),
                    'case_count': len(times)
                }

        ax.bar(types, bias_scores,
               color=['red' if x > 0.2 else 'orange' if x > 0 else 'green' for x in bias_scores])
        ax.set_xticklabels(types, rotation=45, ha='right')
        ax.set_title('Case Priority Discrimination Analysis')
        ax.set_ylabel('Priority Bias Score (>0.2 indicates discrimination)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig, metrics

    def _generate_handover_plot(self):
        """Generate handover pattern plot and metrics"""
        handovers = defaultdict(lambda: defaultdict(int))
        total_handovers = 0
        metrics = {}

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

        # Create handover matrix
        resources = sorted(set(r for h in handovers.values() for r in h.keys()))
        matrix_data = np.zeros((len(resources), len(resources)))

        for i, from_resource in enumerate(resources):
            for j, to_resource in enumerate(resources):
                count = handovers[from_resource][to_resource]
                matrix_data[i, j] = count / total_handovers if total_handovers > 0 else 0

                if count > 0:
                    metrics[f"{from_resource}->{to_resource}"] = {
                        'count': int(count),
                        'percentage': float((count / total_handovers) * 100)
                    }

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(matrix_data,
                    xticklabels=resources,
                    yticklabels=resources,
                    cmap='RdYlGn_r',
                    annot=True,
                    fmt='.2%',
                    ax=ax)
        plt.title('Handover Pattern Bias Analysis')
        plt.xlabel('To Resource')
        plt.ylabel('From Resource')
        plt.tight_layout()

        return fig, metrics


def main():
    st.set_page_config(layout="wide")

    st.title("🔍 Unfair Advanced Process Logs Analytics")

    # Introduction
    st.markdown("""
    ### What is Unfair Process Analytics?
    Unfair process analysis analyzes business processes to identify potential biases and discrimination patterns in:
    - Resource allocation
    - Processing times
    - Case priorities
    - Work handover patterns

    Upload your Advanced Process Logs JSON file to begin the analysis.
    """)

    uploaded_file = st.file_uploader("Choose an Advanced Process Logs JSON file", type=['json'])

    if uploaded_file is not None:
        try:
            with st.spinner('Analyzing process data...'):
                # Load and analyze data
                ocel_data = json.load(uploaded_file)
                analyzer = UnfairOCELAnalyzer(ocel_data)
                plots, metrics = analyzer.get_analysis_plots()

                # Display Analysis Dashboard
                st.success("Analysis complete! Exploring potential unfairness in your process...")

                # Tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Resource Discrimination",
                    "Time Bias",
                    "Case Priority",
                    "Handover Patterns"
                ])

                with tab1:
                    st.subheader("Resource Discrimination Analysis")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.pyplot(plots['resource_discrimination'])
                    with col2:
                        st.markdown("""
                        #### Key Findings:
                        - Red bars indicate significant bias (>0.2)
                        - Green bars show fair distribution
                        """)
                        for resource, data in metrics['resource'].items():
                            if data['bias_score'] > 0.2:
                                st.warning(f"⚠️ {resource}: {data['bias_score']:.2f} bias score")

                with tab2:
                    st.subheader("Processing Time Bias Analysis")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.pyplot(plots['time_bias'])
                    with col2:
                        st.markdown("#### Processing Time Statistics")
                        for resource, data in metrics['time'].items():
                            with st.expander(f"{resource} Details"):
                                st.write(f"Mean time: {data['mean_time']:.2f} hours")
                                st.write(f"Std Dev: {data['std_dev']:.2f} hours")

                with tab3:
                    st.subheader("Case Priority Analysis")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.pyplot(plots['case_priority'])
                    with col2:
                        st.markdown("#### Priority Bias Details")
                        for case_type, data in metrics['case'].items():
                            if abs(data['bias_score']) > 0.2:
                                status = "👎 Potential discrimination" if data['bias_score'] > 0 else "👍 Fair treatment"
                                st.info(f"{case_type}: {status}")

                with tab4:
                    st.subheader("Handover Pattern Analysis")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.pyplot(plots['handover'])
                    with col2:
                        st.markdown("#### Significant Handover Patterns")
                        for pattern, data in metrics['handover'].items():
                            if data['percentage'] > 10:
                                st.write(f"🔄 {pattern}: {data['percentage']:.1f}%")

                # Download Report
                if st.button("📊 Generate Detailed Report"):
                    report = generate_detailed_report(metrics)
                    st.download_button(
                        "📥 Download Report",
                        report,
                        "unfair_process_analysis_report.txt",
                        "text/plain"
                    )

        except Exception as e:
            st.error(f"Error analyzing the file: {str(e)}")
            st.info("Please ensure your file follows the Advanced Process Logs JSON format.")


def generate_detailed_report(metrics):
    """Generate a comprehensive analysis report"""
    report = []
    report.append("Unfair Process Analytics Analysis Report")
    report.append("=====================================")

    # Add sections for each analysis type
    for analysis_type, data in metrics.items():
        report.append(f"\n{analysis_type.upper()} ANALYSIS")
        report.append("-" * len(f"{analysis_type.upper()} ANALYSIS"))

        for key, values in data.items():
            report.append(f"\n{key}:")
            for metric, value in values.items():
                report.append(f"  {metric}: {value}")

    return "\n".join(report)


if __name__ == "__main__":
    main()
