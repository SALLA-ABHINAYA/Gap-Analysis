import json
import pandas as pd
import numpy as np
from collections import defaultdict


class ProcessFairnessAnalyzer:
    def __init__(self, json_path: str):
        """Initialize the analyzer with OCEL JSON file"""
        try:
            # Load JSON file
            with open(json_path, 'r') as f:
                self.ocel_data = json.load(f)

            print("Available keys in JSON:", self.ocel_data.keys())

            # Debug: Print sample raw data
            print("\nSample raw event:")
            print(self.ocel_data['events'][0])
            print("\nSample raw object:")
            print(self.ocel_data['objects'][0])

            # Process core data
            self.events_df = self._process_events()
            print("\nSample processed events:")
            print(self.events_df.head())

            self.objects_df = self._process_objects()
            print("\nSample processed objects:")
            print(self.objects_df.head())

            self.relationships_df = self._process_relationships()
            print("\nSample relationships:")
            print(self.relationships_df.head() if not self.relationships_df.empty else "No relationships found")

        except Exception as e:
            print(f"Error initializing ProcessFairnessAnalyzer: {str(e)}")
            raise

    def _process_events(self):
        """Process events from list format with correct field names"""
        events_list = []
        for event in self.ocel_data['events']:
            # Extract objects from relationships
            object_ids = [rel['objectId'] for rel in event.get('relationships', [])]

            # Get resource from attributes if present
            resource = None
            for attr in event.get('attributes', []):
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
            # Process latest attribute values
            attr_dict = {}
            for attr in obj.get('attributes', []):
                name = attr.get('name')
                value = attr.get('value')
                time = pd.to_datetime(attr.get('time'))

                if name not in attr_dict or time > attr_dict[name]['time']:
                    attr_dict[name] = {'value': value, 'time': time}

            # Add latest attribute values to object data
            for name, info in attr_dict.items():
                obj_data[f'attr_{name}'] = info['value']

            objects_list.append(obj_data)

        return pd.DataFrame(objects_list)

    def _process_relationships(self):
        """Create event-object relationships with qualifiers"""
        relationships_list = []

        for _, event in self.events_df.iterrows():
            event_relationships = next((e['relationships'] for e in self.ocel_data['events']
                                        if e['id'] == event['event_id']), [])

            for rel in event_relationships:
                relationship = {
                    'event_id': event['event_id'],
                    'object_id': rel['objectId'],
                    'qualifier': rel['qualifier'],
                    'timestamp': event['timestamp'],
                    'event_type': event['event_type'],
                    'resource': event['resource']
                }
                relationships_list.append(relationship)

        return pd.DataFrame(relationships_list)

    def analyze_fairness(self):
        """Analyze basic process metrics"""
        processing_times = defaultdict(list)
        resource_distribution = defaultdict(int)

        # Get object types mapping
        obj_types_dict = dict(zip(self.objects_df['object_id'],
                                  self.objects_df['object_type']))

        # Calculate processing times per object and resource distribution
        for obj_id in self.objects_df['object_id'].unique():
            obj_events = self.relationships_df[
                self.relationships_df['object_id'] == obj_id
                ]

            if len(obj_events) > 1:
                duration = (obj_events['timestamp'].max() -
                            obj_events['timestamp'].min()).total_seconds() / 3600

                obj_type = obj_types_dict.get(obj_id)
                if obj_type:
                    processing_times[obj_type].append(duration)

                # Count resource frequencies
                for resource in obj_events['resource'].dropna().unique():
                    resource_distribution[resource] += 1

        return {
            'processing_times': dict(processing_times),
            'resource_distribution': dict(resource_distribution)
        }


if __name__ == "__main__":
    try:
        # Create analyzer instance
        analyzer = ProcessFairnessAnalyzer("ocel2-p2p.json")

        # Get metrics
        metrics = analyzer.analyze_fairness()

        print("\nProcess Analysis Results:")
        print("========================")

        print("\nProcessing Times by Object Type:")
        for obj_type, times in metrics['processing_times'].items():
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                print(f"\n{obj_type}:")
                print(f"  Average processing time: {avg_time:.2f} hours")
                print(f"  Std Dev: {std_time:.2f} hours")
                print(f"  Number of cases: {len(times)}")

        print("\nResource Distribution:")
        for resource, count in sorted(
                metrics['resource_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
        )[:5]:
            print(f"  {resource}: {count} cases")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")