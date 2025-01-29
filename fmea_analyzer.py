import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Set
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OCELFailureMode:
    """Represents a failure mode in OCEL analysis"""
    id: str
    activity: str
    object_type: str
    description: str
    severity: int = 0
    likelihood: int = 0
    detectability: int = 0
    rpn: int = 0
    related_objects: List[str] = None
    control_points: List[str] = None
    effects: List[str] = None
    causes: List[str] = None


class OCELEnhancedFMEA:
    """FMEA analyzer for OCEL process mining"""

    def __init__(self, ocel_path: str):
        """Initialize with OCEL data path"""
        self.ocel_data = self._load_ocel(ocel_path)
        self.events_df = self._process_events()
        self.failure_modes = []
        self.activity_stats = {}
        self._calculate_activity_stats()

    def _load_ocel(self, path: str) -> Dict:
        """Load and validate OCEL file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if 'ocel:events' not in data:
                raise ValueError("Invalid OCEL format: missing events")
            return data
        except Exception as e:
            logger.error(f"Error loading OCEL file: {str(e)}")
            raise

    def _process_events(self) -> pd.DataFrame:
        """Process OCEL events into DataFrame"""
        events_data = []
        for event in self.ocel_data['ocel:events']:
            event_objects = event.get('ocel:objects', [])
            event_data = {
                'event_id': event['ocel:id'],
                'timestamp': pd.to_datetime(event['ocel:timestamp']),
                'activity': event['ocel:activity'],
                'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                'object_types': [obj.get('type', 'Unknown') for obj in event_objects],
                'objects': [obj['id'] for obj in event_objects]
            }
            events_data.append(event_data)
        return pd.DataFrame(events_data)

    def _calculate_activity_stats(self):
        """Calculate activity statistics for FMEA metrics"""
        for activity in self.events_df['activity'].unique():
            activity_events = self.events_df[self.events_df['activity'] == activity]

            # Calculate completion rate
            total_cases = activity_events['case_id'].nunique()
            completed_cases = activity_events[
                activity_events.groupby('case_id')['timestamp'].transform('count') > 0
                ]['case_id'].nunique()
            completion_rate = completed_cases / total_cases if total_cases > 0 else 0

            # Calculate object type consistency
            expected_types = self._get_expected_object_types(activity)
            actual_types = set().union(*[set(types) for types in activity_events['object_types']])
            type_consistency = len(expected_types.intersection(actual_types)) / len(expected_types) \
                if expected_types else 1

            # Calculate resource variability
            resource_count = activity_events['resource'].nunique()

            self.activity_stats[activity] = {
                'completion_rate': completion_rate,
                'type_consistency': type_consistency,
                'resource_count': resource_count,
                'frequency': len(activity_events),
                'avg_objects': activity_events['objects'].apply(len).mean()
            }

    def _get_expected_object_types(self, activity: str) -> Set[str]:
        """Get expected object types for activity"""
        # Define based on activity patterns
        patterns = {
            'Trade': {'Trade', 'Market', 'Risk'},
            'Market': {'Market', 'Price'},
            'Risk': {'Risk', 'Trade'},
            'Settlement': {'Trade', 'Settlement'}
        }
        for pattern, types in patterns.items():
            if pattern in activity:
                return types
        return set()

    def calculate_severity(self, failure_mode: OCELFailureMode) -> int:
        """Calculate severity (1-10) based on potential impact"""
        severity = 1
        activity_stats = self.activity_stats.get(failure_mode.activity, {})

        # Impact on process completion
        if activity_stats.get('completion_rate', 1) < 0.8:
            severity += 3

        # Impact on object relationships
        if len(failure_mode.related_objects or []) > 2:
            severity += 2

        # Impact based on activity importance
        if any(critical in failure_mode.activity.lower()
               for critical in ['trade', 'risk', 'settlement']):
            severity += 2

        # Impact on downstream activities
        if failure_mode.effects and len(failure_mode.effects) > 2:
            severity += 2

        return min(severity, 10)

    def calculate_likelihood(self, failure_mode: OCELFailureMode) -> int:
        """Calculate likelihood (1-10) of failure occurrence"""
        likelihood = 1
        activity_stats = self.activity_stats.get(failure_mode.activity, {})

        # Historical failure rate
        type_consistency = activity_stats.get('type_consistency', 1)
        if type_consistency < 0.9:
            likelihood += 3
        elif type_consistency < 0.95:
            likelihood += 2

        # Complexity-based likelihood
        avg_objects = activity_stats.get('avg_objects', 0)
        if avg_objects > 5:
            likelihood += 2
        elif avg_objects > 3:
            likelihood += 1

        # Resource-based likelihood
        resource_count = activity_stats.get('resource_count', 1)
        if resource_count > 3:
            likelihood += 2

        # Frequency-based likelihood
        frequency = activity_stats.get('frequency', 0)
        if frequency > 100:
            likelihood += 1

        return min(likelihood, 10)

    def calculate_detectability(self, failure_mode: OCELFailureMode) -> int:
        """Calculate detectability (1-10), where 1 is easily detectable"""
        detectability = 10  # Start with worst case
        activity_stats = self.activity_stats.get(failure_mode.activity, {})

        # Control points reduce detectability score (improve detection)
        if failure_mode.control_points:
            detectability -= len(failure_mode.control_points)

        # Frequent activities are easier to monitor
        if activity_stats.get('frequency', 0) > 50:
            detectability -= 2

        # Object type consistency makes issues more detectable
        if activity_stats.get('type_consistency', 0) > 0.9:
            detectability -= 2

        # Resource monitoring improves detectability
        if activity_stats.get('resource_count', 0) < 3:
            detectability -= 1

        return max(detectability, 1)  # Ensure minimum of 1

    def analyze_failure_modes(self) -> List[Dict]:
        """Perform comprehensive FMEA analysis"""
        results = []

        # Identify potential failure modes
        self._identify_failure_modes()

        # Calculate RPN for each failure mode
        for failure_mode in self.failure_modes:
            # Calculate core FMEA metrics
            severity = self.calculate_severity(failure_mode)
            likelihood = self.calculate_likelihood(failure_mode)
            detectability = self.calculate_detectability(failure_mode)

            # Calculate RPN
            rpn = severity * likelihood * detectability

            results.append({
                'id': failure_mode.id,
                'activity': failure_mode.activity,
                'object_type': failure_mode.object_type,
                'description': failure_mode.description,
                'severity': severity,
                'likelihood': likelihood,
                'detectability': detectability,
                'rpn': rpn,
                'effects': failure_mode.effects,
                'causes': failure_mode.causes,
                'recommendations': self._generate_recommendations(failure_mode, rpn)
            })

        return sorted(results, key=lambda x: x['rpn'], reverse=True)

    def _identify_failure_modes(self):
        """Identify potential failure modes"""
        for activity, stats in self.activity_stats.items():
            # Missing object types
            expected_types = self._get_expected_object_types(activity)
            if expected_types and stats['type_consistency'] < 1:
                self.failure_modes.append(OCELFailureMode(
                    id=f"FM_{len(self.failure_modes)}",
                    activity=activity,
                    object_type="Object Type",
                    description=f"Missing required object types in {activity}",
                    effects=["Incomplete process data", "Compliance issues"],
                    causes=["Data validation failure", "Process deviation"],
                    control_points=["Pre-execution validation"]
                ))

            # Resource overallocation
            if stats['resource_count'] > 3:
                self.failure_modes.append(OCELFailureMode(
                    id=f"FM_{len(self.failure_modes)}",
                    activity=activity,
                    object_type="Resource",
                    description=f"Excessive resource variation in {activity}",
                    effects=["Inconsistent execution", "Quality issues"],
                    causes=["Resource management issues", "Training gaps"],
                    control_points=["Resource monitoring"]
                ))

            # Low completion rate
            if stats['completion_rate'] < 0.9:
                self.failure_modes.append(OCELFailureMode(
                    id=f"FM_{len(self.failure_modes)}",
                    activity=activity,
                    object_type="Completion",
                    description=f"Low completion rate in {activity}",
                    effects=["Process delays", "Customer impact"],
                    causes=["Process bottlenecks", "Resource constraints"],
                    control_points=["Completion monitoring"]
                ))

    def _generate_recommendations(self, failure_mode: OCELFailureMode, rpn: int) -> List[str]:
        """Generate recommendations based on failure mode and RPN"""
        recommendations = []

        # RPN-based recommendations
        if rpn > 200:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED",
                "Implement emergency controls",
                "Schedule urgent review"
            ])
        elif rpn > 100:
            recommendations.extend([
                "HIGH PRIORITY",
                "Review process controls",
                "Enhance monitoring"
            ])
        else:
            recommendations.extend([
                "MONITOR",
                "Regular review required",
                "Document findings"
            ])

        # Object type specific recommendations
        if failure_mode.object_type == "Object Type":
            recommendations.extend([
                "Implement strict type validation",
                "Add pre-execution checks",
                "Update process documentation"
            ])
        elif failure_mode.object_type == "Resource":
            recommendations.extend([
                "Review resource allocation",
                "Implement capacity planning",
                "Consider automation options"
            ])
        elif failure_mode.object_type == "Completion":
            recommendations.extend([
                "Analyze process bottlenecks",
                "Implement progress tracking",
                "Review resource availability"
            ])

        return recommendations