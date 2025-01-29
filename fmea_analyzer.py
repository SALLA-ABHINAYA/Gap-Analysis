import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from scipy import stats  # Add this import at the top
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
import logging
from typing import Dict, List
import traceback

logger = logging.getLogger(__name__)



@dataclass
class FailureMode:
    """Represents a failure mode in OCEL process"""
    id: str
    activity: str
    description: str
    severity: int
    occurrence: int
    detection: int
    rpn: int
    object_types: List[str]
    effects: List[str]
    causes: List[str]
    controls: List[str]
    recommendations: List[str]


class OCELFMEAAnalyzer:
    """FMEA analyzer for OCEL process mining logs"""

    def __init__(self, ocel_path: str):
        self.ocel_data = self._load_ocel(ocel_path)
        self.events_df = self._process_events()
        self.failure_modes = []
        self.metrics = {}

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
        """Convert OCEL events to DataFrame for analysis"""
        try:
            events = []
            for event in self.ocel_data['ocel:events']:
                event_data = {
                    'event_id': event['ocel:id'],
                    'timestamp': pd.to_datetime(event['ocel:timestamp']),
                    'activity': event['ocel:activity'],
                    'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                    'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                    'object_type': event.get('ocel:attributes', {}).get('object_type', 'Unknown'),
                    'objects': [obj['id'] for obj in event.get('ocel:objects', [])]
                }
                events.append(event_data)

            df = pd.DataFrame(events)
            logger.info(f"Processed {len(df)} events with columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            logger.error(f"Error processing events: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _add_sequence_failure_modes(self, case_sequences: pd.Series) -> None:
        """Add failure modes based on sequence analysis"""
        try:
            # Define expected activity sequences
            expected_sequences = {
                'Trade': ['Trade Initiated', 'Market Data Validation', 'Quote Provided'],
                'Market': ['Quote Requested', 'Market Making', 'Quote Provided'],
                'Risk': ['Risk Assessment', 'Risk Validation', 'Risk Report']
            }

            # Analyze each case
            for case_id, sequence in case_sequences.items():
                try:
                    case_data = self.events_df[self.events_df['case_id'] == case_id]

                    # Get unique object types, handling potential missing values
                    object_types = case_data['object_type'].unique() if 'object_type' in case_data.columns else [
                        'Unknown']

                    for obj_type, expected_sequence in expected_sequences.items():
                        if obj_type in object_types:
                            actual_sequence = [act for act in sequence if act in expected_sequence]
                            if actual_sequence != expected_sequence:
                                failure_mode = {
                                    'id': f"FM_SEQ_{len(self.failure_modes) + 1}",
                                    'activity': sequence[0],
                                    'description': f"Sequence violation in {obj_type} process",
                                    'severity': 'High',
                                    'occurrence': 1,
                                    'detection': 3,
                                    'rpn': 3,  # Severity * Detection
                                    'object_types': [obj_type],
                                    'effects': ['Process integrity compromised', 'Potential regulatory issues'],
                                    'causes': ['Activity order violation', 'Process bypass'],
                                    'controls': ['Sequence validation', 'Process monitoring'],
                                    'recommendations': [
                                        'Implement strict activity sequencing',
                                        'Add validation controls',
                                        'Monitor sequence compliance'
                                    ]
                                }
                                self.failure_modes.append(failure_mode)
                                logger.info(
                                    f"Added sequence failure mode for case {case_id}: {failure_mode['description']}")

                except Exception as case_error:
                    logger.warning(f"Error processing case {case_id}: {str(case_error)}")
                    continue

        except Exception as e:
            logger.error(f"Error in sequence failure modes analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def identify_failure_modes(self):
        """Identify potential failure modes in the process"""
        try:
            # Analyze activity sequences
            case_sequences = self.events_df.groupby('case_id')['activity'].agg(list)

            # Analyze timing patterns
            timing_patterns = self._analyze_timing_patterns()

            # Analyze resource patterns
            resource_patterns = self._analyze_resource_patterns()

            # Generate failure modes
            self._add_sequence_failure_modes(case_sequences)
            self._add_timing_failure_modes(timing_patterns)
            self._add_resource_failure_modes(resource_patterns)

            logger.info(f"Identified {len(self.failure_modes)} failure modes")

        except Exception as e:
            logger.error(f"Error in failure modes identification: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _analyze_timing_patterns(self) -> Dict:
        """Analyze timing patterns for anomaly detection"""
        timing_metrics = {}

        try:
            # Calculate activity durations
            activity_durations = self.events_df.groupby(['case_id', 'activity']).agg({
                'timestamp': lambda x: (x.max() - x.min()).total_seconds()
            }).reset_index()

            # Calculate statistical measures
            for activity in activity_durations['activity'].unique():
                durations = activity_durations[
                    activity_durations['activity'] == activity
                    ]['timestamp'].values

                if len(durations) > 1:  # Need at least 2 points for z-score
                    try:
                        # Use scipy stats for z-score
                        z_scores = stats.zscore(durations)
                        outliers = sum(np.abs(z_scores) > 3)
                    except ImportError:
                        # Fallback if scipy not available
                        mean = np.mean(durations)
                        std = np.std(durations)
                        z_scores = (durations - mean) / std if std != 0 else np.zeros_like(durations)
                        outliers = sum(np.abs(z_scores) > 3)

                    timing_metrics[activity] = {
                        'mean': float(np.mean(durations)),
                        'std': float(np.std(durations)),
                        'percentile_95': float(np.percentile(durations, 95)),
                        'outliers': int(outliers),
                        'min_duration': float(np.min(durations)),
                        'max_duration': float(np.max(durations))
                    }
                else:
                    # Handle case with insufficient data
                    timing_metrics[activity] = {
                        'mean': float(durations[0]) if len(durations) > 0 else 0.0,
                        'std': 0.0,
                        'percentile_95': float(durations[0]) if len(durations) > 0 else 0.0,
                        'outliers': 0,
                        'min_duration': float(durations[0]) if len(durations) > 0 else 0.0,
                        'max_duration': float(durations[0]) if len(durations) > 0 else 0.0
                    }

            logger.info(f"Calculated timing metrics for {len(timing_metrics)} activities")

        except Exception as e:
            logger.error(f"Error calculating timing metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

        return timing_metrics



    def _add_timing_failure_modes(self, timing_patterns: Dict) -> None:
        """Add failure modes based on timing analysis"""
        try:
            # Define timing thresholds (in seconds)
            thresholds = {
                'Trade Initiated': 3600,  # 1 hour
                'Market Data Validation': 1800,  # 30 minutes
                'Quote Provided': 900,  # 15 minutes
            }

            for activity, stats in timing_patterns.items():
                if stats['mean'] > thresholds.get(activity, 3600):
                    self.failure_modes.append({
                        'id': f"FM_TIME_{len(self.failure_modes) + 1}",
                        'activity': activity,
                        'description': f"Extended duration in {activity}",
                        'severity': 'Medium',
                        'occurrence': 2,
                        'detection': 2,
                        'rpn': 2 * 2,
                        'object_types': ['Trade', 'Market'],
                        'effects': ['Process delays', 'Customer dissatisfaction'],
                        'causes': ['Resource constraints', 'System performance'],
                        'controls': ['Duration monitoring', 'Performance alerts'],
                        'recommendations': [
                            'Optimize activity execution',
                            'Add performance monitoring',
                            'Implement duration alerts'
                        ]
                    })

        except Exception as e:
            logger.error(f"Error adding timing failure modes: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _add_resource_failure_modes(self, resource_patterns: Dict) -> None:
        """Add failure modes based on resource analysis"""
        try:
            workload = resource_patterns.get('workload', {})
            transitions = resource_patterns.get('transitions', {})

            # Check for resource overload
            avg_workload = np.mean(list(workload.values())) if workload else 0
            for resource, load in workload.items():
                if load > (avg_workload * 1.5):  # 50% above average
                    self.failure_modes.append({
                        'id': f"FM_RES_{len(self.failure_modes) + 1}",
                        'activity': 'Resource Allocation',
                        'description': f"Resource overload: {resource}",
                        'severity': 'High',
                        'occurrence': 3,
                        'detection': 2,
                        'rpn': 3 * 2,
                        'object_types': ['Resource'],
                        'effects': ['Process delays', 'Quality issues'],
                        'causes': ['Uneven workload distribution'],
                        'controls': ['Workload monitoring'],
                        'recommendations': [
                            'Redistribute workload',
                            'Add resource capacity',
                            'Implement load balancing'
                        ]
                    })

        except Exception as e:
            logger.error(f"Error adding resource failure modes: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _analyze_resource_patterns(self) -> Dict:
        """Analyze resource allocation patterns"""
        try:
            # Analyze resource workload
            workload = self.events_df['resource'].value_counts().to_dict()

            # Analyze resource transitions
            transitions = self._analyze_resource_transitions()

            return {
                'workload': workload,
                'transitions': transitions
            }

        except Exception as e:
            logger.error(f"Error analyzing resource patterns: {str(e)}")
            logger.error(traceback.format_exc())
            return {'workload': {}, 'transitions': {}}

    def _analyze_resource_transitions(self) -> Dict:
        """Analyze transitions between resources"""
        try:
            transitions = defaultdict(int)
            for case_id in self.events_df['case_id'].unique():
                case_events = self.events_df[
                    self.events_df['case_id'] == case_id
                    ].sort_values('timestamp')

                # Track resource changes within case
                previous_resource = None
                for _, event in case_events.iterrows():
                    current_resource = event['resource']
                    if previous_resource and current_resource != previous_resource:
                        transition_key = (previous_resource, current_resource)
                        transitions[transition_key] += 1
                    previous_resource = current_resource

            return dict(transitions)
        except Exception as e:
            logger.error(f"Error analyzing resource transitions: {str(e)}")
            return {}

    def _generate_failure_modes(
            self,
            case_sequences: pd.Series,
            timing_patterns: Dict,
            resource_patterns: Dict
    ):
        """Generate failure modes based on analysis patterns"""

        # Sequence-based failure modes
        self._add_sequence_failure_modes(case_sequences)

        # Timing-based failure modes
        self._add_timing_failure_modes(timing_patterns)

        # Resource-based failure modes
        self._add_resource_failure_modes(resource_patterns)

        # Object interaction failure modes
        self._add_object_interaction_failure_modes()

    def calculate_rpn(self, severity: int, occurrence: int, detection: int) -> int:
        """Calculate Risk Priority Number"""
        return severity * occurrence * detection

    def generate_report(self) -> Dict:
        """Generate comprehensive FMEA report"""
        try:
            # Calculate summary statistics
            summary = {
                'total_failure_modes': len(self.failure_modes),
                'high_risk_count': sum(1 for fm in self.failure_modes if fm.get('rpn', 0) > 200),
                'medium_risk_count': sum(1 for fm in self.failure_modes if 100 <= fm.get('rpn', 0) <= 200),
                'low_risk_count': sum(1 for fm in self.failure_modes if fm.get('rpn', 0) < 100)
            }

            # Organize failure modes by risk level
            high_risk = [fm for fm in self.failure_modes if fm.get('rpn', 0) > 200]
            medium_risk = [fm for fm in self.failure_modes if 100 <= fm.get('rpn', 0) <= 200]
            low_risk = [fm for fm in self.failure_modes if fm.get('rpn', 0) < 100]

            # Generate recommendations
            recommendations = []
            # High priority recommendations
            for fm in high_risk:
                recommendations.append({
                    'id': f"REC_H_{len(recommendations) + 1}",
                    'priority': 'High',
                    'description': f"Address {fm.get('description', 'Unknown Issue')}",
                    'target_date': (datetime.now() + timedelta(days=7)).isoformat(),
                    'status': 'Open',
                    'impact': fm.get('effects', ['Critical Impact'])[0]
                })

            # Medium priority recommendations
            for fm in medium_risk:
                recommendations.append({
                    'id': f"REC_M_{len(recommendations) + 1}",
                    'priority': 'Medium',
                    'description': f"Investigate {fm.get('description', 'Unknown Issue')}",
                    'target_date': (datetime.now() + timedelta(days=14)).isoformat(),
                    'status': 'Open',
                    'impact': fm.get('effects', ['Moderate Impact'])[0]
                })

            # Compile report
            report = {
                'summary': summary,
                'failure_modes': [
                    {
                        'id': fm.get('id', f'FM_{i}'),
                        'activity': fm.get('activity', 'Unknown'),
                        'description': fm.get('description', 'No description'),
                        'severity': fm.get('severity', 'Unknown'),
                        'occurrence': fm.get('occurrence', 0),
                        'detection': fm.get('detection', 0),
                        'rpn': fm.get('rpn', 0),
                        'object_types': fm.get('object_types', []),
                        'effects': fm.get('effects', []),
                        'causes': fm.get('causes', []),
                        'recommendations': fm.get('recommendations', [])
                    }
                    for i, fm in enumerate(self.failure_modes, 1)
                ],
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Generated report with {summary['total_failure_modes']} failure modes")
            logger.info(f"High Risk: {summary['high_risk_count']}, "
                        f"Medium Risk: {summary['medium_risk_count']}, "
                        f"Low Risk: {summary['low_risk_count']}")

            return report

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _format_failure_mode(self, fm: FailureMode) -> Dict:
        """Format failure mode for reporting"""
        return {
            'id': fm.id,
            'activity': fm.activity,
            'description': fm.description,
            'severity': fm.severity,
            'occurrence': fm.occurrence,
            'detection': fm.detection,
            'rpn': fm.rpn,
            'object_types': fm.object_types,
            'effects': fm.effects,
            'causes': fm.causes,
            'controls': fm.controls,
            'recommendations': fm.recommendations
        }

    def _generate_recommendations(self) -> List[Dict]:
        """Generate prioritized recommendations"""
        recommendations = []

        # Group failure modes by RPN
        high_risk = [fm for fm in self.failure_modes if fm.rpn > 200]
        medium_risk = [fm for fm in self.failure_modes if 100 <= fm.rpn <= 200]

        # Generate recommendations for high-risk failure modes
        for fm in high_risk:
            recommendations.append({
                'priority': 'High',
                'failure_mode': fm.id,
                'recommendation': fm.recommendations[0] if fm.recommendations else '',
                'timeline': 'Immediate',
                'required_resources': self._estimate_required_resources(fm)
            })

        # Generate recommendations for medium-risk failure modes
        for fm in medium_risk:
            recommendations.append({
                'priority': 'Medium',
                'failure_mode': fm.id,
                'recommendation': fm.recommendations[0] if fm.recommendations else '',
                'timeline': '30 days',
                'required_resources': self._estimate_required_resources(fm)
            })

        return recommendations