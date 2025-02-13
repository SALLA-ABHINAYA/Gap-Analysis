# pages/5_FMEA_Analysis.py
import os
import traceback
from collections import defaultdict

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Set
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import tornado

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Increase pandas display limit
pd.set_option("styler.render.max_elements", 600000)


@dataclass
class OCELFailureMode:
    """Enhanced failure mode representation for OCEL process mining"""
    id: str
    activity: str
    object_type: str
    description: str
    severity: int = 0
    likelihood: int = 0
    detectability: int = 0
    rpn: int = 0
    violation_type: str = ""
    effects: List[str] = field(default_factory=list)
    causes: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)  # Added this field

    # Impact Analysis
    business_impact: Dict[str, Any] = field(default_factory=dict)
    operational_impact: Dict[str, Any] = field(default_factory=dict)
    regulatory_impact: Dict[str, Any] = field(default_factory=dict)

    # Detection and Control
    current_controls: List[str] = field(default_factory=list)
    detection_methods: List[str] = field(default_factory=list)

    # Object Relationships
    affected_objects: List[str] = field(default_factory=list)
    dependent_activities: List[str] = field(default_factory=list)

    # Mitigation
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    mitigation_status: str = "Open"
    priority_level: str = "Medium"

    # Detailed failure analysis
    failure_category: str = ""
    subcategory: str = ""

    # Root cause analysis
    direct_causes: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    systemic_causes: List[str] = field(default_factory=list)

    # Effect analysis
    immediate_effects: List[str] = field(default_factory=list)
    intermediate_effects: List[str] = field(default_factory=list)
    long_term_effects: List[str] = field(default_factory=list)

    # Object relationships
    impacted_resources: List[str] = field(default_factory=list)

    # Control mechanisms
    control_effectiveness: Dict[str, float] = field(default_factory=dict)

    # Mitigation strategies
    preventive_measures: List[str] = field(default_factory=list)
    contingency_plans: List[str] = field(default_factory=list)


class OCELDataManager:
    """Manages OCEL data loading and common operations"""

    def __init__(self, ocel_path: str):
        """Initialize with path to OCEL data"""
        try:
            # Load OCEL relationship definitions
            with open('ocpm_output/output_ocel.json', 'r') as f:
                self.ocel_model = json.load(f)

            # Load process data
            with open(ocel_path, 'r') as f:
                self.ocel_data = json.load(f)

            # Create events DataFrame
            self.events_df = self._create_events_df()

            # Store object types and their relationships
            self.object_relationships = {
                obj_type: data.get('relationships', [])
                for obj_type, data in self.ocel_model.items()
            }

            # Store object attributes
            self.object_attributes = {
                obj_type: data.get('attributes', [])
                for obj_type, data in self.ocel_model.items()
            }

            # Store object activities
            self.object_activities = {
                obj_type: data.get('activities', [])
                for obj_type, data in self.ocel_model.items()
            }

        except Exception as e:
            logger.error(f"Error initializing OCEL data manager: {str(e)}")
            raise

    def _create_events_df(self) -> pd.DataFrame:
        """Create DataFrame from OCEL events"""
        events_data = []
        for event in self.ocel_data['ocel:events']:
            event_data = {
                'ocel:id': event.get('ocel:id', ''),
                'ocel:timestamp': pd.to_datetime(event.get('ocel:timestamp')),
                'ocel:activity': event.get('ocel:activity', ''),
                'ocel:objects': event.get('ocel:objects', []),
                'case_id': event.get('ocel:attributes', {}).get('case_id',
                                                                event.get('case:concept:name',
                                                                          event.get('ocel:id', '')))
            }

            # Add all attributes
            attributes = event.get('ocel:attributes', {})
            for key, value in attributes.items():
                event_data[key] = value

            events_data.append(event_data)

        return pd.DataFrame(events_data)

    def get_expected_relationships(self, obj_type: str) -> Set[str]:
        """Get expected relationships for object type from OCEL model"""
        return set(self.object_relationships.get(obj_type, []))

    def get_expected_attributes(self, obj_type: str) -> Set[str]:
        """Get expected attributes for object type from OCEL model"""
        return set(self.object_attributes.get(obj_type, []))

    def get_expected_activities(self, obj_type: str) -> List[str]:
        """Get expected activities for object type from OCEL model"""
        return self.object_activities.get(obj_type, [])

class OCELEnhancedFMEA:
    """Enhanced FMEA analyzer for OCEL data"""

    def __init__(self, ocel_data: Dict):
        """Initialize analyzer with enhanced error handling and logging"""
        logger.info("Initializing OCELEnhancedFMEA")
        try:
            self.ocel_data = ocel_data

            # Initialize OCEL data manager for relationship/attribute management
            self.data_manager = OCELDataManager("ocpm_output/process_data.json")

            # Validate OCEL data structure
            if not isinstance(ocel_data, dict):
                raise ValueError(f"Invalid OCEL data type: {type(ocel_data)}")

            if 'ocel:events' not in ocel_data:
                raise ValueError("Missing ocel:events in data")

            # Convert events to DataFrame with enhanced error handling
            logger.info("Converting events to DataFrame")
            events_data = []
            for event in ocel_data['ocel:events']:
                try:
                    event_data = {
                        'ocel:id': event.get('ocel:id', ''),
                        'ocel:timestamp': pd.to_datetime(event.get('ocel:timestamp')),
                        'ocel:activity': event.get('ocel:activity', ''),
                        'ocel:objects': event.get('ocel:objects', []),
                        'case_id': event.get('ocel:attributes', {}).get('case_id',
                                                                        event.get('case:concept:name',
                                                                                  event.get('ocel:id', ''))),
                    }

                    # Add all attributes
                    attributes = event.get('ocel:attributes', {})
                    for key, value in attributes.items():
                        event_data[key] = value

                    events_data.append(event_data)

                except Exception as e:
                    logger.warning(f"Error processing event: {str(e)}")
                    continue

            self.events = pd.DataFrame(events_data)
            logger.info(f"Created events DataFrame with {len(self.events)} rows")

            self._validate_columns()

            # Initialize core components using data manager
            self.object_types = list(self.data_manager.object_relationships.keys())
            logger.info(f"Found {len(self.object_types)} object types")

            # Build object relationships
            logger.info("Building object relationships")
            self.object_relationships = self._build_object_relationships()
            logger.info(f"Built relationships for {len(self.object_relationships)} objects")

            # Identify convergence points
            logger.info("Identifying convergence points")
            self.convergence_points = self._identify_convergence_points()
            logger.info(f"Identified {len(self.convergence_points)} convergence points")

            # Identify divergence points
            logger.info("Identifying divergence points")
            self.divergence_points = self._identify_divergence_points()
            logger.info(f"Identified {len(self.divergence_points)} divergence points")

            # Compute activity statistics
            logger.info("Computing activity statistics")
            self.activity_stats = self._compute_activity_stats()
            logger.info(f"Computed statistics for {len(self.activity_stats)} activities")

            # Compute object statistics
            logger.info("Computing object statistics")
            self.object_stats = self._compute_object_stats()
            logger.info(f"Computed statistics for {len(self.object_stats)} object types")

            # Analyze sequence patterns
            logger.info("Analyzing sequence patterns")
            self.sequence_patterns = self._analyze_sequence_patterns()
            if self.sequence_patterns:
                logger.info(f"Found patterns for {len(self.sequence_patterns)} object types")
            else:
                logger.warning("No sequence patterns found")

            # Initialize timing thresholds from OCEL model
            logger.info("Initializing timing thresholds")
            self.timing_thresholds = self._initialize_timing_thresholds()

            # Initialize empty failure modes list
            self.failure_modes = []

            # Store expected sequences from OCEL model
            self.expected_sequences = {
                obj_type: data.get('activities', [])
                for obj_type, data in self.data_manager.ocel_model.items()
            }

            logger.info("Initialization complete")

        except Exception as e:
            logger.error(f"Error initializing OCELEnhancedFMEA: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_timing_thresholds(self) -> Dict[str, Dict]:
        """Initialize timing thresholds from OCEL model"""
        try:
            thresholds = {}
            for obj_type, data in self.data_manager.ocel_model.items():
                activities = data.get('activities', [])
                thresholds[obj_type] = {
                    'total_duration': 24,  # Default 24 hours for full process
                    'activity_thresholds': {
                        activity: {
                            'max_duration_hours': 1,  # Default 1 hour per activity
                            'max_gap_after_hours': 2  # Default 2 hour gap allowed
                        }
                        for activity in activities
                    }
                }
            return thresholds
        except Exception as e:
            logger.error(f"Error initializing timing thresholds: {str(e)}")
            return {}

    def _validate_columns(self):
        """Validate required columns exist in DataFrame"""
        required_columns = ['case_id', 'ocel:activity', 'ocel:timestamp']
        missing = [col for col in required_columns if col not in self.events.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _build_object_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build graph of object relationships from events"""
        logger.info("Starting to build object relationships")
        try:
            relationships = defaultdict(list)

            # Track processed pairs to avoid duplicates
            processed_pairs = set()

            for _, event in self.events.iterrows():
                try:
                    objects = event.get('ocel:objects', [])
                    # Skip if less than 2 objects
                    if len(objects) < 2:
                        continue

                    # Analyze each object pair in the event
                    for i, obj1 in enumerate(objects):
                        for obj2 in objects[i + 1:]:
                            # Create unique pair identifier
                            pair_id = tuple(sorted([obj1['id'], obj2['id']]))

                            # Skip if already processed
                            if pair_id in processed_pairs:
                                continue

                            processed_pairs.add(pair_id)

                            # Record relationship for both objects
                            relationships[obj1['id']].append({
                                'related_object': obj2['id'],
                                'object_type': obj2.get('type', 'Unknown'),
                                'activity': event['ocel:activity'],
                                'timestamp': event['ocel:timestamp'],
                                'event_id': event['ocel:id']
                            })

                            relationships[obj2['id']].append({
                                'related_object': obj1['id'],
                                'object_type': obj1.get('type', 'Unknown'),
                                'activity': event['ocel:activity'],
                                'timestamp': event['ocel:timestamp'],
                                'event_id': event['ocel:id']
                            })

                except Exception as e:
                    logger.warning(f"Error processing event relationships: {str(e)}")
                    continue

            # Convert defaultdict to regular dict
            result = dict(relationships)
            logger.info(f"Built relationships for {len(result)} objects with {len(processed_pairs)} unique connections")

            # Log some statistics
            relationship_counts = {obj_id: len(rels) for obj_id, rels in result.items()}
            if relationship_counts:
                avg_relationships = sum(relationship_counts.values()) / len(relationship_counts)
                max_relationships = max(relationship_counts.values())
                logger.info(f"Average relationships per object: {avg_relationships:.2f}")
                logger.info(f"Maximum relationships for an object: {max_relationships}")

            return result

        except Exception as e:
            logger.error(f"Error building object relationships: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _check_timing_violation(self, event: pd.Series) -> bool:
        """
        Check for timing violations in OCPM context considering:
        - Activity-specific thresholds
        - Object type timing requirements
        - Cross-object timing dependencies
        - Business hour constraints
        - Market-specific timing rules

        Args:
            event: Single event series containing timestamp and activity info

        Returns:
            bool: True if timing violation detected, False otherwise
        """
        try:
            # Get activity-specific thresholds
            timing_thresholds = {
                'Trade Execution': timedelta(minutes=5),  # Quick execution required
                'Risk Assessment': timedelta(minutes=15),  # Pre-trade risk check
                'Market Data Validation': timedelta(minutes=2),  # Near real-time
                'Settlement': timedelta(hours=2),  # End-of-day process
                'Position Reconciliation': timedelta(hours=1)  # Regular checks
            }

            # Get event context
            activity = event['ocel:activity']
            timestamp = pd.to_datetime(event['ocel:timestamp'])
            case_id = event['case_id']

            # Get previous event in same case
            case_events = self.events[self.events['case_id'] == case_id]
            case_events = case_events.sort_values('ocel:timestamp')
            prev_event_idx = case_events.index.get_loc(event.name) - 1

            if prev_event_idx >= 0:
                prev_event = case_events.iloc[prev_event_idx]
                time_diff = timestamp - pd.to_datetime(prev_event['ocel:timestamp'])

                # Check activity-specific threshold
                threshold = timing_thresholds.get(activity)
                if threshold and time_diff > threshold:
                    return True

                # Check object-type timing rules
                for obj in event.get('ocel:objects', []):
                    obj_type = obj.get('type')
                    if obj_type == 'Trade':
                        # Trades should complete within market hours
                        if not self._is_market_hours(timestamp):
                            return True
                    elif obj_type == 'Risk':
                        # Risk checks must happen before trade execution
                        if activity == 'Trade Execution' and 'Risk Assessment' not in \
                                case_events[case_events['ocel:timestamp'] < timestamp]['ocel:activity'].values:
                            return True

                # Check cross-object timing dependencies
                if self._has_timing_dependency_violation(event, case_events):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking timing violation: {str(e)}")
            return False

    def _has_temporal_dependency(self, activity: str) -> bool:
        """
        Analyze temporal dependencies of activities in object-centric context.

        This method examines complex temporal relationships between activities, considering:
        - Direct activity dependencies (e.g., validation before execution)
        - Cross-object temporal constraints
        - Business cycle dependencies (e.g., end-of-day processes)
        - Market-driven timing requirements

        Args:
            activity: Name of the activity to check for temporal dependencies

        Returns:
            bool: True if activity has temporal dependencies, False otherwise
        """
        try:
            # Define temporal dependency map for activities
            temporal_dependencies = {
                'Trade Execution': {
                    'required_before': ['Market Data Validation', 'Risk Assessment'],
                    'required_after': [],
                    'timing_constraint': 'market_hours',
                    'max_delay': timedelta(minutes=5)
                },
                'Settlement': {
                    'required_before': ['Trade Execution', 'Position Reconciliation'],
                    'required_after': ['Trade Confirmation'],
                    'timing_constraint': 'end_of_day',
                    'max_delay': timedelta(hours=2)
                },
                'Risk Assessment': {
                    'required_before': [],
                    'required_after': ['Trade Initiated'],
                    'timing_constraint': 'continuous',
                    'max_delay': timedelta(minutes=15)
                },
                'Position Reconciliation': {
                    'required_before': ['Settlement'],
                    'required_after': ['Trade Execution'],
                    'timing_constraint': 'daily',
                    'max_delay': timedelta(hours=1)
                },
                'Market Data Validation': {
                    'required_before': ['Trade Execution'],
                    'required_after': [],
                    'timing_constraint': 'real_time',
                    'max_delay': timedelta(minutes=1)
                }
            }

            # Check if activity has defined temporal dependencies
            if activity in temporal_dependencies:
                dependency_info = temporal_dependencies[activity]

                # Check for any type of temporal constraint
                has_timing_requirements = any([
                    dependency_info['required_before'],
                    dependency_info['required_after'],
                    dependency_info['timing_constraint'] != '',
                    dependency_info['max_delay'] is not None
                ])

                if has_timing_requirements:
                    # Validate temporal patterns in event log
                    activity_events = self.events[self.events['ocel:activity'] == activity]

                    for _, event in activity_events.iterrows():
                        case_id = event['case_id']
                        timestamp = pd.to_datetime(event['ocel:timestamp'])

                        # Get case events
                        case_events = self.events[self.events['case_id'] == case_id]
                        case_events = case_events.sort_values('ocel:timestamp')

                        # Check required predecessor activities
                        for req_activity in dependency_info['required_before']:
                            prior_events = case_events[
                                (case_events['ocel:timestamp'] < timestamp) &
                                (case_events['ocel:activity'] == req_activity)
                                ]
                            if prior_events.empty:
                                return True

                        # Check required successor activities
                        for req_activity in dependency_info['required_after']:
                            subsequent_events = case_events[
                                (case_events['ocel:timestamp'] > timestamp) &
                                (case_events['ocel:activity'] == req_activity)
                                ]
                            if subsequent_events.empty:
                                return True

                        # Check timing constraints
                        constraint = dependency_info['timing_constraint']
                        if constraint == 'market_hours' and not self._is_market_hours(timestamp):
                            return True
                        elif constraint == 'end_of_day' and not self._is_end_of_day(timestamp):
                            return True
                        elif constraint == 'real_time' and not self._is_real_time(timestamp):
                            return True

                        # Check maximum delay constraint
                        max_delay = dependency_info['max_delay']
                        if max_delay and self._exceeds_max_delay(event, max_delay):
                            return True

                return has_timing_requirements

            return False

        except Exception as e:
            logger.error(f"Error checking temporal dependency for {activity}: {str(e)}")
            return False

    def _is_end_of_day(self, timestamp: datetime) -> bool:
        """Check if timestamp is within end-of-day processing window (16:00-18:00)"""
        return 16 <= timestamp.hour < 18 and timestamp.weekday() < 5

    def _is_real_time(self, timestamp: datetime) -> bool:
        """Check if event happened within real-time processing threshold (< 1 second delay)"""
        current_time = pd.Timestamp.now()
        return (current_time - timestamp).total_seconds() < 1

    def _exceeds_max_delay(self, event: pd.Series, max_delay: timedelta) -> bool:
        """Check if event exceeds maximum allowed delay"""
        case_events = self.events[self.events['case_id'] == event['case_id']]
        case_events = case_events.sort_values('ocel:timestamp')

        event_idx = case_events.index.get_loc(event.name)
        if event_idx > 0:
            prev_event = case_events.iloc[event_idx - 1]
            actual_delay = pd.to_datetime(event['ocel:timestamp']) - pd.to_datetime(prev_event['ocel:timestamp'])
            return actual_delay > max_delay

        return False

    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within market hours (9:00-17:00)"""
        return 9 <= timestamp.hour < 17 and timestamp.weekday() < 5

    def _has_timing_dependency_violation(self, event: pd.Series, case_events: pd.DataFrame) -> bool:
        """
        Check for violations of timing dependencies between objects

        Dependencies examples:
        - Market data must be validated before trade execution
        - Risk assessment must complete before settlement
        - Position reconciliation must follow settlement
        """
        activity = event['ocel:activity']
        timestamp = pd.to_datetime(event['ocel:timestamp'])

        # Define required predecessor activities
        dependencies = {
            'Trade Execution': ['Market Data Validation', 'Risk Assessment'],
            'Settlement': ['Trade Execution', 'Position Reconciliation'],
            'Position Reconciliation': ['Trade Execution']
        }

        required_activities = dependencies.get(activity, [])
        if required_activities:
            prior_activities = case_events[
                case_events['ocel:timestamp'] < timestamp
                ]['ocel:activity'].unique()

            # Check if any required activity is missing
            if not all(req in prior_activities for req in required_activities):
                return True

        return False

    def _compute_object_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute object-level statistics"""
        stats = defaultdict(lambda: {
            'count': 0,
            'activities': set(),
            'related_objects': set(),
            'attributes': defaultdict(set)
        })

        for _, event in self.events.iterrows():
            for obj in event.get('ocel:objects', []):
                obj_type = obj.get('type', 'Unknown')
                stats[obj_type]['count'] += 1
                stats[obj_type]['activities'].add(event['ocel:activity'])

                # Track attributes
                for attr, value in obj.items():
                    if attr not in ['id', 'type']:
                        stats[obj_type]['attributes'][attr].add(str(value))

        return dict(stats)

    def _analyze_sequence_patterns(self) -> Dict[str, List[str]]:
        """Analyze common sequence patterns in the log"""
        logger.info("Starting sequence pattern analysis")
        patterns = defaultdict(list)

        try:
            # Get case ID column - try different possible names
            case_id_col = None
            possible_case_columns = ['case:concept:name', 'case_id', 'case', 'case:id']

            for col in possible_case_columns:
                if col in self.events.columns:
                    case_id_col = col
                    logger.info(f"Found case ID column: {col}")
                    break

            if not case_id_col:
                logger.warning("No case ID column found, using ocel:id as case identifier")
                case_id_col = 'ocel:id'

            # Group events by case
            logger.info(f"Grouping events by {case_id_col}")
            case_events = self.events.groupby(case_id_col)

            pattern_count = 0
            for case_id, events in case_events:
                try:
                    sorted_events = events.sort_values('ocel:timestamp')
                    sequence = list(sorted_events['ocel:activity'])

                    # Store sequence for each object type involved
                    for obj in events.iloc[0].get('ocel:objects', []):
                        obj_type = obj.get('type', 'Unknown')
                        if sequence not in patterns[obj_type]:
                            patterns[obj_type].append(sequence)
                            pattern_count += 1

                except Exception as e:
                    logger.error(f"Error processing case {case_id}: {str(e)}")
                    continue

            logger.info(f"Found {pattern_count} unique sequence patterns across {len(patterns)} object types")
            return dict(patterns)

        except Exception as e:
            logger.error(f"Error in sequence pattern analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def calculate_likelihood(self, failure_mode: OCELFailureMode) -> int:
        """Calculate likelihood considering OCPM patterns"""
        likelihood = 1

        # Analyze historical occurrence
        activity_events = self.events[self.events['ocel:activity'] == failure_mode.activity]
        total_instances = len(activity_events)

        if total_instances > 0:
            # Calculate failure rate for this activity
            failure_instances = activity_events[
                activity_events.apply(
                    lambda x: self._has_failure_condition(x, failure_mode.violation_type),
                    axis=1
                )
            ]
            failure_rate = len(failure_instances) / total_instances

            # Scale failure rate to likelihood score (1-10)
            likelihood += int(failure_rate * 10)

        # Add complexity factor
        object_interactions = self._get_object_interactions(failure_mode.activity)
        if len(object_interactions) > 2:
            likelihood += 1

        # Consider temporal patterns
        if self._has_temporal_dependency(failure_mode.activity):
            likelihood += 1

        return min(likelihood, 10)

    def _has_failure_condition(self, event: pd.Series, violation_type: str) -> bool:
        """Check if event has specific failure condition"""
        if violation_type == 'attribute':
            return any(attr not in event for attr in ['currency_pair', 'notional_amount'])
        elif violation_type == 'relationship':
            return len(event.get('ocel:objects', [])) < 2
        elif violation_type == 'timing':
            return self._check_timing_violation(event)
        return False

    def _get_object_interactions(self, activity: str) -> Set[str]:
        """Get unique object types interacting in activity"""
        interactions = set()
        activity_events = self.events[self.events['ocel:activity'] == activity]
        for _, event in activity_events.iterrows():
            for obj in event.get('ocel:objects', []):
                interactions.add(obj.get('type'))
        return interactions

    def _has_failure_condition(self, event: pd.Series, violation_type: str) -> bool:
        """Check if event has specific failure condition"""
        if violation_type == 'attribute':
            return any(attr not in event for attr in ['currency_pair', 'notional_amount'])
        elif violation_type == 'relationship':
            return len(event.get('ocel:objects', [])) < 2
        elif violation_type == 'timing':
            return self._check_timing_violation(event)
        return False

    def _get_object_interactions(self, activity: str) -> Set[str]:
        """Get unique object types interacting in activity"""
        interactions = set()
        activity_events = self.events[self.events['ocel:activity'] == activity]
        for _, event in activity_events.iterrows():
            for obj in event.get('ocel:objects', []):
                interactions.add(obj.get('type'))
        return interactions

    def calculate_detectability(self, failure_mode: OCELFailureMode) -> int:
        """Calculate detectability in OCPM context"""
        detectability = 10  # Start with worst detectability

        # System controls
        if self._has_automated_monitoring(failure_mode.activity):
            detectability -= 3

        # Object visibility
        object_visibility = {
            'Trade': -2,  # High visibility
            'Market': -2,  # Real-time data
            'Risk': -1,  # Regular monitoring
            'Settlement': -2,  # Clear checkpoints
            'Position': -1  # Daily reconciliation
        }
        detectability += object_visibility.get(failure_mode.object_type, 0)

        # Process controls
        control_points = self._get_control_points(failure_mode.activity)
        detectability -= min(len(control_points), 3)

        # Multi-object detectability
        if len(self._get_object_interactions(failure_mode.activity)) > 1:
            detectability -= 1  # Easier to detect with multiple object involvement

        return max(detectability, 1)

    def _has_automated_monitoring(self, activity: str) -> bool:
        """Check if activity has automated monitoring"""
        automated_activities = {
            'Trade Validation', 'Market Data Validation',
            'Position Reconciliation', 'Risk Assessment'
        }
        return activity in automated_activities

    def _get_control_points(self, activity: str) -> List[str]:
        """Get control points for activity"""
        controls = {
            'Trade Execution': ['Pre-trade validation', 'Price validation', 'Limit checks'],
            'Settlement': ['Position verification', 'Payment validation', 'Reconciliation'],
            'Risk Assessment': ['Exposure validation', 'Limit monitoring', 'Portfolio checks']
        }
        return controls.get(activity, [])

    def _get_expected_sequence(self, activity: str) -> List[str]:
        """Get expected sequence for given activity"""
        # This should be customized based on domain knowledge
        # For now, return common sequences from the log
        sequences = self.sequence_patterns.get(activity, [])
        return sequences[0] if sequences else []

    def _get_actual_sequences(self, activity: str) -> List[List[str]]:
        """Get actual sequences from the log"""
        sequences = []
        try:
            case_events = self.events.groupby('case_id')

            for _, events in case_events:
                if activity in events['ocel:activity'].values:
                    sorted_events = events.sort_values('ocel:timestamp')
                    sequences.append(list(sorted_events['ocel:activity']))

            logger.info(f"Found {len(sequences)} sequences for activity {activity}")
            return sequences

        except Exception as e:
            logger.error(f"Error getting sequences for activity {activity}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _validate_sequence(self, actual: List[str], expected: List[str]) -> bool:
        """Validate if actual sequence follows expected pattern"""
        if not expected:
            return True

        # Check if actual sequence contains expected sequence in order
        i = 0
        for act in actual:
            if i < len(expected) and act == expected[i]:
                i += 1
        return i == len(expected)

    def _get_primary_object_type(self, activity: str) -> str:
        """Determine primary object type for activity"""
        activity_objects = defaultdict(int)
        activity_events = self.events[self.events['ocel:activity'] == activity]

        for _, event in activity_events.iterrows():
            for obj in event.get('ocel:objects', []):
                activity_objects[obj.get('type', 'Unknown')] += 1

        return max(activity_objects.items(), key=lambda x: x[1])[0] if activity_objects else 'Unknown'

    def _identify_timing_violations(self) -> List[Dict]:
        """Identify timing violations in the process"""
        violations = []
        case_events = self.events.groupby('case_id')

        for case_id, events in case_events:
            sorted_events = events.sort_values('ocel:timestamp')
            for i in range(len(sorted_events) - 1):
                current = sorted_events.iloc[i]
                next_event = sorted_events.iloc[i + 1]
                time_diff = (next_event['ocel:timestamp'] - current['ocel:timestamp']).total_seconds() / 3600

                # Check for unusual delays
                if time_diff > 24:  # Threshold can be customized
                    violations.append({
                        'id': f"TV_{len(violations)}",
                        'activity': current['ocel:activity'],
                        'object_type': self._get_primary_object_type(current['ocel:activity']),
                        'description': f"Unusual delay between activities",
                        'violation_type': 'timing',
                        'time_diff': time_diff,
                        'next_activity': next_event['ocel:activity']
                    })

        return violations

    def _validate_convergence(self, convergence_point: Dict) -> bool:
        """Validate convergence point"""
        # This should be customized based on domain rules
        # For now, basic validation
        return len(convergence_point['object_types']) <= 3

    def _validate_divergence(self, divergence_point: Dict) -> bool:
        """Validate divergence point"""
        # This should be customized based on domain rules
        # For now, basic validation
        return True

    def _identify_convergence_points(self) -> List[Dict[str, Any]]:
        """Identify points where multiple objects converge in the process"""
        logger.info("Starting convergence point identification")
        try:
            convergence_points = []
            for _, event in self.events.iterrows():
                objects = event.get('ocel:objects', [])
                if len(objects) > 1:
                    convergence_point = {
                        'event_id': event['ocel:id'],
                        'activity': event['ocel:activity'],
                        'timestamp': event['ocel:timestamp'],
                        'objects': objects,
                        'object_types': list(set(obj.get('type') for obj in objects))
                    }
                    convergence_points.append(convergence_point)

            logger.info(f"Found {len(convergence_points)} convergence points")
            return convergence_points

        except Exception as e:
            logger.error(f"Error identifying convergence points: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _identify_divergence_points(self) -> List[Dict[str, Any]]:
        """Identify points where object paths diverge"""
        logger.info("Starting divergence point identification")
        try:
            # Track object paths
            object_paths = defaultdict(list)
            for _, event in self.events.iterrows():
                for obj in event.get('ocel:objects', []):
                    object_paths[obj['id']].append({
                        'event_id': event['ocel:id'],
                        'timestamp': event['ocel:timestamp'],
                        'activity': event['ocel:activity']
                    })

            divergence_points = []
            for obj_id, path in object_paths.items():
                # Sort path by timestamp
                sorted_path = sorted(path, key=lambda x: x['timestamp'])

                for i in range(len(sorted_path) - 1):
                    current_event = sorted_path[i]
                    next_event = sorted_path[i + 1]

                    # Check for divergence conditions
                    current_objects = set([obj['id'] for obj in
                                           self.events[self.events['ocel:id'] == current_event['event_id']]
                                          .iloc[0].get('ocel:objects', [])])

                    next_objects = set([obj['id'] for obj in
                                        self.events[self.events['ocel:id'] == next_event['event_id']]
                                       .iloc[0].get('ocel:objects', [])])

                    if len(current_objects.intersection(next_objects)) < len(current_objects):
                        divergence_points.append({
                            'object_id': obj_id,
                            'event_id': current_event['event_id'],
                            'next_event_id': next_event['event_id'],
                            'timestamp': current_event['timestamp'],
                            'activity': current_event['activity'],
                            'next_activity': next_event['activity']
                        })

            logger.info(f"Found {len(divergence_points)} divergence points")
            return divergence_points

        except Exception as e:
            logger.error(f"Error identifying divergence points: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _compute_activity_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute comprehensive activity statistics"""
        stats = defaultdict(lambda: {
            'count': 0,
            'objects': defaultdict(set),
            'resources': set(),
            'duration_stats': [],
            'sequence_patterns': defaultdict(int),
            'error_rates': defaultdict(int),
            'attribute_patterns': defaultdict(set)
        })

        for _, event in self.events.iterrows():
            activity = event['ocel:activity']
            stats[activity]['count'] += 1

            # Track objects and their relationships
            for obj in event.get('ocel:objects', []):
                stats[activity]['objects'][obj.get('type')].add(obj['id'])

            # Track resources
            if 'resource' in event.get('ocel:attributes', {}):
                stats[activity]['resources'].add(event['ocel:attributes']['resource'])

            # Track attributes
            for attr, value in event.get('ocel:attributes', {}).items():
                stats[activity]['attribute_patterns'][attr].add(str(value))

        return dict(stats)

    def calculate_severity(self, failure_mode: OCELFailureMode) -> int:
        """Calculate severity in OCPM context"""
        severity = 1

        # Object criticality impact
        object_criticality = {
            'Trade': 5,  # Highest criticality - direct business impact
            'Risk': 4,  # Critical for compliance and risk management
            'Market': 3,  # Market data and pricing impact
            'Settlement': 4,  # Financial impact
            'Position': 3  # Portfolio impact
        }
        severity += object_criticality.get(failure_mode.object_type, 1)

        # Multi-object impact
        affected_objects = set()
        events = self.events[self.events['ocel:activity'] == failure_mode.activity]
        for _, event in events.iterrows():
            for obj in event.get('ocel:objects', []):
                affected_objects.add(obj.get('type'))

        # Add severity based on number of affected object types
        severity += min(len(affected_objects), 3)

        # Process stage impact
        if failure_mode.activity in ['Trade Execution', 'Settlement', 'Risk Assessment']:
            severity += 1

        # Regulatory impact
        if any(reg in failure_mode.description.lower() for reg in ['regulatory', 'compliance', 'legal']):
            severity += 1

        return min(severity, 10)

    def _calculate_cascade_depth(self, failure_mode: OCELFailureMode) -> int:
        """Calculate how deep failures can cascade"""
        depth = 0
        affected = {failure_mode.activity}
        current_level = {failure_mode.activity}

        while current_level and depth < 5:
            next_level = set()
            for activity in current_level:
                for _, event in self.events[self.events['ocel:activity'] == activity].iterrows():
                    objects = event.get('ocel:objects', [])
                    for obj in objects:
                        related = self.object_relationships.get(obj['id'], [])
                        for rel_obj in related:
                            rel_events = self.events[
                                self.events['ocel:objects'].apply(
                                    lambda x: rel_obj in [o['id'] for o in x]
                                )
                            ]
                            next_activities = set(rel_events['ocel:activity']) - affected
                            next_level.update(next_activities)

            if next_level:
                depth += 1
                affected.update(next_level)
                current_level = next_level
            else:
                break

        return depth

    def identify_failure_modes(self) -> List[Dict]:
        """Enhanced failure mode identification"""
        failure_modes = []

        # Object-level failures
        failure_modes.extend(self._identify_object_failures())

        # Event-level failures
        failure_modes.extend(self._identify_event_failures())

        # System-level failures
        failure_modes.extend(self._identify_system_failures())

        # Calculate metrics
        for fm in failure_modes:
            fm['severity'] = self.calculate_severity(OCELFailureMode(**fm))
            fm['likelihood'] = self.calculate_likelihood(OCELFailureMode(**fm))
            fm['detectability'] = self.calculate_detectability(OCELFailureMode(**fm))
            fm['rpn'] = fm['severity'] * fm['likelihood'] * fm['detectability']

        return sorted(failure_modes, key=lambda x: x['rpn'], reverse=True)

    def _get_expected_relationships(self, obj_type: str) -> Set[str]:
        """Get expected relationships from OCEL model"""
        return self.data_manager.get_expected_relationships(obj_type)

    def _get_actual_relationships(self, obj_type: str) -> Set[str]:
        """Get actual relationships from event log for an object type"""
        logger.info(f"Getting actual relationships for {obj_type}")
        try:
            relationships = set()

            # Find all events involving this object type
            for _, event in self.events.iterrows():
                objects = event.get('ocel:objects', [])
                current_type_objects = [obj for obj in objects if obj.get('type') == obj_type]
                other_type_objects = [obj for obj in objects if obj.get('type') != obj_type]

                if current_type_objects and other_type_objects:
                    for other_obj in other_type_objects:
                        relationships.add(other_obj.get('type', 'Unknown'))

            logger.info(f"Found {len(relationships)} actual relationships for {obj_type}")
            return relationships

        except Exception as e:
            logger.error(f"Error getting actual relationships for {obj_type}: {str(e)}")
            return set()

    def _check_attribute_violations(self, obj_type: str) -> List[Dict]:
        """Check for attribute violations using OCEL model definition"""
        logger.info(f"Checking attribute violations for {obj_type}")
        try:
            violations = []
            expected_attributes = self.data_manager.get_expected_attributes(obj_type)

            # Check each event involving this object type
            for _, event in self.events.iterrows():
                objects = event.get('ocel:objects', [])
                for obj in objects:
                    if obj.get('type') == obj_type:
                        # Check for missing attributes
                        missing_attrs = expected_attributes - set(obj.keys())
                        if missing_attrs:
                            violations.append({
                                'id': f"AV_{len(violations)}",
                                'activity': event['ocel:activity'],
                                'object_type': obj_type,
                                'description': f"Missing attributes: {', '.join(missing_attrs)}",
                                'violation_type': 'attribute',
                                'attributes': list(missing_attrs)
                            })

            logger.info(f"Found {len(violations)} attribute violations for {obj_type}")
            return violations

        except Exception as e:
            logger.error(f"Error checking attribute violations for {obj_type}: {str(e)}")
            return []

    def _is_valid_relationship(self, source_type: str, target_type: str) -> bool:
        """Check if relationship between object types is valid based on OCEL model"""
        # Get allowed relationships from OCEL model
        allowed_relationships = self.data_manager.ocel_model.get(source_type, {}).get('relationships', [])
        return target_type in allowed_relationships

    def _get_single_relationship_types(self) -> Set[str]:
        """Get object types that should only have single relationship per event"""
        # Objects that should have 1:1 relationships
        single_relation_types = {
            'Trade',  # One trade per event
            'Position',  # One position per event
            'Market'  # One market data point per event
        }
        return single_relation_types

    def _check_relationship_violations(self, obj_type: str) -> List[Dict]:
        """Check for invalid relationships"""
        violations = []

        for _, event in self.events.iterrows():
            objects = event.get('ocel:objects', [])
            current_type_objects = [obj for obj in objects if obj.get('type') == obj_type]
            other_type_objects = [obj for obj in objects if obj.get('type') != obj_type]

            # Check for invalid combinations
            for curr_obj in current_type_objects:
                for other_obj in other_type_objects:
                    if not self._is_valid_relationship(obj_type, other_obj.get('type')):
                        violations.append({
                            'description': f"Invalid relationship between {obj_type} and {other_obj.get('type')}",
                            'objects': [curr_obj['id'], other_obj['id']]
                        })

            # Check for multiple relationships where single is required
            if len(current_type_objects) > 1 and obj_type in self._get_single_relationship_types():
                violations.append({
                    'description': f"Multiple {obj_type} objects in single event",
                    'objects': [obj['id'] for obj in current_type_objects]
                })

        return violations

    def _identify_object_failures(self) -> List[Dict]:
        """Identify object-level failures"""
        failures = []

        for obj_type in self.object_types:
            # Get expected and actual relationships
            expected_relationships = self.data_manager.get_expected_relationships(obj_type)
            actual_relationships = self._get_actual_relationships(obj_type)

            # Missing relationships create multiple failures - one per missing relationship
            missing = expected_relationships - actual_relationships
            for missing_rel in missing:
                failures.append({
                    'id': f"OF_R_{len(failures)}",
                    'activity': f"{obj_type} Object Analysis",
                    'object_type': obj_type,
                    'description': f"Missing mandatory relationship: {missing_rel}",
                    'violation_type': 'missing_relationship',
                    'severity': 8,
                    'likelihood': 7,
                    'detectability': 6,
                    'rpn': 336,
                    'affected_objects': [missing_rel]
                })

            # Check attribute violations - one failure per missing attribute
            attribute_violations = self._check_attribute_violations(obj_type)
            for attr in attribute_violations:
                failures.append({
                    'id': f"OF_A_{len(failures)}",
                    'activity': f"{obj_type} Object Analysis",
                    'object_type': obj_type,
                    'description': f"Missing mandatory attribute: {attr}",
                    'violation_type': 'missing_attribute',
                    'severity': 6,
                    'likelihood': 8,
                    'detectability': 4,
                    'rpn': 192,
                    'affected_objects': [obj_type]
                })

        return failures

    def _identify_event_failures(self) -> List[Dict]:
        """Identify event-level failures - one per case when sequence or timing violation occurs"""
        failures = []

        for case_id in self.events['case_id'].unique():
            case_events = self.events[self.events['case_id'] == case_id].sort_values('ocel:timestamp')

            # Check sequence violations
            for obj_type in self.object_types:
                expected_sequence = self.data_manager.get_expected_activities(obj_type)
                actual_sequence = case_events[
                    case_events['ocel:objects'].apply(lambda x:
                                                      any(obj.get('type') == obj_type for obj in x))
                ]['ocel:activity'].tolist()

                if expected_sequence and not self._validate_sequence(actual_sequence, expected_sequence):
                    failures.append({
                        'id': f"EF_S_{len(failures)}",
                        'activity': actual_sequence[0] if actual_sequence else 'Unknown',
                        'object_type': obj_type,
                        'description': f"Sequence violation in case {case_id}",
                        'violation_type': 'sequence',
                        'severity': 7,
                        'likelihood': 6,
                        'detectability': 5,
                        'rpn': 210,
                        'affected_objects': [obj_type]
                    })

            # Check timing violations for each event
            for _, event in case_events.iterrows():
                if self._check_timing_violation(event):
                    failures.append({
                        'id': f"EF_T_{len(failures)}",
                        'activity': event['ocel:activity'],
                        'object_type': self._get_primary_object_type(event['ocel:activity']),
                        'description': f"Timing violation in case {case_id}",
                        'violation_type': 'timing',
                        'severity': 8,
                        'likelihood': 7,
                        'detectability': 4,
                        'rpn': 224,
                        'affected_objects': [obj.get('type') for obj in event.get('ocel:objects', [])]
                    })

        return failures

    def _identify_system_failures(self) -> List[Dict]:
        """Identify system-wide failures"""
        failures = []

        # Check convergence points
        for cp in self.convergence_points:
            if len(cp['object_types']) > 3:  # Too many objects converging
                failures.append({
                    'id': f"SF_C_{len(failures)}",
                    'activity': cp['activity'],
                    'object_type': 'System',
                    'description': f"Complex convergence point with {len(cp['object_types'])} object types",
                    'violation_type': 'convergence',
                    'severity': 7,
                    'likelihood': 6,
                    'detectability': 5,
                    'rpn': 210
                })

        # Check divergence points
        for dp in self.divergence_points:
            # Get event details
            event = self.events[self.events['ocel:id'] == dp['event_id']].iloc[0]
            next_event = self.events[self.events['ocel:id'] == dp['next_event_id']].iloc[0]

            # Check for uncontrolled divergence
            if len(next_event.get('ocel:objects', [])) < len(event.get('ocel:objects', [])) / 2:
                failures.append({
                    'id': f"SF_D_{len(failures)}",
                    'activity': dp['activity'],
                    'object_type': 'System',
                    'description': f"Uncontrolled object divergence in {dp['activity']}",
                    'violation_type': 'divergence',
                    'severity': 8,
                    'likelihood': 7,
                    'detectability': 4,
                    'rpn': 224
                })

        return failures


def create_object_distribution(results: List[Dict]) -> None:
    """Create object type distribution visualization"""
    object_counts = defaultdict(int)
    for result in results:
        object_counts[result['object_type']] += 1

    fig = go.Figure(data=[
        go.Bar(
            x=list(object_counts.keys()),
            y=list(object_counts.values()),
            name='Object Distribution'
        )
    ])

    fig.update_layout(
        title='Failure Distribution by Object Type',
        xaxis_title='Object Type',
        yaxis_title='Number of Failures',
        showlegend=True
    )

    st.plotly_chart(fig)


def create_violation_distribution(results: List[Dict]) -> None:
    """Create violation type distribution visualization"""
    violation_counts = defaultdict(int)
    for result in results:
        violation_counts[result.get('violation_type', 'Unknown')] += 1

    fig = go.Figure(data=[
        go.Bar(
            x=list(violation_counts.keys()),
            y=list(violation_counts.values()),
            name='Violation Distribution'
        )
    ])

    fig.update_layout(
        title='Distribution by Violation Type',
        xaxis_title='Violation Type',
        yaxis_title='Count',
        showlegend=True
    )

    st.plotly_chart(fig)


def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply styling to dataframe with proper return type"""
    styled = df.style.background_gradient(
        subset=['RPN'],
        cmap='RdYlGn_r',
        vmin=0,
        vmax=1000
    ).format({
        'Severity': '{:.0f}',
        'Likelihood': '{:.0f}',
        'Detectability': '{:.0f}',
        'RPN': '{:.0f}'
    })

    # Return the styled DataFrame
    return styled


def create_relationship_matrix(results: List[Dict]) -> go.Figure:
    """Create relationship matrix visualization"""
    # Get unique object types
    object_types = list(set(r['object_type'] for r in results))
    matrix = np.zeros((len(object_types), len(object_types)))

    # Build relationship matrix
    for result in results:
        if 'affected_objects' in result:
            source_idx = object_types.index(result['object_type'])
            for affected in result['affected_objects']:
                if affected in object_types:
                    target_idx = object_types.index(affected)
                    matrix[source_idx][target_idx] += 1

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=object_types,
        y=object_types,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title='Object Relationship Matrix',
        xaxis_title='Target Object',
        yaxis_title='Source Object'
    )

    return fig


def create_cascade_chart(results: List[Dict]) -> go.Figure:
    """Create cascade effect visualization"""
    cascade_data = defaultdict(list)

    for result in results:
        if 'cascading_effects' in result:
            depth = len(result['cascading_effects'])
            cascade_data['depth'].append(depth)
            cascade_data['rpn'].append(result['rpn'])
            cascade_data['activity'].append(result['activity'])

    fig = go.Figure(data=go.Scatter(
        x=cascade_data['depth'],
        y=cascade_data['rpn'],
        mode='markers+text',
        text=cascade_data['activity'],
        marker=dict(
            size=10,
            color=cascade_data['rpn'],
            colorscale='Viridis',
            showscale=True
        )
    ))

    fig.update_layout(
        title='Failure Cascade Analysis',
        xaxis_title='Cascade Depth',
        yaxis_title='RPN',
        showlegend=False
    )

    return fig


def create_timeline_chart(results: List[Dict]) -> go.Figure:
    """Create timeline visualization of failures"""
    timeline_data = defaultdict(list)

    for result in results:
        if 'timestamp' in result:
            timeline_data['timestamp'].append(result['timestamp'])
            timeline_data['rpn'].append(result['rpn'])
            timeline_data['activity'].append(result['activity'])

    fig = go.Figure(data=go.Scatter(
        x=timeline_data['timestamp'],
        y=timeline_data['rpn'],
        mode='markers+text',
        text=timeline_data['activity'],
        marker=dict(
            size=10,
            color=timeline_data['rpn'],
            colorscale='Viridis',
            showscale=True
        )
    ))

    fig.update_layout(
        title='Failure Timeline',
        xaxis_title='Time',
        yaxis_title='RPN',
        showlegend=False
    )

    return fig


def display_object_recommendations(failures: List[Dict]) -> None:
    """Display recommendations for specific object type"""
    if not failures:
        st.info("No failures found for this object type")
        return

    # Group failures by violation category
    categories = {
        'Object-Level': [f for f in failures if 'relationship' in f.get('violation_type', '')],
        'Activity-Level': [f for f in failures if f.get('violation_type') == 'sequence'],
        'System-Level': [f for f in failures if f.get('violation_type') in ['convergence', 'divergence']]
    }

    # Create tabs for different categories
    tabs = st.tabs(list(categories.keys()))

    for tab, (category, items) in zip(tabs, categories.items()):
        with tab:
            if not items:
                st.info(f"No {category} failures detected")
                continue

            total_rpn = sum(f['rpn'] for f in items)
            avg_rpn = total_rpn / len(items) if items else 0
            st.metric("Average RPN", f"{avg_rpn:.2f}")

            for failure in sorted(items, key=lambda x: x['rpn'], reverse=True):
                with st.expander(f"{failure['activity']} (RPN: {failure['rpn']})"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Severity", failure['severity'])
                    with col2:
                        st.metric("Likelihood", failure['likelihood'])
                    with col3:
                        st.metric("Detectability", failure['detectability'])

                    st.write("**Description:**", failure['description'])
                    if 'affected_objects' in failure:
                        st.write("**Affected Objects:**", ', '.join(failure['affected_objects']))

def paginate_dataframe(df: pd.DataFrame, page_size: int = 1000) -> pd.DataFrame:
    """Handle pagination for large dataframes"""
    n_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)

    page = st.sidebar.number_input(
        'Page',
        min_value=1,
        max_value=n_pages,
        value=1
    )

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))

    return df.iloc[start_idx:end_idx].copy()


def display_rpn_distribution(fmea_results: List[Dict]):
    """
    Display comprehensive RPN distribution visualization with analysis breakdowns.
    This shows the spread of risk levels across different failure modes and helps
    identify risk clusters and patterns.
    """
    try:
        # Create base RPN histogram
        df = pd.DataFrame(fmea_results)
        fig = go.Figure()

        # Add main RPN distribution histogram
        fig.add_trace(go.Histogram(
            x=df['rpn'],
            nbinsx=20,
            name='RPN Distribution',
            marker_color='blue',
            opacity=0.7
        ))

        # Add critical threshold line
        fig.add_vline(
            x=200,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold (RPN=200)",
            annotation_position="top right"
        )

        # Add risk zone annotations
        fig.add_vrect(
            x0=0, x1=100,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Low Risk",
            annotation_position="bottom"
        )
        fig.add_vrect(
            x0=100, x1=200,
            fillcolor="yellow", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Medium Risk",
            annotation_position="bottom"
        )
        fig.add_vrect(
            x0=200, x1=1000,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="High Risk",
            annotation_position="bottom"
        )

        # Update layout with detailed information
        fig.update_layout(
            title={
                'text': 'Risk Priority Number (RPN) Distribution',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='RPN Value',
            yaxis_title='Number of Failure Modes',
            showlegend=True,
            height=500,
            # Add annotations for risk interpretation
            annotations=[
                dict(
                    x=50, y=1.05,
                    text="Low Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="green")
                ),
                dict(
                    x=150, y=1.05,
                    text="Medium Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="orange")
                ),
                dict(
                    x=250, y=1.05,
                    text="High Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="red")
                )
            ]
        )

        # Display the main distribution plot
        st.plotly_chart(fig, use_container_width=True)

        # Add distribution statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Average RPN",
                f"{df['rpn'].mean():.1f}",
                help="Average Risk Priority Number across all failure modes"
            )
        with col2:
            st.metric(
                "Median RPN",
                f"{df['rpn'].median():.1f}",
                help="Median Risk Priority Number (50th percentile)"
            )
        with col3:
            st.metric(
                "Std Deviation",
                f"{df['rpn'].std():.1f}",
                help="Standard deviation of RPN values"
            )
        with col4:
            st.metric(
                "90th Percentile",
                f"{df['rpn'].quantile(0.9):.1f}",
                help="90% of RPNs fall below this value"
            )

        # Add risk zone breakdown
        st.subheader("Risk Zone Analysis")
        risk_zones = {
            'Low Risk (RPN ≤ 100)': len(df[df['rpn'] <= 100]),
            'Medium Risk (100 < RPN ≤ 200)': len(df[(df['rpn'] > 100) & (df['rpn'] <= 200)]),
            'High Risk (RPN > 200)': len(df[df['rpn'] > 200])
        }

        # Create risk zone bar chart
        risk_fig = go.Figure(data=[
            go.Bar(
                x=list(risk_zones.keys()),
                y=list(risk_zones.values()),
                marker_color=['green', 'yellow', 'red']
            )
        ])
        risk_fig.update_layout(
            title="Distribution by Risk Zone",
            xaxis_title="Risk Zone",
            yaxis_title="Number of Failure Modes",
            height=400
        )
        st.plotly_chart(risk_fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error in display_rpn_distribution: {str(e)}")
        st.error("Error displaying RPN distribution")


def display_fmea_analysis(fmea_results: List[Dict]):
    """Display comprehensive FMEA analysis with categorized tabs"""
    try:
        @st.cache_data
        def get_summary_stats(results):
            return {
                'total': len(results),
                'high_risk': sum(1 for r in results if r['rpn'] > 200),
                'medium_risk': sum(1 for r in results if 100 < r['rpn'] <= 200),
                'low_risk': sum(1 for r in results if r['rpn'] <= 100)
            }

        stats = get_summary_stats(fmea_results)

        # Display summary metrics
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Failure Modes", stats['total'])
            with col2:
                st.metric("High Risk (RPN > 200)", stats['high_risk'])
            with col3:
                st.metric("Medium Risk", stats['medium_risk'])
            with col4:
                st.metric("Low Risk", stats['low_risk'])

        # Create main tabs
        main_tabs = st.tabs(["Overview", "Detailed Analysis", "Recommendations"])

        with main_tabs[0]:
            if len(fmea_results) > 0:
                display_rpn_distribution(fmea_results)

        with main_tabs[1]:
            # Create subtabs for different failure types
            detailed_tabs = st.tabs(["Object-Level", "Activity-Level", "System-Level"])

            # Object-Level Analysis
            with detailed_tabs[0]:
                object_failures = [r for r in fmea_results if r['object_type'] != 'System' and
                                   r.get('violation_type', '').startswith('missing_')]
                if object_failures:
                    for failure in sorted(object_failures, key=lambda x: x['rpn'], reverse=True):
                        with st.expander(
                                f"{failure['object_type']} - {failure['description']} (RPN: {failure['rpn']})"):
                            cols = st.columns(3)
                            cols[0].metric("Severity", failure['severity'])
                            cols[1].metric("Likelihood", failure['likelihood'])
                            cols[2].metric("Detectability", failure['detectability'])
                            if 'affected_objects' in failure:
                                st.write("**Affected Objects:**", ', '.join(failure['affected_objects']))
                else:
                    st.info("No object-level failures detected")

            # Activity-Level Analysis
            with detailed_tabs[1]:
                activity_failures = [r for r in fmea_results if r.get('violation_type') in ['sequence', 'timing'] and
                                     r['object_type'] != 'System']
                if activity_failures:
                    for failure in sorted(activity_failures, key=lambda x: x['rpn'], reverse=True):
                        with st.expander(f"{failure['activity']} (RPN: {failure['rpn']})"):
                            cols = st.columns(3)
                            cols[0].metric("Severity", failure['severity'])
                            cols[1].metric("Likelihood", failure['likelihood'])
                            cols[2].metric("Detectability", failure['detectability'])
                            st.write("**Violation Type:**", failure['violation_type'])
                else:
                    st.info("No activity-level failures detected")

            # System-Level Analysis
            with detailed_tabs[2]:
                system_failures = [r for r in fmea_results if r['object_type'] == 'System']
                if system_failures:
                    for failure in sorted(system_failures, key=lambda x: x['rpn'], reverse=True):
                        with st.expander(f"{failure['activity']} (RPN: {failure['rpn']})"):
                            cols = st.columns(3)
                            cols[0].metric("Severity", failure['severity'])
                            cols[1].metric("Likelihood", failure['likelihood'])
                            cols[2].metric("Detectability", failure['detectability'])
                            st.write("**System Impact:**", failure['description'])
                else:
                    st.info("No system-level failures detected")

        with main_tabs[2]:
            # Display recommendations as before
            priorities = {
                'High': [r for r in fmea_results if r['rpn'] > 200],
                'Medium': [r for r in fmea_results if 100 < r['rpn'] <= 200],
                'Low': [r for r in fmea_results if r['rpn'] <= 100]
            }

            for priority, items in priorities.items():
                if items:
                    with st.expander(f"{priority} Priority Items ({len(items)})"):
                        for item in items[:5]:
                            st.write(f"**{item['activity']}** (RPN: {item['rpn']})")
                            st.write(item['description'])
                        if len(items) > 5:
                            st.write(f"... and {len(items) - 5} more items")

        # Add failure mode type breakdown
        st.subheader("Failure Mode Analysis Breakdown")
        breakdown_cols = st.columns(3)
        with breakdown_cols[0]:
            st.metric("Object-Level Failures",
                      len([r for r in fmea_results if 'Object Analysis' in r.get('activity', '')]))
        with breakdown_cols[1]:
            st.metric("Activity-Level Failures",
                      len([r for r in fmea_results if r.get('violation_type') in ['sequence', 'timing']]))
        with breakdown_cols[2]:
            st.metric("System-Level Failures",
                      len([r for r in fmea_results if r.get('violation_type') in ['convergence', 'divergence']]))

    except Exception as e:
        logger.error(f"Error in display_fmea_analysis: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying analysis: {str(e)}")


def display_summary_metrics(results: List[Dict]):
    """Display summary metrics with enhanced visualizations"""
    col1, col2, col3, col4 = st.columns(4)

    # Basic metrics
    with col1:
        st.metric("Total Failure Modes", len(results))
    with col2:
        high_risk = sum(1 for r in results if r['rpn'] > 200)
        st.metric("High Risk (RPN > 200)", high_risk)
    with col3:
        medium_risk = sum(1 for r in results if 100 < r['rpn'] <= 200)
        st.metric("Medium Risk", medium_risk)
    with col4:
        low_risk = sum(1 for r in results if r['rpn'] <= 100)
        st.metric("Low Risk", low_risk)

    # RPN Distribution
    create_rpn_distribution(results)

    # Object Type Distribution
    create_object_distribution(results)

    # Violation Type Distribution
    create_violation_distribution(results)


def display_detailed_analysis(results: List[Dict]):
    """Display detailed analysis with filtering and sorting"""
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_types = st.multiselect(
            "Filter by Object Type",
            options=list(set(r['object_type'] for r in results))
        )

    with col2:
        selected_violations = st.multiselect(
            "Filter by Violation Type",
            options=list(set(r.get('violation_type', 'Unknown') for r in results))
        )

    with col3:
        rpn_threshold = st.slider("RPN Threshold", 0, 1000, 100)

    # Filter results
    filtered_results = results
    if selected_types:
        filtered_results = [r for r in filtered_results if r['object_type'] in selected_types]
    if selected_violations:
        filtered_results = [r for r in filtered_results if r.get('violation_type') in selected_violations]
    filtered_results = [r for r in filtered_results if r['rpn'] >= rpn_threshold]

    # Display table
    display_df = pd.DataFrame([{
        'ID': r['id'],
        'Object Type': r['object_type'],
        'Activity': r['activity'],
        'Description': r['description'],
        'Violation Type': r.get('violation_type', 'Unknown'),
        'Severity': r['severity'],
        'Likelihood': r['likelihood'],
        'Detectability': r['detectability'],
        'RPN': r['rpn']
    } for r in filtered_results])

    # Apply styling
    styled_df = style_dataframe(display_df)
    st.dataframe(styled_df)


def display_object_interactions(results: List[Dict]):
    """Display object interaction analysis"""
    # Create network graph
    object_network = create_object_network(results)

    # Display network
    st.plotly_chart(object_network)

    # Object relationship matrix
    st.subheader("Object Relationship Matrix")
    relationship_matrix = create_relationship_matrix(results)
    st.plotly_chart(relationship_matrix)

    # Cascade analysis
    st.subheader("Failure Cascade Analysis")
    cascade_chart = create_cascade_chart(results)
    st.plotly_chart(cascade_chart)


def display_system_impact(results: List[Dict]):
    """Display system-wide impact analysis"""
    # System metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Convergence Points")
        convergence_df = pd.DataFrame([r for r in results if r.get('violation_type') == 'convergence'])
        if not convergence_df.empty:
            st.dataframe(convergence_df)

    with col2:
        st.subheader("Divergence Points")
        divergence_df = pd.DataFrame([r for r in results if r.get('violation_type') == 'divergence'])
        if not divergence_df.empty:
            st.dataframe(divergence_df)

    # Timeline analysis
    st.subheader("Failure Timeline")
    timeline_chart = create_timeline_chart(results)
    st.plotly_chart(timeline_chart)


def display_recommendations(results: List[Dict]):
    """Display recommendations and mitigation strategies"""
    # High-risk items
    st.subheader("Critical Risk Recommendations")
    high_risk_items = [r for r in results if r['rpn'] > 200]

    if high_risk_items:
        for item in high_risk_items:
            with st.expander(f"{item['activity']} (RPN: {item['rpn']})"):
                st.write("**Description:**", item['description'])

                cols = st.columns(3)
                with cols[0]:
                    st.write(f"**Severity:** {item['severity']}")
                with cols[1]:
                    st.write(f"**Likelihood:** {item['likelihood']}")
                with cols[2]:
                    st.write(f"**Detectability:** {item['detectability']}")

                # Detailed impact analysis
                st.write("**Impact Analysis:**")
                if 'cascading_effects' in item:
                    st.write("Cascading Effects:")
                    for effect in item['cascading_effects']:
                        st.write(f"- {effect}")

                # Root causes
                if 'root_causes' in item:
                    st.write("**Root Causes:**")
                    for cause in item['root_causes']:
                        st.write(f"- {cause}")

                # Mitigation strategies
                if 'mitigation_strategies' in item:
                    st.write("**Recommended Actions:**")
                    for strategy in item['mitigation_strategies']:
                        st.write(f"- {strategy}")

                # Monitoring points
                if 'monitoring_points' in item:
                    st.write("**Monitoring Points:**")
                    for point in item['monitoring_points']:
                        st.write(f"- {point}")
    else:
        st.info("No high-risk failure modes identified")

    # Object-specific recommendations
    st.subheader("Object-Type Recommendations")
    for obj_type in set(r['object_type'] for r in results):
        with st.expander(f"Recommendations for {obj_type}"):
            type_failures = [r for r in results if r['object_type'] == obj_type]
            display_object_recommendations(type_failures)


def create_rpn_distribution(results: List[Dict]) -> None:
    """Create RPN distribution visualization"""
    fig = go.Figure(data=go.Histogram(
        x=[r['rpn'] for r in results],
        nbinsx=20,
        name='RPN Distribution'
    ))

    fig.add_vline(
        x=200,
        line_dash="dash",
        line_color="red",
        annotation_text="Critical Threshold"
    )

    fig.update_layout(
        title='RPN Distribution',
        xaxis_title='RPN',
        yaxis_title='Count',
        showlegend=True
    )

    st.plotly_chart(fig)


def create_object_network(results: List[Dict]) -> go.Figure:
    """Create network visualization of object interactions"""
    # Implementation of network graph visualization
    nodes = []
    edges = []

    # Process results to create network
    object_types = set()
    relationships = set()

    for result in results:
        object_types.add(result['object_type'])
        if 'affected_objects' in result:
            for obj in result['affected_objects']:
                relationships.add((result['object_type'], obj))

    # Create nodes
    for obj_type in object_types:
        nodes.append(
            dict(
                id=obj_type,
                label=obj_type,
                size=len([r for r in results if r['object_type'] == obj_type])
            )
        )

    # Create edges
    for source, target in relationships:
        edges.append(
            dict(
                source=source,
                target=target,
                value=len([r for r in results
                           if r['object_type'] == source and target in r.get('affected_objects', [])])
            )
        )

    # Create network visualization
    fig = go.Figure(data=[
        go.Scatter(
            x=[node['id'] for node in nodes],
            y=[node['size'] for node in nodes],
            mode='markers+text',
            text=[node['label'] for node in nodes],
            marker=dict(size=[node['size'] * 5 for node in nodes]),
            name='Objects'
        )
    ])

    for edge in edges:
        fig.add_trace(
            go.Scatter(
                x=[edge['source'], edge['target']],
                y=[nodes[i]['size'] for i in range(2)],
                mode='lines',
                line=dict(width=edge['value']),
                showlegend=False
            )
        )

    fig.update_layout(
        title='Object Interaction Network',
        showlegend=True
    )

    return fig


# Page execution
st.set_page_config(page_title="FMEA Analysis", layout="wide")
st.title("FMEA Analysis")

# Modified main execution code for 5_FMEA_Analysis.py
try:
    # First verify OCEL model file exists
    if not os.path.exists('ocpm_output/output_ocel.json'):
        st.error("OCEL model file (output_ocel.json) not found. Please ensure the file exists.")
        logger.error("Missing required OCEL model file")
        raise FileNotFoundError("output_ocel.json not found")

    # Load process data
    with open("ocpm_output/process_data.json", 'r') as f:
        ocel_data = json.load(f)

    # Validate OCEL data structure
    if 'ocel:events' not in ocel_data:
        st.error("Invalid OCEL data structure - missing events")
        logger.error("Invalid OCEL data structure")
        raise ValueError("Invalid OCEL data structure")

    # Initialize analyzer with validated data
    analyzer = OCELEnhancedFMEA(ocel_data)

    # Track analysis progress
    with st.spinner('Performing FMEA analysis...'):
        fmea_results = analyzer.identify_failure_modes()

        # Log analysis summary
        logger.info(f"Analysis complete. Found {len(fmea_results)} failure modes")
        logger.info(
            f"Using relationships from OCEL model with {len(analyzer.data_manager.object_relationships)} object types")

    # Display results with additional context
    st.success(f"Analysis complete - identified {len(fmea_results)} failure modes")
    display_fmea_analysis(fmea_results)

    if st.session_state.get('websocket_error'):
        st.warning("Previous session ended unexpectedly. Data has been refreshed.")
        st.session_state.websocket_error = False

except tornado.websocket.WebSocketClosedError:
    st.session_state.websocket_error = True
    st.error("Connection lost. Please refresh the page.")
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    st.error(f"An unexpected error occurred: {str(e)}")

except FileNotFoundError as e:
    st.error(f"File error: {str(e)}")
    logger.error(f"File error in FMEA analysis: {str(e)}")
except ValueError as e:
    st.error(f"Data validation error: {str(e)}")
    logger.error(f"Validation error in FMEA analysis: {str(e)}")
except Exception as e:
    st.error(f"Error in FMEA analysis: {str(e)}")
    logger.error(f"FMEA analysis error: {str(e)}", exc_info=True)