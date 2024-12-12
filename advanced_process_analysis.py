"""Advanced process mining analysis module"""
import pm4py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from pm4py.statistics.traces.generic.log import case_statistics
import logging

logger = logging.getLogger(__name__)


class AdvancedProcessAnalyzer:
    """Advanced analysis capabilities for process mining results"""

    def __init__(self, event_log):
        """Initialize analyzer with event log data"""
        self.event_log = event_log
        self.case_attributes = {}
        self.activity_pairs = []
        self._initialize_analysis()

    def _initialize_analysis(self):
        """Initialize analysis data structures"""
        self.case_attributes = pm4py.get_event_attributes(self.event_log)
        self.activity_pairs = self._get_activity_pairs()

    def _get_activity_pairs(self) -> List[tuple]:
        """Get all pairs of consecutive activities"""
        pairs = set()
        for trace in self.event_log:
            for i in range(len(trace) - 1):
                pairs.add((trace[i]['concept:name'], trace[i + 1]['concept:name']))
        return list(pairs)

    def detailed_conformance_analysis(self) -> Dict:
        """Perform detailed conformance analysis"""
        try:
            variants_count = case_statistics.get_variant_statistics(self.event_log)
            deviations = self._analyze_process_deviations()
            rework_stats = self._calculate_rework()

            return {
                'variants': variants_count,
                'deviations': deviations,
                'rework_statistics': rework_stats,
                'conformance_metrics': self._calculate_conformance_metrics()
            }
        except Exception as e:
            logger.error(f"Error in conformance analysis: {str(e)}")
            return {}

    def _calculate_rework(self) -> Dict:
        """Calculate rework statistics manually"""
        rework_stats = defaultdict(int)
        for trace in self.event_log:
            activity_counts = defaultdict(int)
            for event in trace:
                activity = event['concept:name']
                activity_counts[activity] += 1
                if activity_counts[activity] > 1:
                    rework_stats[activity] += 1
        return dict(rework_stats)

    def _analyze_process_deviations(self) -> List[Dict]:
        """Analyze process deviations"""
        deviations = []
        expected_sequence = self._get_expected_sequence()

        if not expected_sequence:
            return deviations

        for trace in self.event_log:
            trace_activities = [event['concept:name'] for event in trace]
            if trace_activities != expected_sequence:
                deviation = {
                    'case_id': trace.attributes.get('concept:name', 'Unknown'),
                    'expected': expected_sequence,
                    'actual': trace_activities,
                    'deviation_points': self._find_deviation_points(expected_sequence, trace_activities)
                }
                deviations.append(deviation)

        return deviations

    def _find_deviation_points(self, expected: List[str], actual: List[str]) -> List[Dict]:
        """Find points where actual sequence deviates from expected"""
        deviations = []
        for i in range(min(len(expected), len(actual))):
            if expected[i] != actual[i]:
                deviations.append({
                    'position': i,
                    'expected': expected[i],
                    'actual': actual[i],
                    'type': 'mismatch'
                })

        if len(actual) > len(expected):
            for i in range(len(expected), len(actual)):
                deviations.append({
                    'position': i,
                    'expected': None,
                    'actual': actual[i],
                    'type': 'additional'
                })
        elif len(expected) > len(actual):
            for i in range(len(actual), len(expected)):
                deviations.append({
                    'position': i,
                    'expected': expected[i],
                    'actual': None,
                    'type': 'missing'
                })
        return deviations

    def _get_expected_sequence(self) -> List[str]:
        """Get the most common activity sequence"""
        variants = case_statistics.get_variant_statistics(self.event_log)
        if variants and len(variants) > 0:
            most_frequent = variants[0]
            if isinstance(most_frequent['variant'], tuple):
                return list(most_frequent['variant'])
            elif isinstance(most_frequent['variant'], list):
                return most_frequent['variant']
            else:
                try:
                    return [act.strip() for act in most_frequent['variant'].split(',')]
                except (AttributeError, TypeError):
                    return []
        return []

    def _calculate_conformance_metrics(self) -> Dict:
        """Calculate conformance metrics"""
        try:
            df = pm4py.convert_to_dataframe(self.event_log)
            follows_relations = self._calculate_eventually_follows()

            return {
                'trace_statistics': self._calculate_trace_statistics(df),
                'follow_relations': follows_relations
            }
        except Exception as e:
            logger.error(f"Error calculating conformance metrics: {str(e)}")
            return {}

    def _calculate_trace_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate trace-level statistics"""
        try:
            case_durations = df.groupby('case:concept:name').agg({
                'time:timestamp': lambda x: (max(x) - min(x)).total_seconds()
            })
            return {
                'mean_duration': case_durations['time:timestamp'].mean(),
                'median_duration': case_durations['time:timestamp'].median(),
                'min_duration': case_durations['time:timestamp'].min(),
                'max_duration': case_durations['time:timestamp'].max()
            }
        except Exception as e:
            logger.error(f"Error calculating trace statistics: {str(e)}")
            return {}

    def _calculate_eventually_follows(self) -> Dict:
        """Calculate eventually follows relations"""
        follows = defaultdict(int)
        for trace in self.event_log:
            activities = [event['concept:name'] for event in trace]
            for i, act1 in enumerate(activities):
                for act2 in activities[i + 1:]:
                    follows[(act1, act2)] += 1
        return dict(follows)

    def detailed_performance_analysis(self) -> Dict:
        """Perform detailed performance analysis"""
        try:
            case_durations = case_statistics.get_all_case_durations(self.event_log)
            activity_durations = self._analyze_activity_durations()
            concurrent_activities = self._analyze_concurrent_activities()

            return {
                'activity_durations': activity_durations,
                'case_durations': {
                    'mean': np.mean(case_durations) if case_durations else 0,
                    'median': np.median(case_durations) if case_durations else 0,
                    'min': min(case_durations) if case_durations else 0,
                    'max': max(case_durations) if case_durations else 0
                },
                'concurrent_activities': concurrent_activities,
                'resource_utilization': self._analyze_resource_utilization(),
                'throughput_analysis': self._analyze_throughput()
            }
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return {}

    def _analyze_activity_durations(self) -> Dict:
        """Analyze duration of each activity"""
        durations = {}
        try:
            activities = pm4py.get_event_attribute_values(self.event_log, "concept:name")
            for activity in activities:
                filtered_log = pm4py.filter_event_attribute_values(
                    self.event_log, "concept:name", [activity], level="event"
                )
                if filtered_log:
                    activity_durations = []
                    for trace in filtered_log:
                        for event in trace:
                            if 'time:timestamp' in event:
                                activity_durations.append(
                                    event['time:timestamp'].timestamp()
                                )
                    if activity_durations:
                        durations[activity] = {
                            'mean': np.mean(activity_durations),
                            'median': np.median(activity_durations),
                            'std': np.std(activity_durations)
                        }
        except Exception as e:
            logger.error(f"Error analyzing activity durations: {str(e)}")
        return durations

    def _analyze_concurrent_activities(self) -> Dict:
        """Analyze activities that occur concurrently"""
        concurrent_pairs = defaultdict(int)
        try:
            for trace in self.event_log:
                activities_in_trace = set()
                for event in trace:
                    current_activity = event['concept:name']
                    timestamp = event['time:timestamp']

                    for other_event in trace:
                        if other_event != event:
                            other_activity = other_event['concept:name']
                            other_timestamp = other_event['time:timestamp']
                            time_diff = abs((timestamp - other_timestamp).total_seconds())

                            if time_diff <= 60:  # 1 minute window
                                pair = tuple(sorted([current_activity, other_activity]))
                                concurrent_pairs[pair] += 1

            return {
                'concurrent_pairs': dict(concurrent_pairs),
                'total_concurrent_instances': sum(concurrent_pairs.values()),
                'most_common_concurrent': sorted(
                    concurrent_pairs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        except Exception as e:
            logger.error(f"Error analyzing concurrent activities: {str(e)}")
            return {}

    def _analyze_resource_utilization(self) -> Dict:
        """Analyze resource utilization"""
        resource_stats = {}
        try:
            resources = pm4py.get_event_attribute_values(self.event_log, "org:resource")
            for resource in resources:
                filtered_log = pm4py.filter_event_attribute_values(
                    self.event_log, "org:resource", [resource], level="event"
                )
                if filtered_log:
                    resource_stats[resource] = {
                        'total_activities': len(filtered_log),
                        'unique_activities': len(set(event['concept:name']
                                                     for trace in filtered_log for event in trace)),
                        'avg_duration': np.mean([len(trace) for trace in filtered_log])
                    }
        except Exception as e:
            logger.error(f"Error analyzing resource utilization: {str(e)}")
        return resource_stats

    def _analyze_throughput(self) -> Dict:
        """Analyze process throughput"""
        try:
            case_durations = case_statistics.get_all_case_durations(self.event_log)
            return {
                'avg_throughput': np.mean(case_durations) if case_durations else 0,
                'median_throughput': np.median(case_durations) if case_durations else 0,
                'min_throughput': min(case_durations) if case_durations else 0,
                'max_throughput': max(case_durations) if case_durations else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing throughput: {str(e)}")
            return {}

    def comprehensive_root_cause_analysis(self) -> Dict:
        """Perform comprehensive root cause analysis"""
        try:
            return {
                'attribute_correlation': self._analyze_attribute_correlation(),
                'performance_factors': self._analyze_performance_factors(),
                'deviation_patterns': self._analyze_deviation_patterns()
            }
        except Exception as e:
            logger.error(f"Error in root cause analysis: {str(e)}")
            return {}

    def _analyze_attribute_correlation(self) -> Dict:
        """Analyze correlation between attributes and performance"""
        correlations = {}
        for attribute in self.case_attributes:
            if attribute not in ['concept:name', 'time:timestamp']:
                correlation = self._calculate_attribute_correlation(attribute)
                if correlation:
                    correlations[attribute] = correlation
        return correlations

    def _calculate_attribute_correlation(self, attribute: str) -> Optional[Dict]:
        """Calculate correlation between attribute and case duration"""
        try:
            values = []
            durations = []

            for trace in self.event_log:
                if any(attribute in event for event in trace):
                    attr_value = next(event[attribute] for event in trace if attribute in event)
                    values.append(attr_value)
                    duration = (trace[-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds()
                    durations.append(duration)

            if values and durations and len(values) == len(durations):
                if isinstance(values[0], (int, float)):
                    correlation = np.corrcoef(values, durations)[0, 1]
                    return {
                        'correlation_coefficient': correlation,
                        'impact_level': 'high' if abs(correlation) > 0.7 else
                        'medium' if abs(correlation) > 0.4 else 'low'
                    }
                else:
                    categorical_analysis = pd.DataFrame({'value': values, 'duration': durations})
                    return {
                        'category_analysis': categorical_analysis.groupby('value')['duration'].mean().to_dict()
                    }
        except Exception as e:
            logger.error(f"Error calculating attribute correlation: {str(e)}")
            return None

    def _analyze_performance_factors(self) -> Dict:
        """Analyze factors affecting performance"""
        return {
            'rework_impact': self._analyze_rework_impact(),
            'time_impact': self._analyze_time_impact()
        }

    def _analyze_rework_impact(self) -> Dict:
        """Analyze impact of rework on process performance"""
        try:
            df = pm4py.convert_to_dataframe(self.event_log)
            return {
                'rework_activities': self._get_rework_activities(df),
                'impact_on_duration': self._calculate_rework_duration_impact()
            }
        except Exception as e:
            logger.error(f"Error analyzing rework impact: {str(e)}")
            return {}

    def _get_rework_activities(self, df: pd.DataFrame) -> Dict:
        """Get activities with rework"""
        return df.groupby('concept:name').filter(lambda x: len(x) > 1).groupby('concept:name').size().to_dict()

    def _calculate_rework_duration_impact(self) -> Dict:
        """Calculate how rework affects case duration"""
        try:
            cases_with_rework = []
            cases_without_rework = []

            for trace in self.event_log:
                duration = (trace[-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds()
                activities = defaultdict(int)

                for event in trace:
                    activities[event['concept:name']] += 1

                if any(count > 1 for count in activities.values()):
                    cases_with_rework.append(duration)
                else:
                    cases_without_rework.append(duration)

            return {
                'avg_duration_with_rework': np.mean(cases_with_rework) if cases_with_rework else 0,
                'avg_duration_without_rework': np.mean(cases_without_rework) if cases_without_rework else 0,
                'rework_impact_percentage': ((np.mean(cases_with_rework) - np.mean(cases_without_rework)) /
                                             np.mean(
                                                 cases_without_rework) * 100) if cases_without_rework and cases_with_rework else 0
            }
        except Exception as e:
            logger.error(f"Error calculating rework duration impact: {str(e)}")
            return {}

    def _analyze_time_impact(self) -> Dict:
        """Analyze time-based patterns impact on performance"""
        try:
            time_patterns = defaultdict(list)

            for trace in self.event_log:
                start_time = trace[0]['time:timestamp']
                duration = (trace[-1]['time:timestamp'] - start_time).total_seconds()
                hour = start_time.hour
                time_patterns[hour].append(duration)

            hourly_impact = {hour: np.mean(durations) for hour, durations in time_patterns.items()}

            return {
                'hourly_performance': hourly_impact,
                'peak_hours': sorted(hourly_impact.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        except Exception as e:
            logger.error(f"Error analyzing time impact: {str(e)}")
            return {}

    def _analyze_deviation_patterns(self) -> Dict:
        """Analyze patterns in process deviations"""
        try:
            deviation_patterns = defaultdict(list)

            for trace in self.event_log:
                trace_activities = [event['concept:name'] for event in trace]
                expected_sequence = self._get_expected_sequence()

                if trace_activities != expected_sequence:
                    deviation_points = self._find_deviation_points(expected_sequence, trace_activities)
                    for point in deviation_points:
                        key = f"{point['expected']} -> {point['actual']}"
                        deviation_patterns[key].append(trace.attributes['concept:name'])

            return {
                'common_deviations': {k: len(v) for k, v in deviation_patterns.items()},
                'deviation_details': dict(deviation_patterns)
            }
        except Exception as e:
            logger.error(f"Error analyzing deviation patterns: {str(e)}")
            return {}

    def generate_process_improvements(self) -> Dict:
        """Generate detailed process improvement recommendations"""
        try:
            performance_analysis = self.detailed_performance_analysis()
            root_cause_analysis = self.comprehensive_root_cause_analysis()

            return {
                'bottleneck_solutions': self._generate_bottleneck_solutions(performance_analysis),
                'resource_optimization': self._generate_resource_recommendations(performance_analysis),
                'compliance_improvements': self._generate_compliance_recommendations(),
                'automation_opportunities': self._identify_automation_opportunities()
            }
        except Exception as e:
            logger.error(f"Error generating process improvements: {str(e)}")
            return {}

    def _generate_bottleneck_solutions(self, performance_analysis: Dict) -> List[Dict]:
        """Generate solutions for identified bottlenecks"""
        solutions = []
        try:
            activity_durations = performance_analysis.get('activity_durations', {})
            if not activity_durations:
                return solutions

            mean_durations = [m['mean'] for m in activity_durations.values()]
            if not mean_durations:
                return solutions

            avg_duration = np.mean(mean_durations)

            for activity, metrics in activity_durations.items():
                if metrics['mean'] > avg_duration:
                    solutions.append({
                        'activity': activity,
                        'issue': 'High duration variability' if metrics['std'] > metrics[
                            'mean'] else 'Consistent delays',
                        'recommendations': self._get_activity_recommendations(activity, metrics)
                    })
        except Exception as e:
            logger.error(f"Error generating bottleneck solutions: {str(e)}")

        return solutions

    def _generate_resource_recommendations(self, performance_analysis: Dict) -> List[Dict]:
        """Generate resource optimization recommendations"""
        recommendations = []
        try:
            resource_stats = performance_analysis.get('resource_utilization', {})
            if not resource_stats:
                return recommendations

            avg_activities = np.mean([stats['total_activities'] for stats in resource_stats.values()])

            for resource, stats in resource_stats.items():
                if stats['total_activities'] > avg_activities * 1.2:
                    recommendations.append({
                        'resource': resource,
                        'issue': 'High workload',
                        'recommendation': 'Consider redistributing work or adding support'
                    })
                elif stats['total_activities'] < avg_activities * 0.8:
                    recommendations.append({
                        'resource': resource,
                        'issue': 'Low utilization',
                        'recommendation': 'Consider increasing workload or reassigning tasks'
                    })
        except Exception as e:
            logger.error(f"Error generating resource recommendations: {str(e)}")

        return recommendations

    def _generate_compliance_recommendations(self) -> List[Dict]:
        """Generate compliance-related recommendations"""
        recommendations = []
        try:
            conformance = self.detailed_conformance_analysis()

            if conformance.get('deviations'):
                recommendations.append({
                    'issue': 'Process variations detected',
                    'recommendation': 'Implement stricter process controls and monitoring'
                })

            if conformance.get('rework_statistics'):
                recommendations.append({
                    'issue': 'Significant rework detected',
                    'recommendation': 'Review and optimize quality control procedures'
                })
        except Exception as e:
            logger.error(f"Error generating compliance recommendations: {str(e)}")

        return recommendations

    def _identify_automation_opportunities(self) -> List[Dict]:
        """Identify activities suitable for automation"""
        opportunities = []
        try:
            activity_stats = self._analyze_activity_durations()
            for activity, stats in activity_stats.items():
                automation_potential = self._evaluate_automation_potential(activity, stats)
                if automation_potential:
                    opportunities.append(automation_potential)
        except Exception as e:
            logger.error(f"Error identifying automation opportunities: {str(e)}")

        return opportunities

    def _evaluate_automation_potential(self, activity: str, stats: Dict) -> Optional[Dict]:
        """Evaluate if an activity is suitable for automation"""
        try:
            all_means = [s['mean'] for s in self._analyze_activity_durations().values()]
            avg_duration = np.mean(all_means) if all_means else 0

            low_variability = stats['std'] < stats['mean'] * 0.2
            high_frequency = stats['mean'] > avg_duration
            is_manual = any(term in activity.lower() for term in ['manual', 'check', 'validate', 'verify', 'input'])

            if low_variability or high_frequency or is_manual:
                potential = {
                    'activity': activity,
                    'factors': [],
                    'automation_potential': 'Unknown'
                }

                score = 0
                if low_variability:
                    score += 1
                    potential['factors'].append('Standardized process with low variability')
                if high_frequency:
                    score += 1
                    potential['factors'].append('High frequency activity')
                if is_manual:
                    score += 1
                    potential['factors'].append('Manual/repetitive task')

                potential['automation_potential'] = {
                    3: 'High',
                    2: 'Medium',
                    1: 'Low'
                }.get(score, 'Low')

                return potential

        except Exception as e:
            logger.error(f"Error evaluating automation potential for {activity}: {str(e)}")

        return None

    def _get_activity_recommendations(self, activity: str, metrics: Dict) -> List[str]:
        """Generate specific recommendations for an activity"""
        recommendations = []
        try:
            activity_stats = self._analyze_activity_durations()
            median_values = [m['median'] for m in activity_stats.values()]
            avg_median = np.mean(median_values) if median_values else 0

            if metrics['std'] > metrics['mean']:
                recommendations.append(f"Standardize {activity} process to reduce variability")
            if metrics['median'] > avg_median:
                recommendations.append(f"Review and optimize {activity} workflow")
            if 'manual' in activity.lower():
                recommendations.append(f"Consider automation opportunities for {activity}")
        except Exception as e:
            logger.error(f"Error generating activity recommendations for {activity}: {str(e)}")

        return recommendations