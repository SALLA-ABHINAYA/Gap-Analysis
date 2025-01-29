# process_mining_enhanced.py
import pandas as pd
import pm4py
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pm4py.algo.filtering.log import timestamp as timestamp_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from risk_analysis import ProcessRiskAnalyzer, EnhancedFMEA
from advanced_process_analysis import AdvancedProcessAnalyzer


class FXProcessMining:
    def __init__(self, event_log_path: str, separator: str = ';'):
        self.raw_df = pd.read_csv(event_log_path, sep=separator)
        self.event_log = None
        self.process_tree = None
        self.process_model = None  # This will hold the Petri net
        self.initial_marking = None
        self.final_marking = None
        self.activity_stats = None
        self.advanced_analyzer = None

    def preprocess_data(self) -> None:
        """Step 2: Data Preprocessing"""
        # Remove incomplete cases (missing mandatory fields)
        self.raw_df = self.raw_df.dropna(subset=['case_id', 'activity', 'timestamp'])

        # Standardize activity names
        self.raw_df['activity'] = self.raw_df['activity'].str.strip().str.title()

        # Convert timestamp to datetime
        self.raw_df['timestamp'] = pd.to_datetime(self.raw_df['timestamp'])

        # Sort by case_id and timestamp
        self.raw_df = self.raw_df.sort_values(['case_id', 'timestamp'])

        # Convert to PM4Py event log
        self.event_log = pm4py.format_dataframe(
            self.raw_df,
            case_id='case_id',
            activity_key='activity',
            timestamp_key='timestamp'
        )
        self.event_log = pm4py.convert_to_event_log(self.event_log)

        # Initialize advanced analyzer
        self.advanced_analyzer = AdvancedProcessAnalyzer(self.event_log)

    def discover_process(self) -> None:
        """Step 3: Process Discovery"""
        # First get the process tree
        self.process_tree = inductive_miner.apply(self.event_log)

        # Then convert to Petri net for analysis
        self.process_model, self.initial_marking, self.final_marking = \
            pm4py.convert_to_petri_net(self.process_tree)

    def check_conformance(self) -> Dict:
        """Step 4: Conformance Checking"""
        if not self.process_model:
            raise ValueError("Process model not discovered. Run discover_process() first.")

        # Token-based replay
        replayed_traces = token_replay.apply(
            self.event_log,
            self.process_model,
            self.initial_marking,
            self.final_marking
        )

        # Calculate conformance metrics
        conformance_metrics = {
            'fitness': np.mean([trace['trace_fitness'] for trace in replayed_traces]),
            'completed_traces': sum(1 for trace in replayed_traces if trace['trace_is_fit']),
            'total_traces': len(replayed_traces)
        }

        return conformance_metrics

    def _get_cases_with_sequence(self, act1: str, act2: str) -> List[Dict]:
        """Helper method to get cases containing a specific sequence of activities"""
        cases = []
        for trace in self.event_log:
            case_id = trace.attributes['concept:name']
            activities = [event['concept:name'] for event in trace]

            # Find all occurrences of act1 in the trace
            for i in range(len(activities)-1):
                if activities[i] == act1 and activities[i+1] == act2:
                    # Get timestamps
                    act1_time = trace[i]['time:timestamp']
                    act2_time = trace[i+1]['time:timestamp']
                    duration = act2_time - act1_time

                    cases.append({
                        'case_id': case_id,
                        'duration': duration
                    })
                    break  # Only consider first occurrence in each case

        return cases

    def analyze_risks(self) -> Dict:
        """Analyze process risks using ProcessRiskAnalyzer and EnhancedFMEA"""
        if not self.process_tree:
            raise ValueError("Process model not discovered. Run discover_process() first.")

        # Convert process tree to BPMN for risk analysis
        bpmn_graph = pm4py.convert_to_bpmn(self.process_tree)

        # Initialize ProcessRiskAnalyzer
        risk_analyzer = ProcessRiskAnalyzer(
            event_log=self.event_log,
            bpmn_graph=bpmn_graph
        )

        # Analyze BPMN graph for failure modes
        risk_analyzer.analyze_bpmn_graph()

        # Get activity statistics
        self.activity_stats = risk_analyzer.activity_stats

        # Initialize EnhancedFMEA with failure modes and statistics
        fmea = EnhancedFMEA(
            failure_modes=risk_analyzer.failure_modes,
            activity_stats=self.activity_stats
        )

        # Perform FMEA analysis
        risk_assessment = fmea.assess_risk()

        # Calculate process metrics
        process_metrics = {
            'total_activities': len(self.activity_stats),
            'high_risk_activities': len([r for r in risk_assessment if r['rpn'] > 200]),
            'medium_risk_activities': len([r for r in risk_assessment if 100 < r['rpn'] <= 200]),
            'low_risk_activities': len([r for r in risk_assessment if r['rpn'] <= 100])
        }

        return {
            'risk_assessment': risk_assessment,
            'process_metrics': process_metrics
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive process mining report including advanced analysis"""
        if not self.advanced_analyzer:
            raise ValueError("Advanced analyzer not initialized. Run preprocess_data() first.")

        try:
            # Calculate conformance using token-based replay
            token_replay_result = token_replay.apply(
                self.event_log,
                self.process_model,
                self.initial_marking,
                self.final_marking
            )

            # Calculate fitness
            fitness = sum(case['trace_fitness'] for case in token_replay_result) / len(
                token_replay_result) if token_replay_result else 0

            # Get variants statistics
            variants = case_statistics.get_variant_statistics(self.event_log)
            completed_traces = len(
                [case for case in token_replay_result if case['trace_is_fit']]) if token_replay_result else 0

            # Get activity statistics
            start_activities = pm4py.get_start_activities(self.event_log)
            end_activities = pm4py.get_end_activities(self.event_log)

            # Get detailed analyses
            detailed_conformance = self.advanced_analyzer.detailed_conformance_analysis()
            detailed_performance = self.advanced_analyzer.detailed_performance_analysis()
            root_cause_results = self.advanced_analyzer.comprehensive_root_cause_analysis()
            improvement_suggestions = self.advanced_analyzer.generate_process_improvements()

            # Get case statistics
            case_stats = case_statistics.get_cases_description(self.event_log)

            # Calculate average case duration
            durations = [case['caseDuration'] for case in case_stats.values()]
            avg_case_duration = sum(durations) / len(durations) if durations else 0
            median_case_duration = sorted(durations)[len(durations) // 2] if durations else 0

            # Get risk analysis
            try:
                risk_results = self.analyze_risks()
            except Exception as e:
                risk_results = {"error": str(e)}

            # Structure the complete report
            report = {
                'process_statistics': {
                    'start_activities': dict(start_activities),
                    'end_activities': dict(end_activities),
                    'variants': variants,
                    'total_cases': len(self.event_log),
                    'total_events': sum(len(trace) for trace in self.event_log),
                    'case_statistics': case_stats
                },
                'conformance': {
                    'fitness': fitness,
                    'completed_traces': completed_traces,
                    'total_traces': len(self.event_log),
                    'detailed_metrics': detailed_conformance
                },
                'performance': {
                    'avg_case_duration': avg_case_duration,
                    'median_case_duration': median_case_duration,
                    'bottlenecks': [
                        {
                            'source': activity,
                            'target': 'End',
                            'avg_duration': stats.get('mean', 0),
                            'frequency': stats.get('count', 0)
                        }
                        for activity, stats in detailed_performance.get('activity_durations', {}).items()
                    ] if detailed_performance.get('activity_durations') else []
                },
                'root_causes': {
                    'variant_analysis': [
                        {'activity': act, 'count': count}
                        for act, count in start_activities.items()
                    ],
                    'attribute_impact': root_cause_results.get('attribute_correlation', {}),
                    'temporal_patterns': detailed_performance.get('throughput_analysis', {})
                },
                'improvements': [
                    {
                        'type': 'performance',
                        'issue': sugg.get('issue', ''),
                        'recommendation': sugg.get('recommendation', '')
                    }
                    for sugg in improvement_suggestions.get('bottleneck_solutions', [])
                ],
                'risks': risk_results
            }

            return report

        except Exception as e:
            import traceback
            print(f"Error generating report: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            # Return a minimal valid report structure even in case of error
            return {
                'process_statistics': {
                    'start_activities': {},
                    'end_activities': {},
                    'variants': [],
                    'total_cases': 0,
                    'total_events': 0,
                    'case_statistics': {}
                },
                'conformance': {
                    'fitness': 0,
                    'completed_traces': 0,
                    'total_traces': 0,
                    'detailed_metrics': {}
                },
                'performance': {
                    'avg_case_duration': 0,
                    'median_case_duration': 0,
                    'bottlenecks': []
                },
                'root_causes': {
                    'variant_analysis': [],
                    'attribute_impact': {},
                    'temporal_patterns': {}
                },
                'improvements': [],
                'risks': {'error': str(e)}
            }

    def analyze_performance(self) -> Dict:
        """Step 5: Performance Analysis"""
        if not self.advanced_analyzer:
            raise ValueError("Advanced analyzer not initialized. Run preprocess_data() first.")

        return self.advanced_analyzer.detailed_performance_analysis()

    def _identify_bottlenecks(self, dfg) -> List[Dict]:
        """Helper method to identify process bottlenecks"""
        bottlenecks = []
        avg_time_between_activities = {}

        for (act1, act2), freq in dfg.items():
            # Calculate average time between activities
            relevant_cases = self._get_cases_with_sequence(act1, act2)
            if relevant_cases:
                avg_time = np.mean([case['duration'].total_seconds()
                                    for case in relevant_cases])
                avg_time_between_activities[(act1, act2)] = avg_time

                if avg_time > np.median(list(avg_time_between_activities.values())):
                    bottlenecks.append({
                        'source': act1,
                        'target': act2,
                        'avg_duration': avg_time,
                        'frequency': freq
                    })

        return sorted(bottlenecks, key=lambda x: x['avg_duration'], reverse=True)

    def perform_root_cause_analysis(self) -> Dict:
        """Step 6: Root Cause Analysis"""
        if not self.advanced_analyzer:
            raise ValueError("Advanced analyzer not initialized. Run preprocess_data() first.")

        return self.advanced_analyzer.comprehensive_root_cause_analysis()

    def _analyze_variants(self) -> List[Dict]:
        """Analyze process variants"""
        from pm4py.statistics.traces.generic.log import case_statistics

        variants = case_statistics.get_variant_statistics(self.event_log)
        return sorted(variants, key=lambda x: x['count'], reverse=True)

    def _analyze_attribute_impact(self) -> Dict:
        """Analyze impact of case attributes on performance"""
        impacts = {}
        for attribute in ['resource', 'currency_pair']:
            if attribute in self.raw_df.columns:
                attribute_stats = self.raw_df.groupby(attribute).agg({
                    'case_id': 'count',
                    'timestamp': lambda x: (x.max() - x.min()).total_seconds()
                }).reset_index()
                impacts[attribute] = attribute_stats.to_dict('records')
        return impacts

    def _analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns in the process"""
        self.raw_df['hour'] = self.raw_df['timestamp'].dt.hour
        self.raw_df['day_of_week'] = self.raw_df['timestamp'].dt.day_name()

        temporal_patterns = {
            'hourly_distribution': self.raw_df.groupby('hour')['case_id'].count().to_dict(),
            'daily_distribution': self.raw_df.groupby('day_of_week')['case_id'].count().to_dict()
        }
        return temporal_patterns

    def suggest_improvements(self) -> List[Dict]:
        """Step 7: Process Enhancement"""
        if not self.advanced_analyzer:
            raise ValueError("Advanced analyzer not initialized. Run preprocess_data() first.")

        return self.advanced_analyzer.generate_process_improvements()


# Usage example
if __name__ == "__main__":
    # Initialize process mining
    miner = FXProcessMining("staging/fx_trade_log.csv")

    # Execute full analysis
    miner.preprocess_data()
    miner.discover_process()

    # Generate report
    report = miner.generate_report()

    # Print key findings
    print("=== Process Mining Analysis Report ===")
    print(f"\nConformance Metrics:")
    print(f"Fitness: {report['conformance']['fitness']:.2f}")
    print(f"Completed Traces: {report['conformance']['completed_traces']}")

    print("\nTop Bottlenecks:")
    for bottleneck in report['performance']['bottlenecks'][:3]:
        print(f"- {bottleneck['source']} → {bottleneck['target']}: "
              f"{bottleneck['avg_duration']:.2f} seconds")

    print("\nSuggested Improvements:")
    for improvement in report['improvements']:
        print(f"\n{improvement['type'].title()}:")
        print(f"Issue: {improvement['issue']}")
        print(f"Recommendation: {improvement['recommendation']}")