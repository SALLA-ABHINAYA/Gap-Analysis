from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random
import pandas as pd
import logging
import json
from collections import Counter

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ActivitySequence:
    """Defines expected activity sequence and timing for an object type"""
    name: str
    activities: List[str]
    total_duration: float  # in hours
    activity_durations: Dict[str, float]  # max duration for each activity
    activity_gaps: Dict[str, float]  # max gap after each activity


class FXTradeGenerator:
    """Enhanced FX trade data generator with OCPM alignment"""

    def __init__(self):
        # Load OCPM model and thresholds
        with open('../ocpm_output/output_ocel.json', 'r') as f:
            self.ocpm_model = json.load(f)
        with open('../ocpm_output/output_ocel_threshold.json', 'r') as f:
            self.thresholds = json.load(f)

        # Configuration for synthetic data
        self.traders = [
            "Market Maker A", "Market Maker B", "Client Desk A",
            "Options Desk A", "Hedge Desk A", "Risk Desk A"
        ]
        self.currencies = [
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "EUR/GBP",
            "USD/CHF", "USD/CAD", "NZD/USD"
        ]
        self.client_types = [
            "Hedge Fund", "Corporate", "Bank", "Asset Manager",
            "Pension Fund", "Central Bank"
        ]
        self.booking_systems = ["Murex", "Summit", "Calypso", "Internal"]

    def _initialize_sequences(self) -> Dict[str, ActivitySequence]:
        """Initialize activity sequences from OCPM model"""
        sequences = {}
        for obj_type, config in self.ocpm_model.items():
            thresholds = self.thresholds[obj_type]
            sequences[obj_type] = ActivitySequence(
                name=obj_type,
                activities=config['activities'],
                total_duration=thresholds['total_duration_hours'],
                activity_durations={
                    act: thresholds['activity_thresholds'][act]['max_duration_hours']
                    for act in config['activities']
                },
                activity_gaps={
                    act: thresholds['activity_thresholds'][act]['max_gap_after_hours']
                    for act in config['activities']
                }
            )
        return sequences

    def generate_case_events(self, case_id: str, start_time: datetime) -> List[Dict]:
        """Generate events with debugging to ensure Final Settlement is included"""
        events = []
        current_time = start_time

        # Get and verify sequence
        trade_sequence = [
            "Trade Initiated",
            "Trade Validation",
            "Trade Execution",
            "Trade Confirmation",
            "Trade Matching",
            "Trade Reconciliation",
            "Trade Transparency Assessment",
            "Final Settlement"  # Must be included
        ]

        print(f"Generating case {case_id} with sequence: {trade_sequence}")  # Debug print

        characteristics = {
            'resource': random.choice(self.traders),
            'currency_pair': random.choice(self.currencies),
            'client_type': random.choice(self.client_types),
            'booking_system': random.choice(self.booking_systems),
            'notional_amount': random.randint(1000000, 50000000)
        }

        # Track what we're generating
        generated_activities = []

        for activity in trade_sequence:
            duration = random.uniform(0.1, 0.5)
            gap = random.uniform(0.1, 0.3)

            event = {
                'case_id': case_id,
                'activity': activity,
                'timestamp': current_time,
                'object_type': 'Trade',
                **characteristics
            }
            events.append(event)
            generated_activities.append(activity)
            current_time += timedelta(hours=duration + gap)

        # Verify Final Settlement was included
        if "Final Settlement" not in generated_activities:
            raise ValueError(f"Failed to generate Final Settlement for case {case_id}")

        print(f"Case {case_id} activities: {generated_activities}")  # Debug print
        return events

    def generate_fx_trade_data(self, num_cases: int = 1000) -> pd.DataFrame:
        """Generate dataset with verification of Final Settlement and CSV output"""
        all_events = []
        start_date = datetime.now() - timedelta(days=30)

        print("Starting data generation...")  # Debug print

        # Generate base cases
        for case_num in range(num_cases):
            case_id = f"Case_{case_num:05d}"
            case_start = start_date + timedelta(minutes=random.randint(0, 43200))
            case_events = self.generate_case_events(case_id, case_start)

            # Verify this case has Final Settlement
            if not any(e['activity'] == "Final Settlement" for e in case_events):
                raise ValueError(f"Case {case_id} missing Final Settlement after generation")

            all_events.extend(case_events)

            if case_num % 100 == 0:
                print(f"Generated {case_num} cases...")

        # Convert to DataFrame
        df = pd.DataFrame(all_events)

        # Verify before injecting outliers
        total_cases = df['case_id'].nunique()
        cases_with_settlement = df[df['activity'] == "Final Settlement"]['case_id'].nunique()
        print(f"Before outliers: {total_cases} total cases, {cases_with_settlement} with Final Settlement")

        if total_cases != cases_with_settlement:
            raise ValueError(f"Missing Final Settlement in {total_cases - cases_with_settlement} cases before outliers")

        # Now inject a small number of controlled outliers
        df = self.inject_controlled_outliers(df)

        # Verify after outliers
        final_cases_with_settlement = df[df['activity'] == "Final Settlement"]['case_id'].nunique()
        missing_settlement = total_cases - final_cases_with_settlement
        print(f"After outliers: {missing_settlement} cases missing Final Settlement")

        if missing_settlement > 10:  # We want fewer than 10 incomplete cases
            raise ValueError(f"Too many cases missing Final Settlement: {missing_settlement}")

        # Sort and save to CSV
        df = df.sort_values(['case_id', 'timestamp'])
        output_path = f"fx_trade_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Generated {len(df)} events across {df['case_id'].nunique()} cases")
        logger.info(f"Data saved to {output_path}")

        return df

    def inject_controlled_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject small number of controlled outliers"""
        df = df.copy()
        case_ids = list(df['case_id'].unique())

        # Select exactly 8 cases for incompleteness
        incomplete_cases = random.sample(case_ids, 8)

        for case_id in incomplete_cases:
            case_mask = df['case_id'] == case_id
            case_events = df[case_mask]

            # Find Final Settlement event and remove it
            final_settlement_mask = (case_mask) & (df['activity'] == 'Final Settlement')
            if any(final_settlement_mask):
                df = df.drop(df[final_settlement_mask].index)

        # Add other outlier types (sequence violations, etc.)
        remaining_cases = [c for c in case_ids if c not in incomplete_cases]
        other_outliers = random.sample(remaining_cases, 20)  # 20 cases for other outlier types

        current_idx = 0
        for outlier_type in ['sequence_violation', 'resource_switch', 'long_running', 'rework_activity']:
            cases_for_type = other_outliers[current_idx:current_idx + 5]
            current_idx += 5

            for case_id in cases_for_type:
                case_mask = df['case_id'] == case_id
                case_events = df[case_mask]

                if outlier_type == 'sequence_violation':
                    # Swap two adjacent non-Final-Settlement activities
                    non_final_events = case_events[case_events['activity'] != 'Final Settlement']
                    if len(non_final_events) >= 2:
                        idx1, idx2 = non_final_events.index[0:2]
                        timestamps = df.loc[[idx1, idx2], 'timestamp'].values
                        df.loc[idx1, 'timestamp'] = timestamps[1]
                        df.loc[idx2, 'timestamp'] = timestamps[0]

                elif outlier_type == 'resource_switch':
                    # Switch resources mid-sequence
                    mid_events = case_events[case_events['activity'] != 'Final Settlement'].index[1:-1]
                    if len(mid_events) >= 2:
                        for idx in mid_events[:2]:
                            new_resource = random.choice([r for r in self.traders if r != df.loc[idx, 'resource']])
                            df.loc[idx, 'resource'] = new_resource

                elif outlier_type == 'long_running':
                    # Add delay before Final Settlement
                    non_final_mask = (case_mask) & (df['activity'] != 'Final Settlement')
                    if any(non_final_mask):
                        df.loc[non_final_mask, 'timestamp'] += pd.Timedelta(hours=4)

                elif outlier_type == 'rework_activity':
                    # Repeat a mid-sequence activity
                    repeatable = case_events[case_events['activity'].isin(['Trade Validation', 'Trade Matching'])]
                    if not repeatable.empty:
                        activity_to_repeat = repeatable.iloc[0]
                        new_row = activity_to_repeat.copy()
                        new_row['timestamp'] += pd.Timedelta(hours=2)
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        return df.sort_values(['case_id', 'timestamp'])


def main():
    """Generate FX trade data with proper OCPM alignment"""
    generator = FXTradeGenerator()
    df = generator.generate_fx_trade_data()
    logger.info(f"Generated {len(df)} events across {df['case_id'].nunique()} cases")


if __name__ == "__main__":
    main()