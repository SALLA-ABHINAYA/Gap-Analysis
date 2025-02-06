from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random
import pandas as pd
import logging
import json
from collections import Counter
import numpy as np

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

        # Expanded configuration for larger dataset
        self.traders = [
            "Market Maker A", "Market Maker B", "Market Maker C", "Market Maker D",
            "Client Desk A", "Client Desk B", "Client Desk C", "Client Desk D",
            "Options Desk A", "Options Desk B", "Options Desk C",
            "Hedge Desk A", "Hedge Desk B", "Hedge Desk C",
            "Risk Desk A", "Risk Desk B", "Risk Desk C",
            "Trading Desk A", "Trading Desk B", "Trading Desk C"
        ]

        self.currencies = [
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "EUR/GBP",
            "USD/CHF", "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY",
            "AUD/JPY", "EUR/CHF", "GBP/CHF", "EUR/CAD", "GBP/CAD",
            "USD/MXN", "USD/BRL", "USD/CNH", "USD/SGD", "USD/HKD"
        ]

        self.client_types = [
            "Hedge Fund", "Corporate", "Bank", "Asset Manager",
            "Pension Fund", "Central Bank", "Insurance Company",
            "Sovereign Wealth Fund", "Private Bank", "Broker",
            "Market Maker", "Proprietary Trading", "Family Office"
        ]

        self.booking_systems = [
            "Murex", "Summit", "Calypso", "Internal", "FXConnect",
            "FXall", "360T", "Bloomberg FXGO", "Reuters RTFX"
        ]

        # Add trade strategies for more variety
        self.trade_strategies = [
            "Spot", "Forward", "Swap", "Option", "NDF",
            "Multi-leg", "Algorithmic", "Manual", "API"
        ]

        # Initialize volume profiles for realistic distribution
        self.volume_profiles = {
            'high': {'min': 10000000, 'max': 100000000},
            'medium': {'min': 1000000, 'max': 10000000},
            'low': {'min': 100000, 'max': 1000000}
        }

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

    def _generate_realistic_amount(self, client_type: str) -> int:
        """Generate realistic trade amounts based on client type"""
        if client_type in ["Central Bank", "Sovereign Wealth Fund"]:
            profile = 'high'
        elif client_type in ["Hedge Fund", "Bank", "Asset Manager"]:
            profile = random.choices(['high', 'medium'], weights=[0.3, 0.7])[0]
        else:
            profile = random.choices(['medium', 'low'], weights=[0.6, 0.4])[0]

        volume_range = self.volume_profiles[profile]
        return random.randint(volume_range['min'], volume_range['max'])

    def generate_case_events(self, case_id: str, start_time: datetime) -> List[Dict]:
        """Generate events with enhanced characteristics"""
        events = []
        current_time = start_time

        trade_sequence = [
            "Trade Initiated",
            "Trade Validation",
            "Trade Execution",
            "Trade Confirmation",
            "Trade Matching",
            "Trade Reconciliation",
            "Trade Transparency Assessment",
            "Final Settlement"
        ]

        # Generate base characteristics
        client_type = random.choice(self.client_types)
        characteristics = {
            'resource': random.choice(self.traders),
            'currency_pair': random.choice(self.currencies),
            'client_type': client_type,
            'booking_system': random.choice(self.booking_systems),
            'trading_strategy': random.choice(self.trade_strategies),
            'notional_amount': self._generate_realistic_amount(client_type)
        }

        # Add time-based variation
        hour = start_time.hour
        if 14 <= hour <= 16:  # Peak trading hours
            duration_multiplier = 0.8  # Faster processing
        elif 6 <= hour <= 18:  # Normal business hours
            duration_multiplier = 1.0
        else:  # Off hours
            duration_multiplier = 1.3  # Slower processing

        for activity in trade_sequence:
            # Add realistic duration variations
            base_duration = random.uniform(0.1, 0.5)
            adjusted_duration = base_duration * duration_multiplier

            # Add realistic gaps between activities
            base_gap = random.uniform(0.1, 0.3)
            adjusted_gap = base_gap * duration_multiplier

            event = {
                'case_id': case_id,
                'activity': activity,
                'timestamp': current_time,
                'object_type': 'Trade',
                **characteristics
            }

            # Add activity-specific attributes
            if activity == "Trade Execution":
                event['execution_price'] = round(random.uniform(0.98, 1.02) *
                                                 characteristics['notional_amount'], 2)
            elif activity == "Trade Validation":
                event['validation_level'] = random.choice(['L1', 'L2', 'L3'])

            events.append(event)
            current_time += timedelta(hours=adjusted_duration + adjusted_gap)

        return events

    def generate_fx_trade_data(self, num_cases: int = 10000) -> pd.DataFrame:
        """Generate larger dataset with verification"""
        all_events = []
        start_date = datetime.now() - timedelta(days=30)
        batch_size = 1000  # Process in batches for memory efficiency

        logger.info(f"Starting data generation for {num_cases} cases...")

        for batch_start in range(0, num_cases, batch_size):
            batch_end = min(batch_start + batch_size, num_cases)
            batch_events = []

            for case_num in range(batch_start, batch_end):
                case_id = f"Case_{case_num:06d}"
                # Distribute cases realistically across the time period
                hour_weight = random.choices(
                    range(24),
                    weights=[1] * 6 + [2] * 2 + [5] * 8 + [3] * 2 + [1] * 6
                )[0]  # Higher weights during business hours
                case_start = start_date + timedelta(
                    days=random.randint(0, 29),
                    hours=hour_weight,
                    minutes=random.randint(0, 59)
                )
                case_events = self.generate_case_events(case_id, case_start)
                batch_events.extend(case_events)

            all_events.extend(batch_events)
            logger.info(f"Generated {batch_end} cases...")

        # Convert to DataFrame
        df = pd.DataFrame(all_events)

        # Inject controlled outliers (1% of cases)
        num_outliers = num_cases // 100
        df = self.inject_controlled_outliers(df, num_outliers)

        # Sort and save
        df = df.sort_values(['case_id', 'timestamp'])
        output_path = f"fx_trade_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

        # Save in chunks to handle large file size
        chunk_size = 100000
        for i in range(0, len(df), chunk_size):
            if i == 0:
                df[i:i + chunk_size].to_csv(output_path, index=False)
            else:
                df[i:i + chunk_size].to_csv(output_path, mode='a', header=False, index=False)

        logger.info(f"Generated {len(df)} events across {df['case_id'].nunique()} cases")
        logger.info(f"Data saved to {output_path}")

        return df

    def inject_controlled_outliers(self, df: pd.DataFrame, num_outliers: int) -> pd.DataFrame:
        """Inject outliers proportionally to dataset size"""
        df = df.copy()
        case_ids = list(df['case_id'].unique())

        # Calculate outlier distribution
        num_incomplete = num_outliers // 5  # 20% incomplete cases
        num_sequence = num_outliers // 5  # 20% sequence violations
        num_resource = num_outliers // 5  # 20% resource switches
        num_long = num_outliers // 5  # 20% long running
        num_rework = num_outliers // 5  # 20% rework activities

        # Select cases for each type of outlier
        all_outlier_cases = random.sample(case_ids, num_outliers)
        current_idx = 0

        # Incomplete cases
        incomplete_cases = all_outlier_cases[current_idx:current_idx + num_incomplete]
        current_idx += num_incomplete

        # Process each outlier type
        for case_id in incomplete_cases:
            case_mask = df['case_id'] == case_id
            final_settlement_mask = case_mask & (df['activity'] == 'Final Settlement')
            df = df.drop(df[final_settlement_mask].index)

        # Other outlier types
        outlier_types = {
            'sequence_violation': all_outlier_cases[current_idx:current_idx + num_sequence],
            'resource_switch': all_outlier_cases[current_idx + num_sequence:current_idx + num_sequence + num_resource],
            'long_running': all_outlier_cases[
                            current_idx + num_sequence + num_resource:current_idx + num_sequence + num_resource + num_long],
            'rework_activity': all_outlier_cases[current_idx + num_sequence + num_resource + num_long:]
        }

        for outlier_type, cases in outlier_types.items():
            for case_id in cases:
                case_mask = df['case_id'] == case_id
                case_events = df[case_mask]

                if outlier_type == 'sequence_violation':
                    non_final_events = case_events[case_events['activity'] != 'Final Settlement']
                    if len(non_final_events) >= 2:
                        idx1, idx2 = non_final_events.index[0], non_final_events.index[1]
                        df.loc[idx1, 'timestamp'], df.loc[idx2, 'timestamp'] = \
                            df.loc[idx2, 'timestamp'], df.loc[idx1, 'timestamp']

                elif outlier_type == 'resource_switch':
                    mid_events = case_events[case_events['activity'] != 'Final Settlement'].index[1:-1]
                    if len(mid_events) >= 2:
                        new_resource = random.choice(
                            [r for r in self.traders if r != df.loc[mid_events[0], 'resource']])
                        df.loc[mid_events[0], 'resource'] = new_resource

                elif outlier_type == 'long_running':
                    non_final_mask = case_mask & (df['activity'] != 'Final Settlement')
                    if non_final_mask.any():
                        df.loc[non_final_mask, 'timestamp'] += pd.Timedelta(hours=4)

                elif outlier_type == 'rework_activity':
                    repeatable = case_events[case_events['activity'].isin(['Trade Validation', 'Trade Matching'])]
                    if not repeatable.empty:
                        new_row = repeatable.iloc[0].copy()
                        new_row['timestamp'] += pd.Timedelta(hours=2)
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        return df.sort_values(['case_id', 'timestamp'])


def main():
    """Generate larger FX trade dataset"""
    generator = FXTradeGenerator()
    # Generate 10,000 cases instead of 1,000 to create ~10MB file
    df = generator.generate_fx_trade_data(num_cases=10000)
    logger.info(f"Generated {len(df)} events across {df['case_id'].nunique()} cases")


if __name__ == "__main__":
    main()