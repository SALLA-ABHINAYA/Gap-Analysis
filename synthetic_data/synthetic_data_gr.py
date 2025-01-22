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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fx_trade_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FXTradeConfig:
    """Enhanced configuration for different FX trade types"""
    trade_type: str
    traders: List[str] = field(default_factory=list)
    currencies: List[str] = field(default_factory=list)
    client_types: List[str] = field(default_factory=list)
    booking_systems: List[str] = field(default_factory=list)

    # Specific configurations for different trade types
    spot_configs: Dict = field(default_factory=lambda: {
        "settlement_types": ["T+1", "T+2"],
        "execution_venues": ["Primary Market", "ECN", "Direct Bank"]
    })

    forward_configs: Dict = field(default_factory=lambda: {
        "tenor_range": ["1W", "1M", "3M", "6M", "1Y"],
        "fixing_conventions": ["Spot", "Tom", "Today"]
    })

    futures_configs: Dict = field(default_factory=lambda: {
        "exchanges": ["CME", "ICE", "Eurex"],
        "contract_months": ["H", "M", "U", "Z"],
        "lot_sizes": [100000, 125000, 500000]
    })

    swap_configs: Dict = field(default_factory=lambda: {
        "swap_types": ["Spot-Forward", "Forward-Forward"],
        "tenor_pairs": ["1M-3M", "3M-6M", "6M-1Y"]
    })

    options_configs: Dict = field(default_factory=lambda: {
        "option_types": ["Vanilla Call", "Vanilla Put", "Butterfly", "Risk Reversal"],
        "strike_types": ["ATM", "25D", "10D"],
        "expiry_tenors": ["1W", "1M", "3M", "6M"]
    })

    def __post_init__(self):
        """Initialize common configurations"""
        self.traders = [
            "Spot Desk A", "Forward Desk B", "Options Desk C",
            "Market Maker D", "Hedge Desk E", "Client Desk F"
        ]
        self.currencies = [
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "EUR/GBP",
            "USD/CHF", "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY"
        ]
        self.client_types = [
            "Hedge Fund", "Corporate", "Bank", "Asset Manager",
            "Pension Fund", "Central Bank", "Broker"
        ]
        self.booking_systems = [
            "Murex", "Summit", "Calypso", "Internal", "FXAll",
            "360T", "FXT"
        ]


@dataclass
class TradeEventGenerator:
    """Generates trade events based on trade type"""
    config: FXTradeConfig
    metadata: 'MetadataTracker'

    def generate_base_events(self) -> List[str]:
        """Generate base event sequence common to all trade types"""
        return [
            "Trade Initiated",
            "Market Data Validation",
            "Quote Requested",
            "Quote Provided",
            "Trade Execution",
            "Trade Validation",
            "Risk Assessment",
            "Trade Confirmation",
            "Trade Matching",
            "Settlement Instructions",
            "Position Reconciliation",
            "Trade Reconciliation"
        ]

    def generate_trade_specific_events(self) -> List[str]:
        """Generate trade type specific events"""
        specific_events = {
            "SPOT": [
                "Rate Lock",
                "Payment Instructions",
                "Funds Verification",
                "T+1 Settlement Processing"
            ],
            "FORWARD": [
                "Forward Points Calculation",
                "Credit Line Check",
                "Forward Value Date Confirmation",
                "Margin Calculation"
            ],
            "FUTURES": [
                "Exchange Connectivity Check",
                "Margin Requirement Calculation",
                "Clearing House Submission",
                "Position Marking"
            ],
            "SWAP": [
                "Swap Points Calculation",
                "Term Structure Analysis",
                "Cash Flow Generation",
                "Swap Reset Confirmation"
            ],
            "OPTIONS": [
                "Volatility Surface Analysis",
                "Premium Calculation",
                "Greeks Calculation",
                "Exercise Decision"
            ]
        }
        return specific_events.get(self.config.trade_type.upper(), [])

    def generate_regulatory_events(self) -> List[str]:
        """Generate regulatory compliance events"""
        return [
            "Transaction Reporting Check",
            "Best Execution Validation",
            "Trade Transparency Assessment",
            "Regulatory Reporting Generation"
        ]


class MetadataTracker:
    """Enhanced metadata tracking for different trade types"""

    def __init__(self, trade_type: str):
        self.trade_type = trade_type
        self.total_events = 0
        self.total_cases = 0
        self.activity_counts = Counter()
        self.field_usage = Counter()
        self.client_type_distribution = Counter()
        self.currency_pair_distribution = Counter()
        self.avg_activities_per_case = 0
        self.path_variations = set()
        self.trade_specific_metrics = {}

    def track_trade_specific_metrics(self, metrics: Dict):
        """Track metrics specific to trade type"""
        self.trade_specific_metrics.update(metrics)

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary"""
        return {
            "trade_type": self.trade_type,
            "total_events": self.total_events,
            "total_cases": self.total_cases,
            "activity_distribution": dict(self.activity_counts),
            "field_usage": dict(self.field_usage),
            "client_type_distribution": dict(self.client_type_distribution),
            "currency_pair_distribution": dict(self.currency_pair_distribution),
            "avg_activities_per_case": self.avg_activities_per_case,
            "unique_path_variations": len(self.path_variations),
            "trade_specific_metrics": self.trade_specific_metrics
        }


def generate_fx_trade_data(trade_type: str, num_cases: int = 30000) -> pd.DataFrame:
    """Main function to generate FX trade data for specific trade type"""
    logger.info(f"Starting {trade_type} trade data generation for {num_cases} cases")

    # Initialize configuration and tracking
    config = FXTradeConfig(trade_type=trade_type)
    metadata = MetadataTracker(trade_type)
    generator = TradeEventGenerator(config, metadata)

    event_log = []
    start_time = datetime.now()

    # Initialize metadata counters
    metadata.total_cases = num_cases
    total_events = 0

    for case_id in range(1, num_cases + 1):
        if case_id % 1000 == 0:
            logger.info(f"Generated {case_id} cases...")

        # Generate trade characteristics
        trade_chars = generate_trade_characteristics(config)

        # Track distributions
        metadata.client_type_distribution[trade_chars['client_type']] += 1
        metadata.currency_pair_distribution[trade_chars['currency']] += 1

        # Build event sequence
        events = generator.generate_base_events()
        events.extend(generator.generate_trade_specific_events())

        if random.random() < 0.4:  # 40% chance of regulatory events
            events.extend(generator.generate_regulatory_events())

        # Track path variation
        metadata.path_variations.add(tuple(events))

        # Generate event sequence
        case_events = generate_event_sequence(
            case_id, events, trade_chars, config, metadata
        )

        # Update event count
        total_events += len(case_events)
        event_log.extend(case_events)

    # Update metadata
    metadata.total_events = total_events

    # Process and save metadata
    process_metadata(metadata, start_time)

    # Create and save DataFrame
    df = pd.DataFrame(event_log)
    df.sort_values(by=["case_id", "timestamp"], inplace=True)

    output_path = f"fx_trade_log_{trade_type.lower()}.csv"
    df.to_csv(output_path, sep=';', index=False)

    logger.info(f"Generated {trade_type} trade data saved to {output_path}")
    logger.info(f"Total events generated: {total_events}")
    logger.info(f"Average events per case: {total_events / num_cases:.2f}")

    return df


def generate_trade_characteristics(config: FXTradeConfig) -> Dict:
    """Generate characteristics specific to trade type"""
    base_chars = {
        "client_type": random.choice(config.client_types),
        "currency": random.choice(config.currencies),
        "trader": random.choice(config.traders),
        "booking_system": random.choice(config.booking_systems)
    }

    # Add trade-type specific characteristics
    if config.trade_type == "SPOT":
        base_chars.update({
            "settlement_type": random.choice(config.spot_configs["settlement_types"]),
            "execution_venue": random.choice(config.spot_configs["execution_venues"])
        })
    elif config.trade_type == "FORWARD":
        base_chars.update({
            "tenor": random.choice(config.forward_configs["tenor_range"]),
            "fixing_convention": random.choice(config.forward_configs["fixing_conventions"])
        })
    # Add similar blocks for other trade types

    return base_chars


def generate_event_sequence(case_id: int, events: List[str],
                            trade_chars: Dict, config: FXTradeConfig,
                            metadata: MetadataTracker) -> List[Dict]:
    """Generate properly sequenced events with timestamps"""
    sequence = []
    start_time = datetime.now() + timedelta(days=random.randint(-30, 0))

    for i, event in enumerate(events):
        timestamp = start_time + timedelta(minutes=random.randint(10, 60) * i)

        event_data = {
            "case_id": f"Case_{case_id}",
            "activity": event,
            "timestamp": timestamp,
            **trade_chars  # Include all trade characteristics
        }

        # Add event-specific fields
        event_data.update(generate_event_specific_fields(event, config.trade_type))

        # Track metadata
        metadata.activity_counts[event] += 1
        metadata.field_usage.update(event_data.keys())

        sequence.append(event_data)

    return sequence


def generate_event_specific_fields(event: str, trade_type: str) -> Dict:
    """Generate fields specific to event and trade type"""
    fields = {}

    if "Execution" in event:
        fields.update({
            "execution_price": round(random.uniform(0.9, 1.1), 4),
            "notional_amount": random.randint(1000000, 50000000),
            "execution_time": round(random.uniform(0.01, 0.5), 3)  # in seconds
        })

    if "Risk" in event:
        fields.update({
            "risk_score": round(random.uniform(1, 5), 2),
            "limit_usage": f"{random.randint(0, 100)}%"
        })

    # Add trade-type specific fields
    if trade_type == "OPTIONS" and "Greeks" in event:
        fields.update({
            "delta": round(random.uniform(-1, 1), 2),
            "gamma": round(random.uniform(0, 0.2), 3),
            "vega": round(random.uniform(0, 50000), 2),
            "theta": round(random.uniform(-1000, 0), 2)
        })

    return fields


def process_metadata(metadata: MetadataTracker, start_time: datetime):
    """Process and save metadata"""
    # Calculate average activities per case
    if metadata.total_cases > 0:  # Add safety check
        metadata.avg_activities_per_case = metadata.total_events / metadata.total_cases
    else:
        metadata.avg_activities_per_case = 0
        logger.warning("No cases processed in metadata")

    # Save metadata
    metadata_file = f'event_log_metadata_{metadata.trade_type.lower()}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=4)

    # Log generation statistics
    duration = datetime.now() - start_time
    logger.info(f"\n=== {metadata.trade_type} Trade Generation Complete ===")
    logger.info(f"Duration: {duration}")
    logger.info(f"Total events: {metadata.total_events}")
    logger.info(f"Total cases: {metadata.total_cases}")
    logger.info(f"Average activities per case: {metadata.avg_activities_per_case:.2f}")
    logger.info(f"Unique path variations: {len(metadata.path_variations)}")
    logger.info(f"Metadata saved to: {metadata_file}")


def main():
    """Generate data for all FX trade types"""
    trade_types = ["SPOT", "FORWARD", "FUTURES", "OPTIONS", "SWAP"]
    start_time = datetime.now()

    logger.info("Starting FX trade data generation")

    try:
        for trade_type in trade_types:
            logger.info(f"\nGenerating {trade_type} trade data...")
            df = generate_fx_trade_data(trade_type)
            logger.info(f"Completed {trade_type} generation. DataFrame shape: {df.shape}")

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise

    finally:
        duration = datetime.now() - start_time
        logger.info(f"\nTotal generation time: {duration}")


if __name__ == "__main__":
    main()