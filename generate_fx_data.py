#!/usr/bin/env python3
"""
FX Trading Data Generator
Generates CSV files for organization context and guidelines for FX trading analysis
"""

import pandas as pd
import os
import logging
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FXDataGenerator:
    """Generates sample data for FX trading analysis"""

    def __init__(self, output_dir="data"):
        """Initialize generator with output directory"""
        self.output_dir = output_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Output directory confirmed: {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory: {str(e)}")
            raise

    def generate_org_context(self):
        """Generate organization context data"""
        try:
            org_context_data = {
                'id': [
                    'FO_001', 'FO_002', 'FO_003', 'MO_001', 'MO_002',
                    'BO_001', 'BO_002', 'RISK_001', 'RISK_002', 'COMP_001'
                ],
                'department': [
                    'Front Office', 'Front Office', 'Front Office',
                    'Middle Office', 'Middle Office',
                    'Back Office', 'Back Office',
                    'Risk Management', 'Risk Management',
                    'Compliance'
                ],
                'role': [
                    'FX Trader', 'Sales Trader', 'Trading Desk Head',
                    'Market Risk Analyst', 'Trade Support',
                    'Settlements', 'Reconciliation',
                    'Risk Officer', 'Senior Risk Manager',
                    'Compliance Officer'
                ],
                'responsibilities': [
                    'Execute trades, Market making',
                    'Client relationship, Quote provision',
                    'Trading strategy, Risk limits',
                    'Market risk monitoring, P&L analysis',
                    'Trade processing, Position keeping',
                    'Trade settlement, Payment processing',
                    'Position reconciliation, Break resolution',
                    'Risk assessment, Limit monitoring',
                    'Risk policy, Framework management',
                    'Regulatory compliance, Policy enforcement'
                ],
                'operating_hours': [
                    '24/5', '24/5', '24/5',
                    '8/5', '24/5',
                    '8/5', '8/5',
                    '8/5', '8/5',
                    '8/5'
                ],
                'location': [
                    'Trading Floor', 'Trading Floor', 'Trading Floor',
                    'Operations', 'Operations',
                    'Operations', 'Operations',
                    'Risk Department', 'Risk Department',
                    'Compliance Department'
                ],
                'reporting_line': [
                    'Trading Desk Head', 'Trading Desk Head', 'Head of Trading',
                    'Head of Middle Office', 'Head of Middle Office',
                    'Head of Operations', 'Head of Operations',
                    'Chief Risk Officer', 'Chief Risk Officer',
                    'Chief Compliance Officer'
                ],
                'systems_used': [
                    'Trading System, Market Data Terminals',
                    'CRM, Trading System',
                    'Risk Systems, Trading System',
                    'Risk Systems, Analytics Platform',
                    'Trade Processing System',
                    'Settlement System',
                    'Reconciliation System',
                    'Risk Management System',
                    'Risk Management System',
                    'Compliance Monitoring System'
                ]
            }

            df = pd.DataFrame(org_context_data)
            output_path = os.path.join(self.output_dir, 'fx_org_context.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"Generated organization context CSV: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating organization context: {str(e)}")
            raise

    def generate_guidelines(self):
        """Generate FX trading guidelines data"""
        try:
            guidelines_data = {
                'id': [
                    'GL001', 'GL002', 'GL003', 'GL004', 'GL005',
                    'GL006', 'GL007', 'GL008', 'GL009', 'GL010'
                ],
                'category': [
                    'Trade Execution', 'Market Risk', 'Compliance',
                    'Operations', 'Risk Management',
                    'Settlement', 'Market Making', 'Position Management',
                    'Documentation', 'Reporting'
                ],
                'guideline': [
                    'All trades must be executed within approved limits',
                    'Market risk must be monitored real-time',
                    'All trades must comply with regulatory requirements',
                    'Trade confirmation within 15 minutes',
                    'Risk assessment required for new trading strategies',
                    'Settlement must be completed within T+2',
                    'Continuous two-way quotes during market hours',
                    'Position reconciliation required daily',
                    'All trade documentation must be complete and archived',
                    'Daily risk reports must be generated and reviewed'
                ],
                'priority': [
                    'High', 'High', 'High',
                    'Medium', 'High',
                    'High', 'Medium', 'High',
                    'Medium', 'High'
                ],
                'compliance_requirement': [
                    'Mandatory', 'Mandatory', 'Mandatory',
                    'Mandatory', 'Mandatory',
                    'Mandatory', 'Required', 'Mandatory',
                    'Required', 'Mandatory'
                ],
                'monitoring_frequency': [
                    'Real-time', 'Real-time', 'Real-time',
                    'Hourly', 'Daily',
                    'Daily', 'Real-time', 'Daily',
                    'Per Trade', 'Daily'
                ],
                'responsible_department': [
                    'Front Office', 'Risk Management', 'Compliance',
                    'Middle Office', 'Risk Management',
                    'Back Office', 'Front Office', 'Middle Office',
                    'Middle Office', 'Risk Management'
                ],
                'control_measure': [
                    'System limits, Dealer oversight',
                    'Automated risk monitoring',
                    'Pre-trade compliance checks',
                    'Automated confirmation matching',
                    'Risk committee approval',
                    'Settlement system controls',
                    'Market making obligations',
                    'Automated reconciliation',
                    'Document management system',
                    'Automated reporting system'
                ]
            }

            df = pd.DataFrame(guidelines_data)
            output_path = os.path.join(self.output_dir, 'fx_guidelines.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"Generated guidelines CSV: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating guidelines: {str(e)}")
            raise

    def generate_all(self):
        """Generate all required CSV files"""
        try:
            logger.info("Starting data generation process...")
            org_path = self.generate_org_context()
            guidelines_path = self.generate_guidelines()

            logger.info("Data generation completed successfully")
            return {
                'org_context': org_path,
                'guidelines': guidelines_path
            }
        except Exception as e:
            logger.error(f"Error in data generation process: {str(e)}")
            raise


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate FX Trading Analysis Data')
    parser.add_argument('--output-dir', default='data',
                        help='Output directory for generated files (default: data)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify generated files after creation')

    args = parser.parse_args()

    try:
        generator = FXDataGenerator(output_dir=args.output_dir)
        generated_files = generator.generate_all()

        print("\nGenerated Files:")
        for file_type, file_path in generated_files.items():
            print(f"{file_type}: {file_path}")

        if args.verify:
            print("\nVerifying generated files:")
            for file_type, file_path in generated_files.items():
                df = pd.read_csv(file_path)
                print(f"\n{file_type} sample data:")
                print(df.head(2))
                print(f"Total rows: {len(df)}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()