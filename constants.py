# constants.py

# Domain object types and their properties
OBJECT_TYPES = {
    'Application': {
        'activities': ['Application Submission', 'Initial Assessment', 'Pre-Approval', 'Underwriting Approved', 'Final Approval', 'Rejected'],
        'attributes': ['applicant_id', 'credit_score', 'loan_amount', 'application_date', 'status'],
        'relationships': ['Property', 'Customer', 'Loan']
    },
    'Property': {
        'activities': ['Appraisal Request', 'Valuation Accepted', 'Valuation Issues'],
        'attributes': ['property_address', 'property_type', 'valuation_amount', 'property_id'],
        'relationships': ['Application', 'Appraisal']
    },
    'Appraisal': {
        'activities': ['Appraisal Request', 'Valuation Accepted', 'Valuation Issues'],
        'attributes': ['appraiser_id', 'appraisal_date', 'appraisal_value', 'appraisal_status'],
        'relationships': ['Property', 'Application']
    },
    'Loan': {
        'activities': ['Signing of Loan Agreement', 'Loan Funding', 'Disbursement of Funds', 'Loan Closure'],
        'attributes': ['loan_id', 'interest_rate', 'loan_term', 'loan_type', 'monthly_payment'],
        'relationships': ['Application', 'Customer', 'Property']
    },
    'Customer': {
        'activities': ['Application Submission', 'Signing of Loan Agreement'],
        'attributes': ['customer_id', 'income', 'employment_status', 'debt_to_income_ratio'],
        'relationships': ['Application', 'Loan']
    }
}

# Business hours configuration
BUSINESS_HOURS = {
    'market_start_hour': 9,
    'market_end_hour': 17,
    'end_of_day_start': 16,
    'end_of_day_end': 18,
    'business_days': [0, 1, 2, 3, 4]  # Monday to Friday
}

# Process dependencies
ACTIVITY_DEPENDENCIES = {
    'Trade Execution': ['Market Data Validation', 'Risk Assessment'],
    'Settlement': ['Trade Execution', 'Position Reconciliation'],
    'Position Reconciliation': ['Trade Execution']
}

# Dashboard labels and terminology
DOMAIN_TERMINOLOGY = {
    'domain_name': 'FX Trading',
    'dashboard_title': 'FX Trading Analytics Dashboard',
    'compliance_framework': 'FX Global Code',
    'currency_list': ['EUR', 'USD', 'GBP']
}

# Mock data configuration for visualization
MOCK_DATA_CONFIG = {
    'start_date': '2024-01-01',
    'periods': 6,
    'frequency': 'M',
    'currencies': {
        'EUR': [4000, 3000, 2000, 2780, 1890, 2390],
        'USD': [2400, 1398, 9800, 3908, 4800, 3800],
        'GBP': [2400, 2210, 2290, 2000, 2181, 2500]
    }
}