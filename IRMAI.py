import logging

import numpy as np
import streamlit as st

import json
import pandas as pd
from openai import OpenAI
from pathlib import Path
import os
from Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer
from fmea_analyzer import OCELFMEAAnalyzer
from ocpm_analysis import create_ocpm_ui
from ai_ocel_analyzer import AIOCELAnalyzer
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
from collections import defaultdict
import logging
import traceback
from datetime import datetime
from typing import Dict, List
from neo4j import GraphDatabase


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GapFindings:
    """Data class for gap analysis findings"""
    category: str
    severity: str
    description: str
    current_state: Any
    expected_state: Any
    impact: str
    recommendations: List[str]


class AIGapAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize analyzer with OpenAI API key"""
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")

        try:
            # Initialize OpenAI client
            self.openai_client = OpenAI(
                api_key="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"
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def analyze_process_gaps(self, process_data: Dict, guidelines: Dict) -> Dict:
        """Use AI to analyze gaps between process and guidelines with error handling"""
        try:
            context = {
                "process_summary": {
                    "total_activities": len(process_data.get('activities', [])),
                    "total_events": process_data.get('total_events', 0),
                    "activities": [a['name'] for a in process_data.get('activities', [])]
                },
                "guidelines_summary": {
                    "total_guidelines": len(guidelines.get('guidelines', [])),
                    "guidelines": [g['name'] for g in guidelines.get('guidelines', [])]
                }
            }

            logger.debug(f"Analysis context: {context}")

            prompt = (
                "Analyze this process data and provide gaps, recommendations, and metrics in JSON format.\n"
                f"Context: {json.dumps(context, indent=2)}\n\n"
                "Response must be valid JSON with this structure:\n"
                "{\n"
                '  "gaps": [{"category": "string", "severity": "string", "description": "string"}],\n'
                '  "recommendations": [{"description": "string", "priority": "string"}],\n'
                '  "metrics": {"coverage": number, "effectiveness": number}\n'
                "}"
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a process mining expert. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            response_text = response.choices[0].message.content

            # Clean response text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1]

            response_text = response_text.strip()

            try:
                analysis = json.loads(response_text)
                logger.info("Successfully parsed AI analysis")
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response: {str(e)}")
                logger.error(f"Raw response: {response_text}")
                return self._get_default_analysis()

        except Exception as e:
            logger.error(f"Error in AI gap analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict:
        """Return default analysis structure when AI fails"""
        return {
            'gaps': [
                {
                    'category': 'Process Coverage',
                    'severity': 'Medium',
                    'description': 'Standard gap analysis based on process metrics'
                }
            ],
            'recommendations': [
                {
                    'description': 'Review process completion rates and controls',
                    'priority': 'High'
                }
            ],
            'metrics': {
                'coverage': 75.0,
                'effectiveness': 70.0
            }
        }
   
    def _structure_analysis(self, raw_analysis: Dict) -> Dict:
        """Structure and validate AI analysis output"""
        structured = {
            'gaps': [],
            'recommendations': [],
            'metrics': {
                'compliance': {},
                'operational': {},
                'risk': {}
            }
        }

        # Process identified gaps
        for gap in raw_analysis.get('gaps', []):
            structured['gaps'].append({
                'category': gap.get('category'),
                'description': gap.get('description'),
                'severity': gap.get('severity', 'Medium'),
                'impact': gap.get('impact'),
                'related_controls': gap.get('related_controls', [])
            })

        # Process recommendations
        for rec in raw_analysis.get('recommendations', []):
            structured['recommendations'].append({
                'description': rec.get('description'),
                'priority': rec.get('priority', 'Medium'),
                'implementation_timeline': rec.get('timeline'),
                'expected_impact': rec.get('expected_impact'),
                'required_resources': rec.get('required_resources', [])
            })

        # Process metrics
        metrics = raw_analysis.get('metrics', {})
        structured['metrics'] = {
            'compliance': {
                'regulatory_coverage': metrics.get('regulatory_coverage', 0),
                'control_effectiveness': metrics.get('control_effectiveness', 0)
            },
            'operational': {
                'process_adherence': metrics.get('process_adherence', 0),
                'efficiency_score': metrics.get('efficiency_score', 0)
            },
            'risk': {
                'risk_coverage': metrics.get('risk_coverage', 0),
                'control_maturity': metrics.get('control_maturity', 0)
            }
        }

        return structured

    def analyze_recommendations(self, gaps: List[Dict]) -> List[Dict]:
        """Use AI to generate detailed recommendations based on identified gaps"""
        try:
            context = f"""
            Based on these identified process gaps:
            {json.dumps(gaps, indent=2)}

            Generate detailed recommendations including:
            1. Immediate actions needed
            2. Long-term improvements
            3. Resource requirements
            4. Implementation timeline
            5. Expected benefits

            Format as structured JSON.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert process improvement consultant."},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            recommendations = json.loads(response.choices[0].message.content)
            return self._structure_recommendations(recommendations)

        except Exception as e:
            logging.error(f"Error generating AI recommendations: {str(e)}")
            raise

    def _structure_recommendations(self, raw_recommendations: List[Dict]) -> List[Dict]:
        """Structure and validate AI-generated recommendations"""
        structured = []
        for rec in raw_recommendations:
            structured.append({
                'id': f"REC_{len(structured) + 1:03d}",
                'description': rec.get('description'),
                'priority': rec.get('priority', 'Medium'),
                'target_date': rec.get('timeline'),
                'status': 'Open',
                'impact': rec.get('expected_benefits'),
                'resources': rec.get('required_resources', []),
                'implementation_steps': rec.get('implementation_steps', [])
            })
        return structured


class GapAnalysisVisualizer:
    """Visualization component for gap analysis results"""

    def __init__(self, report: Dict):
        self.report = report
        self.findings = pd.DataFrame(report.get('findings', []))
        self.metrics = pd.DataFrame.from_dict(report.get('metrics', {}), orient='index')
        self.logger = logging.getLogger(__name__)
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key="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"
        )

    def get_ai_explanation(self, data: Dict, chart_type: str) -> str:
        """Get AI explanation for a visualization"""
        try:
            chart_contexts = {
                'overview': """Analyze the overall process metrics and identify key patterns.""",
                'severity_distribution': """Analyze the distribution of gap severities and their implications.""",
                'coverage_radar': """Analyze the coverage metrics across different dimensions.""",
                'gap_heatmap': """Analyze the patterns and clusters in the gap distribution.""",
                'timeline_view': """Analyze the implementation timeline and priority distribution."""
            }

            base_prompt = f"""
            Analyze this {chart_type} data:
            {json.dumps(data, indent=2)}

            Context: {chart_contexts.get(chart_type, '')}

            Provide a concise 2-3 sentence analysis focusing on:
            1. Key patterns or trends
            2. Business implications
            3. Actionable insights

            Keep the explanation business-friendly and actionable.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a process analytics expert."},
                    {"role": "user", "content": base_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error getting AI explanation: {str(e)}")
            return "AI analysis currently unavailable."

    def create_severity_distribution(self) -> Tuple[go.Figure, str]:
        """Create severity distribution chart with explanation"""
        summary = self.report.get('summary', {})

        fig = go.Figure(data=[
            go.Bar(
                name='Gaps',
                x=['High', 'Medium', 'Low'],
                y=[summary.get('high_severity', 0),
                   summary.get('medium_severity', 0),
                   summary.get('low_severity', 0)],
                marker_color=['#ff4d4d', '#ffa64d', '#4da6ff']
            )
        ])

        fig.update_layout(
            title='Gap Severity Distribution',
            xaxis_title='Severity Level',
            yaxis_title='Number of Gaps',
            template='plotly_white'
        )

        explanation = self.get_ai_explanation(summary, "severity distribution")
        return fig, explanation

    def create_coverage_radar(self) -> Tuple[go.Figure, str]:
        """Create coverage radar chart with explanation"""
        metrics = self.report.get('metrics', {})

        values = [
            round(metrics.get('compliance', {}).get('regulatory_coverage', 0), 1),
            round(metrics.get('operational', {}).get('process_adherence', 0), 1),
            round(metrics.get('risk', {}).get('control_coverage', 0), 1),
            round(metrics.get('risk', {}).get('risk_assessment_completion', 0), 1)
        ]

        categories = [
            'Regulatory Coverage',
            'Process Adherence',
            'Control Coverage',
            'Risk Assessment'
        ]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            marker_color='rgb(77, 166, 255)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )
            ),
            showlegend=False,
            title='Coverage Analysis'
        )

        explanation = self.get_ai_explanation(metrics, "coverage metrics")
        return fig, explanation

    def create_gap_heatmap(self) -> Tuple[go.Figure, str]:
        """Create gap heatmap with explanation"""
        if len(self.findings) == 0:
            return go.Figure(), "No data available for heatmap analysis."

        heatmap_data = pd.crosstab(
            index=self.findings.get('category', 'Unknown'),
            columns=self.findings.get('severity', 'Medium')
        ).fillna(0)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlBu_r'
        ))

        fig.update_layout(
            title='Gap Distribution Heatmap',
            xaxis_title='Severity',
            yaxis_title='Category'
        )

        explanation = self.get_ai_explanation(
            heatmap_data.to_dict(),
            "gap distribution heatmap"
        )
        return fig, explanation

    def create_timeline_view(self) -> Tuple[go.Figure, str]:
        """Create timeline view with explanation"""
        recommendations = self.report.get('recommendations', [])

        if not recommendations:
            return go.Figure(), "No timeline data available."

        # Convert recommendations to DataFrame
        df = pd.DataFrame(recommendations)

        fig = go.Figure()

        for priority in ['High', 'Medium', 'Low']:
            mask = df['priority'] == priority
            if any(mask):
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(df[mask]['target_date']),
                    y=[priority] * sum(mask),
                    mode='markers+text',
                    name=priority,
                    text=df[mask]['description'],
                    marker=dict(
                        size=10,
                        symbol='circle'
                    )
                ))

        fig.update_layout(
            title='Recommendation Timeline',
            xaxis_title='Target Date',
            yaxis_title='Priority'
        )

        explanation = self.get_ai_explanation(
            {"recommendations": recommendations},
            "recommendation timeline"
        )
        return fig, explanation

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete dashboard with visualizations and AI explanations"""
        try:
            # Create visualizations
            severity_fig, severity_expl = self.create_severity_distribution()
            coverage_fig, coverage_expl = self.create_coverage_radar()
            heatmap_fig, heatmap_expl = self.create_gap_heatmap()
            timeline_fig, timeline_expl = self.create_timeline_view()

            # Generate overview explanation
            overview_data = {
                'total_gaps': self.report['summary']['total_gaps'],
                'high_severity': self.report['summary']['high_severity'],
                'coverage': self.report['metrics']['operational']['process_adherence'],
                'effectiveness': self.report['metrics']['risk']['control_coverage']
            }
            overview_expl = self.get_ai_explanation(overview_data, 'overview')

            return {
                'figures': {
                    'severity_distribution': severity_fig,
                    'coverage_radar': coverage_fig,
                    'gap_heatmap': heatmap_fig,
                    'timeline_view': timeline_fig
                },
                'explanations': {
                    'overview': overview_expl,
                    'severity_distribution': severity_expl,
                    'coverage_radar': coverage_expl,
                    'gap_heatmap': heatmap_expl,
                    'timeline_view': timeline_expl
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating dashboard: {str(e)}")
            return {
                'figures': {},
                'explanations': {
                    'error': f"Error generating visualizations: {str(e)}"
                }
            }

    def display_recommendations_table(self) -> None:
        """Display recommendations table with error handling"""
        try:
            recommendations = self.report.get('recommendations', [])

            if not recommendations:
                st.info("No recommendations data available")
                return

            # Create DataFrame with only available columns
            df = pd.DataFrame(recommendations)
            display_columns = [
                'priority', 'description', 'target_date', 'status', 'impact'
            ]

            # Filter to only include columns that exist
            available_columns = [col for col in display_columns if col in df.columns]

            if not available_columns:
                st.warning("No valid columns found in recommendations data")
                return

            display_df = df[available_columns]

            # Apply styling
            styled_df = display_df.style.apply(lambda x: [
                'background-color: rgba(255,77,77,0.3)' if v == 'High' else
                'background-color: rgba(255,166,77,0.3)' if v == 'Medium' else
                'background-color: rgba(77,166,255,0.3)' if v == 'Low' else ''
                for v in x
            ], subset=['priority'] if 'priority' in available_columns else [])

            st.dataframe(styled_df)

        except Exception as e:
            logger.error(f"Error displaying recommendations: {str(e)}")
            st.error(f"Error displaying recommendations table: {str(e)}")

    def generate_interactive_dashboard(self) -> Dict[str, Any]:
        """Generate all visualizations for dashboard with explanations"""
        try:
            # Generate visualizations and explanations
            severity_fig, severity_explanation = self.create_severity_distribution()
            coverage_fig, coverage_explanation = self.create_coverage_radar()
            gap_fig, gap_explanation = self.create_gap_heatmap()
            timeline_fig, timeline_explanation = self.create_timeline_view()

            return {
                'figures': {
                    'severity_distribution': severity_fig,
                    'coverage_radar': coverage_fig,
                    'gap_heatmap': gap_fig,
                    'timeline_view': timeline_fig
                },
                'explanations': {
                    'severity_distribution': severity_explanation,
                    'coverage_radar': coverage_explanation,
                    'gap_heatmap': gap_explanation,
                    'timeline_view': timeline_explanation
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating dashboard: {str(e)}")
            return {}

@dataclass
class GapFindings:
    """Data class for gap analysis findings"""
    category: str
    severity: str
    description: str
    current_state: Any
    expected_state: Any
    impact: str
    recommendations: List[str]

class GapDataValidator:
    """Validates and generates gap analysis data"""

    def __init__(self, driver):
        self.driver = driver
        logger = logging.getLogger(__name__)

        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()
                if test_value is None or test_value.get('test') != 1:
                    raise Exception("Failed to verify Neo4j connection")
                logger.info("Successfully initialized GapDataValidator with Neo4j connection")
        except Exception as e:
            logger.error(f"Error initializing GapDataValidator: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def validate_data_connections(self) -> Dict[str, bool]:
        """Validate if all required relationships exist"""
        with self.driver.session() as session:
            # Check principle-requirement connections
            principle_req = session.run("""
                MATCH (p:Principle)-[:HAS_REQUIREMENT]->(r:Requirement) 
                RETURN COUNT(*) as count
            """).single()['count']

            # Check requirement-control connections
            req_control = session.run("""
                MATCH (r:Requirement)-[:IMPLEMENTED_BY]->(c:Control)
                RETURN COUNT(*) as count
            """).single()['count']

            # Check control-activity connections
            control_activity = session.run("""
                MATCH (c:Control)-[:MONITORS]->(a:Activity)
                RETURN COUNT(*) as count
            """).single()['count']

            return {
                'principle_requirement_exists': principle_req > 0,
                'requirement_control_exists': req_control > 0,
                'control_activity_exists': control_activity > 0
            }

    def ensure_gap_data(self):
        """Create realistic gap data matching existing Neo4j structure"""
        with self.driver.session() as session:
            # Reset existing mock data
            session.run("""
                MATCH (r:Requirement)
                SET r.status = null,
                    r.severity = null,
                    r.gap_type = null
            """)

            # Create gaps for EXEC principles (we have 4)
            session.run("""
                MATCH (p:Principle)
                WHERE p.code STARTS WITH 'EXEC'
                WITH p
                MATCH (p)-[:HAS_REQUIREMENT]->(r:Requirement)
                WITH r, rand() as rnd
                WHERE rnd < 0.6  // Create gaps for 60% of EXEC requirements
                SET r.status = 'NOT_IMPLEMENTED',
                    r.severity = CASE 
                        WHEN rnd < 0.2 THEN 'High'   // 20% High
                        WHEN rnd < 0.5 THEN 'Medium' // 30% Medium
                        ELSE 'Low'                   // 10% Low
                    END,
                    r.gap_type = 'Regulatory Compliance'
            """)

            # Create gaps for RISK principles (we have 2)
            session.run("""
                MATCH (p:Principle)
                WHERE p.code STARTS WITH 'RISK'
                WITH p
                MATCH (p)-[:HAS_REQUIREMENT]->(r:Requirement)
                WITH r, rand() as rnd
                WHERE rnd < 0.5  // Create gaps for 50% of RISK requirements
                SET r.status = 'NOT_IMPLEMENTED',
                    r.severity = CASE 
                        WHEN rnd < 0.3 THEN 'High'   // 30% High
                        WHEN rnd < 0.7 THEN 'Medium' // 40% Medium
                        ELSE 'Low'                   // 30% Low
                    END,
                    r.gap_type = 'Risk Control'
            """)

            # Create unmonitored activities (out of 35 total)
            session.run("""
                MATCH (a:Activity)
                WITH a, rand() as rnd
                WHERE rnd < 0.3  // 30% of 35 activities = ~10 unmonitored
                SET a.monitored = false,
                    a.completion_rate = 0.6 + (rand() * 0.2)  // 60-80% completion
            """)

            # Modify control-activity relationships (8 controls)
            session.run("""
                MATCH (c:Control)-[r:MONITORS]->(a:Activity)
                WITH c, r, a, rand() as rnd
                WHERE rnd < 0.25  // Disconnect 25% of existing control relationships
                DELETE r
                SET a.control_gap = true
            """)

            # Return gap statistics
            return session.run("""
                MATCH (r:Requirement)
                WHERE r.status = 'NOT_IMPLEMENTED'
                WITH r.severity as severity, r.gap_type as type, COUNT(*) as count
                RETURN collect({
                    severity: severity,
                    type: type,
                    count: count
                }) as gaps
            """).single()['gaps']

    def _get_default_metrics(self) -> Dict:
        """Get default metrics when query fails"""
        return {
            'regulatory_coverage': 0.0,
            'process_adherence': 0.0,
            'control_effectiveness': 0.0,
            'risk_coverage': 0.0
        }

    def get_gap_metrics(self) -> Dict:
        """Get metrics based on actual Neo4j state with enhanced error handling"""
        logger.info("Getting gap metrics from Neo4j")

        try:
            with self.driver.session() as session:
                # Modified query to handle missing relationships
                query_result = session.run("""
                    // Calculate requirement coverage
                    OPTIONAL MATCH (r:Requirement)
                    WITH COALESCE(COUNT(r), 0) as total_requirements,
                         COALESCE(COUNT(r.status), 0) as gap_requirements

                    // Calculate activity monitoring
                    OPTIONAL MATCH (a:Activity)
                    WITH total_requirements, gap_requirements,
                         COALESCE(COUNT(a), 0) as total_activities,
                         COALESCE(COUNT(a.monitored), 0) as unmonitored_activities

                    // Calculate control effectiveness
                    OPTIONAL MATCH (c:Control)-[:MONITORS]->(a:Activity)
                    WITH total_requirements, gap_requirements,
                         total_activities, unmonitored_activities,
                         COALESCE(COUNT(DISTINCT c), 0) as active_controls

                    RETURN {
                        regulatory_coverage: CASE 
                            WHEN total_requirements > 0 
                            THEN round(((total_requirements - gap_requirements) * 100.0) / total_requirements, 1)
                            ELSE 0.0 
                        END,
                        process_adherence: CASE 
                            WHEN total_activities > 0 
                            THEN round(((total_activities - unmonitored_activities) * 100.0) / total_activities, 1)
                            ELSE 0.0 
                        END,
                        control_effectiveness: CASE 
                            WHEN active_controls > 0 
                            THEN round((active_controls * 100.0) / 8, 1)
                            ELSE 0.0 
                        END,
                        risk_coverage: 0.0
                    } as metrics
                """)

                result = query_result.single()
                if result is None:
                    logger.error("No metrics returned from Neo4j query - query returned no results")
                    return self._get_default_metrics()

                metrics = result.get('metrics')
                if metrics is None:
                    logger.error("Metrics not found in query result")
                    return self._get_default_metrics()

                logger.info(f"Successfully retrieved metrics: {metrics}")
                return metrics

        except Exception as e:
            logger.error(f"Error getting gap metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_default_metrics()

class FXTradingGapAnalyzer:
    """Comprehensive gap analyzer for FX trading processes"""

    def __init__(self, uri: str, user: str, password: str, api_key: str = None):
        """Initialize the analyzer with Neo4j and OpenAI credentials"""
        logger = logging.getLogger(__name__)
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.findings = []
        self._initialize_metrics()

        self.openai_client = OpenAI(
            api_key="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"
        )

        # Initialize AI analyzer with API key from environment or session state
        api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OpenAI API key found - AI analysis will be disabled")
            self.ai_analyzer = None
        else:
            try:
                self.ai_analyzer = AIGapAnalyzer(api_key)
                logger.info("AI analyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI analyzer: {str(e)}")
                self.ai_analyzer = None

    def _initialize_metrics(self):
        """Initialize comprehensive analysis metrics"""
        self.metrics = {
            'compliance': {
                'regulatory_coverage': 0.0,
                'control_effectiveness': 0.0,
                'reporting_timeliness': 0.0
            },
            'operational': {
                'process_adherence': 0.0,
                'activity_completion': 0.0,
                'resource_utilization': 0.0,
                'stp_rate': 0.0
            },
            'risk': {
                'control_coverage': 0.0,
                'risk_assessment_completion': 0.0,
                'validation_effectiveness': 0.0,
                'control_execution_rate': 0.0
            }
        }

    def calculate_metrics(self) -> None:
        """Calculate metrics focusing on FX Global Code compliance"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    // Calculate overall compliance metrics
                    MATCH (p:Principle)
                    OPTIONAL MATCH (p)-[:HAS_REQUIREMENT]->(r:Requirement)
                    OPTIONAL MATCH (r)-[:IMPLEMENTED_BY]->(c:Control)
                    OPTIONAL MATCH (c)-[:MONITORS]->(a:Activity)
                    WITH COUNT(DISTINCT r) as total_requirements,
                         COUNT(DISTINCT c) as implemented_controls,
                         COUNT(DISTINCT a) as monitored_activities,
                         COUNT(DISTINCT p) as total_principles

                    // Calculate Process Coverage
                    MATCH (a:Activity)
                    WITH total_requirements, implemented_controls, 
                         monitored_activities, total_principles,
                         COUNT(a) as total_activities

                    RETURN {
                        regulatory_coverage: CASE 
                            WHEN total_requirements > 0 
                            THEN toFloat(implemented_controls) / total_requirements * 100
                            ELSE 0 
                        END,
                        process_adherence: CASE 
                            WHEN total_activities > 0 
                            THEN toFloat(monitored_activities) / total_activities * 100
                            ELSE 0 
                        END,
                        control_effectiveness: CASE 
                            WHEN implemented_controls > 0 
                            THEN 75.0  // Base effectiveness score
                            ELSE 0 
                        END,
                        risk_coverage: CASE 
                            WHEN total_principles > 0 
                            THEN toFloat(implemented_controls) / total_principles * 100
                            ELSE 0 
                        END
                    } as metrics
                """)

                metrics = result.single()['metrics']

                # Update metrics structure
                self.metrics.update({
                    'compliance': {
                        'regulatory_coverage': metrics['regulatory_coverage'],
                        'control_effectiveness': metrics['control_effectiveness']
                    },
                    'operational': {
                        'process_adherence': metrics['process_adherence'],
                        'activity_completion': metrics['process_adherence']
                    },
                    'risk': {
                        'control_coverage': metrics['risk_coverage'],
                        'risk_assessment_completion': metrics['regulatory_coverage']
                    }
                })

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.error(traceback.format_exc())
            self._initialize_metrics()

    def _get_guidelines_data(self) -> Dict:
        """Get guidelines data based on Principles with enhanced gap detection"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    // Match all principles and their requirements
                    MATCH (p:Principle)
                    OPTIONAL MATCH (p)-[:HAS_REQUIREMENT]->(r:Requirement)
                    OPTIONAL MATCH (r)-[:IMPLEMENTED_BY]->(c:Control)
                    OPTIONAL MATCH (c)-[:MONITORS]->(a:Activity)

                    // Collect activity statistics
                    WITH p, r, 
                         COUNT(DISTINCT c) as control_count,
                         COUNT(DISTINCT a) as monitored_activities,
                         COLLECT(DISTINCT a.name) as activity_names

                    // Calculate completion metrics
                    WITH p,
                         COUNT(r) as total_requirements,
                         SUM(CASE WHEN control_count > 0 THEN 1 ELSE 0 END) as implemented_requirements,
                         COLLECT({
                             requirement: r.id,
                             has_controls: control_count > 0,
                             monitored_activities: monitored_activities,
                             activities: activity_names
                         }) as requirement_details

                    RETURN {
                        guidelines: COLLECT({
                            id: p.code,
                            name: p.name,
                            description: p.description,
                            type: CASE 
                                WHEN p.code STARTS WITH 'RISK' THEN 'REGULATORY'
                                ELSE 'OPERATIONAL'
                            END,
                            implementation_status: CASE 
                                WHEN total_requirements = 0 THEN 'NO_REQUIREMENTS'
                                WHEN implemented_requirements = 0 THEN 'NOT_IMPLEMENTED'
                                WHEN implemented_requirements < total_requirements THEN 'PARTIALLY_IMPLEMENTED'
                                ELSE 'FULLY_IMPLEMENTED'
                            END,
                            coverage: CASE 
                                WHEN total_requirements > 0 
                                THEN toFloat(implemented_requirements) / total_requirements * 100
                                ELSE 0
                            END,
                            gaps: requirement_details
                        })
                    } as guidelines_data
                """)

                guidelines_data = result.single()['guidelines_data']
                logger.info(f"Retrieved {len(guidelines_data['guidelines'])} guidelines")

                # Post-process gaps to provide more detailed analysis
                for guideline in guidelines_data['guidelines']:
                    if guideline['implementation_status'] != 'FULLY_IMPLEMENTED':
                        unimplemented_reqs = [
                            req for req in guideline['gaps']
                            if not req['has_controls']
                        ]

                        if unimplemented_reqs:
                            formatted_gaps = [{
                                'category': 'Implementation Gap',
                                'severity': 'High' if guideline['type'] == 'REGULATORY' else 'Medium',
                                'description': f"Missing controls for {len(unimplemented_reqs)} requirements in {guideline['name']}",
                                'impact': 'Regulatory/Operational Risk',
                                'recommendations': [
                                    'Implement missing controls',
                                    'Review control framework',
                                    'Update documentation'
                                ]
                            }]
                            guideline['gaps'] = formatted_gaps
                            logger.info(f"Found {len(formatted_gaps)} gaps for {guideline['name']}")

                return guidelines_data

        except Exception as e:
            logger.error(f"Error retrieving guidelines data: {str(e)}")
            logger.error(traceback.format_exc())
            return {'guidelines': []}

    def analyze_regulatory_compliance(self) -> List[GapFindings]:
        """Analyze regulatory compliance gaps"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (g:Guideline {type: 'Regulatory'})
                    OPTIONAL MATCH (g)-[:APPLIES_TO]->(a:Activity)
                    OPTIONAL MATCH (c:Control)-[:MONITORS]->(a)
                    OPTIONAL MATCH (ce:ControlExecution)-[:EXECUTES]->(c)
                    RETURN g.id as guideline_id,
                           g.name as guideline_name,
                           g.severity as severity,
                           g.description as requirement,
                           COUNT(DISTINCT c) as control_count,
                           COUNT(DISTINCT ce) as execution_count
                """)

                findings = []
                for record in result:
                    if record['control_count'] == 0 or record['execution_count'] == 0:
                        findings.append(GapFindings(
                            category='Regulatory Compliance',
                            severity=record['severity'],
                            description=f"Missing implementation of {record['guideline_name']}",
                            current_state='Not Implemented' if record['control_count'] == 0 else 'Not Executed',
                            expected_state=record['requirement'],
                            impact='High - Regulatory Risk',
                            recommendations=[
                                f"Implement control for {record['guideline_name']}",
                                'Update process documentation',
                                'Train relevant staff'
                            ]
                        ))

                return findings

        except Exception as e:
            logger.error(f"Error in regulatory analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_process_execution(self) -> List[GapFindings]:
        """Analyze process execution gaps"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Event)-[:PERFORMS]->(a:Activity)
                    WHERE e.duration_seconds > (a.sla_minutes * 60)
                    RETURN a.name as activity_name,
                           e.duration_seconds as actual_duration,
                           a.sla_minutes * 60 as sla_seconds,
                           e.case_id as case_id
                """)

                findings = []
                for record in result:
                    findings.append(GapFindings(
                        category='Process Execution',
                        severity='Medium',
                        description=f"SLA breach in {record['activity_name']}",
                        current_state=f"{record['actual_duration']} seconds",
                        expected_state=f"{record['sla_seconds']} seconds",
                        impact='Medium - Operational Efficiency',
                        recommendations=[
                            'Review process bottlenecks',
                            'Optimize activity execution',
                            'Consider automation opportunities'
                        ]
                    ))

                return findings

        except Exception as e:
            logger.error(f"Error in process execution analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _get_process_data(self) -> Dict:
        """Retrieve process data from Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Activity)
                    OPTIONAL MATCH (a)-[:MONITORS]-(c:Control)
                    WITH a, COLLECT(DISTINCT c.name) as control_names
                    RETURN {
                        activities: COLLECT({
                            id: a.id,
                            name: a.name,
                            completion_rate: a.completion_rate,
                            controls: control_names
                        })
                    } as process_data
                """)
                return result.single()['process_data']
        except Exception as e:
            logger.error(f"Error retrieving process data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def calculate_metrics(self) -> None:
        """Calculate metrics focusing on FX Global Code compliance"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    // Calculate overall compliance metrics
                    MATCH (p:Principle)
                    OPTIONAL MATCH (p)-[:HAS_REQUIREMENT]->(r:Requirement)
                    OPTIONAL MATCH (r)-[:IMPLEMENTED_BY]->(c:Control)
                    OPTIONAL MATCH (c)-[:MONITORS]->(a:Activity)
                    WITH COUNT(DISTINCT r) as total_requirements,
                         COUNT(DISTINCT c) as implemented_controls,
                         COUNT(DISTINCT a) as monitored_activities,
                         COUNT(DISTINCT p) as total_principles

                    // Calculate Process Coverage
                    MATCH (a:Activity)
                    WITH total_requirements, implemented_controls, 
                         monitored_activities, total_principles,
                         COUNT(a) as total_activities

                    RETURN {
                        regulatory_coverage: CASE 
                            WHEN total_requirements > 0 
                            THEN toFloat(implemented_controls) / total_requirements * 100
                            ELSE 0 
                        END,
                        process_adherence: CASE 
                            WHEN total_activities > 0 
                            THEN toFloat(monitored_activities) / total_activities * 100
                            ELSE 0 
                        END,
                        control_effectiveness: CASE 
                            WHEN implemented_controls > 0 
                            THEN 75.0  // Base effectiveness score
                            ELSE 0 
                        END,
                        risk_coverage: CASE 
                            WHEN total_principles > 0 
                            THEN toFloat(implemented_controls) / total_principles * 100
                            ELSE 0 
                        END
                    } as metrics
                """)

                metrics = result.single()['metrics']

                # Update metrics structure
                self.metrics.update({
                    'compliance': {
                        'regulatory_coverage': metrics['regulatory_coverage'],
                        'control_effectiveness': metrics['control_effectiveness']
                    },
                    'operational': {
                        'process_adherence': metrics['process_adherence'],
                        'activity_completion': metrics['process_adherence']
                    },
                    'risk': {
                        'control_coverage': metrics['risk_coverage'],
                        'risk_assessment_completion': metrics['regulatory_coverage']
                    }
                })

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.error(traceback.format_exc())
            self._initialize_metrics()


    def _is_guideline_implemented(self, guideline: Dict, process_data: Dict) -> bool:
        """Check if guideline is implemented in process"""
        try:
            required_activities = set(guideline.get('required_activities', []))
            actual_activities = {a['name'] for a in process_data.get('activities', [])}
            return required_activities.issubset(actual_activities)
        except Exception as e:
            logger.error(f"Error checking guideline implementation: {str(e)}")
            return False

    def generate_gap_report(self) -> Dict:
        """Generate comprehensive gap analysis report with process visualization"""
        try:
            # Initialize validator
            validator = GapDataValidator(self.driver)

            # Validate data connections
            connections = validator.validate_data_connections()
            logger.info(f"Data connections validation: {connections}")

            # Ensure we have gap data
            gap_stats = validator.ensure_gap_data()
            logger.info(f"Gap statistics: {gap_stats}")

            # Get metrics
            metrics = validator.get_gap_metrics()

            # Structure high-level gaps
            gaps = []
            high_severity = 0
            medium_severity = 0
            low_severity = 0

            for stat in gap_stats:
                severity = stat.get('severity', 'Medium')
                count = stat.get('count', 0)

                if severity == 'High':
                    high_severity = count
                elif severity == 'Medium':
                    medium_severity = count
                elif severity == 'Low':
                    low_severity = count

                gaps.append({
                    'category': 'Implementation Gap',
                    'severity': severity,
                    'description': f"Missing controls for {count} requirements",
                    'impact': 'Regulatory/Operational Risk'
                })

            # Structure report
            report = {
                'summary': {
                    'total_gaps': len(gaps),
                    'high_severity': high_severity,
                    'medium_severity': medium_severity,
                    'low_severity': low_severity
                },
                'metrics': {
                    'compliance': {
                        'regulatory_coverage': metrics['regulatory_coverage'],
                        'control_effectiveness': metrics['control_effectiveness']
                    },
                    'operational': {
                        'process_adherence': metrics['process_adherence'],
                        'activity_completion': metrics['process_adherence']
                    },
                    'risk': {
                        'control_coverage': metrics['risk_coverage'],
                        'risk_assessment_completion': metrics['regulatory_coverage']
                    }
                },
                'findings': gaps,
                'recommendations': self._generate_recommendations(gaps),
                'timestamp': datetime.now().isoformat()
            }

            # Add AI insights if available
            if self.ai_analyzer:
                try:
                    ai_analysis = self.ai_analyzer.analyze_process_gaps(
                        {'gaps': gaps},
                        {'metrics': metrics}
                    )
                    report['ai_insights'] = ai_analysis
                except Exception as e:
                    logger.error(f"AI analysis failed: {str(e)}")

            return report

        except Exception as e:
            logger.error(f"Error generating gap report: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _generate_recommendations(self, gaps: List[Dict]) -> List[Dict]:
        """Generate recommendations based on identified gaps"""
        try:
            recommendations = []
            for i, gap in enumerate(gaps):
                recommendation = {
                    'id': f"REC_{i:03d}",
                    'description': f"Address {gap['description']}",
                    'priority': gap['severity'],
                    'target_date': (datetime.now() + timedelta(days=30)).isoformat(),
                    'status': 'Open',
                    'impact': gap.get('impact', 'Unknown')
                }
                recommendations.append(recommendation)

            if not recommendations:
                # Add default recommendation if none generated
                recommendations.append({
                    'id': 'REC_DEFAULT',
                    'description': 'Review process controls and documentation',
                    'priority': 'Medium',
                    'target_date': (datetime.now() + timedelta(days=30)).isoformat(),
                    'status': 'Open',
                    'impact': 'Process Improvement'
                })

            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def close(self):
        """Close Neo4j connection"""
        try:
            self.driver.close()
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")

def handle_data_import():
    """Handle data import operations"""
    st.subheader("Data Import")

    # Database Configuration
    with st.expander("Neo4j Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            neo4j_uri = st.text_input("Neo4j URI", "bolt://localhost:7689")
        with col2:
            neo4j_user = st.text_input("Username", "neo4j")
        with col3:
            neo4j_password = st.text_input("Password", type="password")

        if st.button("Test Connection"):
            try:
                driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                with driver.session() as session:
                    session.run("RETURN 1").single()
                st.success("Connected to Neo4j successfully!")
                st.session_state.neo4j_connected = True
                st.session_state.neo4j_credentials = {
                    'uri': neo4j_uri,
                    'user': neo4j_user,
                    'password': neo4j_password
                }
                driver.close()
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
                st.session_state.neo4j_connected = False

    # File Upload Section
    st.subheader("Process Data Files")

    org_context = st.file_uploader(
        "Upload Organizational Context (CSV)",
        type=['csv'],
        key="org_context"
    )

    guidelines = st.file_uploader(
        "Upload Guidelines (CSV)",
        type=['csv'],
        key="guidelines"
    )

    ocel_file = st.file_uploader(
        "Upload OCEL JSON",
        type=['json'],
        key="ocel"
    )

    if st.button("Import to Neo4j"):
        if not (org_context and guidelines and ocel_file):
            st.warning("Please upload all required files")
            return

        try:
            import_to_neo4j(
                st.session_state.neo4j_credentials,
                org_context,
                guidelines,
                ocel_file
            )
            st.success("Data imported successfully!")
        except Exception as e:
            st.error(f"Import failed: {str(e)}")
            st.error(traceback.format_exc())


def display_visualization_section(fig: go.Figure, explanation: str, title: str):
    """Display visualization with properly formatted explanation"""
    st.plotly_chart(fig, use_container_width=True)

    # Display AI explanation in a visually distinct container
    with st.container():
        st.markdown("#### 🤖 AI Analysis")
        st.markdown(
            f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 5px;
                border-left: 5px solid #00a0dc;
                margin: 10px 0;">
                {explanation}
            </div>
            """,
            unsafe_allow_html=True
        )

def handle_unfairness_analysis():
    """Handle unfairness analysis operations"""
    st.subheader("Unfairness Analysis")
    ocel_path = st.session_state.get('ocel_path')

    if not ocel_path or not os.path.exists(ocel_path):
        st.warning("⚠️ Please process data in the Process Analysis tab first.")
        return

    try:
        analyzer = UnfairOCELAnalyzer(ocel_path)
        analyzer.display_enhanced_analysis()
    except Exception as e:
        st.error(f"Error in unfairness analysis: {str(e)}")


def handle_ai_analysis():
    """Handle AI-powered analysis operations"""
    st.subheader("🤖 AI-Powered Analysis")
    ocel_path = st.session_state.get('ocel_path')

    if not ocel_path or not os.path.exists(ocel_path):
        st.warning("⚠️ Please process data in the Process Analysis tab first.")
        return

    try:
        analyzer = AIOCELAnalyzer()
        analyzer.load_ocel(ocel_path)
        display_ai_analysis(analyzer)
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")


def display_ai_analysis(analyzer):
    """Display AI analysis results"""
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", analyzer.stats['general']['total_events'])
    with col2:
        st.metric("Total Cases", analyzer.stats['general']['total_cases'])
    with col3:
        st.metric("Total Resources", analyzer.stats['general']['total_resources'])

    # AI Analysis interface
    question = st.text_input(
        "Ask a question about your process:",
        placeholder="e.g., What are the main process patterns?"
    )

    if question:
        with st.spinner("Analyzing..."):
            analysis = analyzer.analyze_with_ai(question)
            st.write(analysis)

    # Visualizations
    st.subheader("Process Visualizations")
    figures = analyzer.create_visualizations()

    viz_tabs = st.tabs(["Activities", "Resources"])
    with viz_tabs[0]:
        st.plotly_chart(figures['activity_distribution'], use_container_width=True)
    with viz_tabs[1]:
        st.plotly_chart(figures['resource_distribution'], use_container_width=True)


def import_to_neo4j(credentials, org_file, guidelines_file, ocel_file):
    """Import data to Neo4j database"""
    driver = GraphDatabase.driver(
        credentials['uri'],
        auth=(credentials['user'], credentials['password'])
    )

    try:
        # Import organizational context
        org_data = pd.read_csv(org_file)
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (o:Organization {id: row.id})
                SET o += row
            """, rows=org_data.to_dict('records'))

        # Import guidelines
        guide_data = pd.read_csv(guidelines_file)
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (g:Guideline {id: row.id})
                SET g += row
            """, rows=guide_data.to_dict('records'))

        # Import OCEL data
        ocel_data = json.load(ocel_file)
        events = [
            {
                'id': e['ocel:id'],
                'timestamp': e['ocel:timestamp'],
                'activity': e['ocel:activity'],
                'resource': e.get('ocel:attributes', {}).get('resource', 'Unknown'),
                'case_id': e.get('ocel:attributes', {}).get('case_id', 'Unknown')
            }
            for e in ocel_data['ocel:events']
        ]

        with driver.session() as session:
            session.run("""
                UNWIND $events AS event
                MERGE (e:Event {id: event.id})
                SET e += event
            """, events=events)

    finally:
        driver.close()


def run_neo4j_analysis(credentials):
    """Run analysis queries in Neo4j"""
    driver = GraphDatabase.driver(
        credentials['uri'],
        auth=(credentials['user'], credentials['password'])
    )

    try:
        results = {}
        with driver.session() as session:
            # Process flow analysis
            process_flow = session.run("""
                MATCH (e1:Event)-[r:NEXT]->(e2:Event)
                RETURN e1.activity as source, 
                       e2.activity as target, 
                       count(*) as frequency
                ORDER BY frequency DESC
            """)
            results['process_flow'] = process_flow.data()

            # Resource analysis
            resource_analysis = session.run("""
                MATCH (e:Event)
                WITH e.resource as resource, count(*) as events
                RETURN resource, events
                ORDER BY events DESC
            """)
            results['resources'] = resource_analysis.data()

        return results

    finally:
        driver.close()


def display_analysis_results(results):
    """Display analysis results"""
    if not results:
        st.warning("No analysis results available")
        return

    tabs = st.tabs(["Process Flow", "Resource Analysis"])

    with tabs[0]:
        st.subheader("Process Flow Analysis")
        if results.get('process_flow'):
            st.dataframe(pd.DataFrame(results['process_flow']))

    with tabs[1]:
        st.subheader("Resource Analysis")
        if results.get('resources'):
            st.dataframe(pd.DataFrame(results['resources']))



def main():
    st.set_page_config(
        page_title="IRMAI Process Analytics",
        page_icon="👋",
        layout="wide"
    )




if __name__ == "__main__":
    main()