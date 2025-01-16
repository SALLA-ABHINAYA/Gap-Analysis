import logging

import numpy as np
import streamlit as st
from neo4j import GraphDatabase
import traceback
import json
import pandas as pd
from openai import OpenAI
from pathlib import Path
import os
from Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer
from ocpm_analysis import create_ocpm_ui
from ai_ocel_analyzer import AIOCELAnalyzer
import plotly.graph_objects as go
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import plotly.express as px
from collections import defaultdict

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
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def analyze_process_gaps(self, process_data: Dict, guidelines: Dict) -> Dict:
        """Use AI to analyze gaps between process and guidelines"""
        try:
            # Prepare context for AI analysis
            context = f"""
            Process Mining Expert Analysis Task:

            Current Process Data:
            {json.dumps(process_data, indent=2)}

            Process Guidelines:
            {json.dumps(guidelines, indent=2)}

            Task: Analyze the gaps between the current process and guidelines. Consider:
            1. Regulatory compliance gaps
            2. Process execution inefficiencies
            3. Control effectiveness issues
            4. Resource utilization problems
            5. Risk management gaps

            Provide analysis and recommendations in JSON format with:
            - Identified gaps
            - Severity levels
            - Potential impacts
            - Specific recommendations
            - Implementation priorities
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an expert process mining and regulatory compliance analyst."},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # Parse AI response
            analysis = json.loads(response.choices[0].message.content)

            # Validate and structure the analysis
            structured_analysis = self._structure_analysis(analysis)

            return structured_analysis

        except Exception as e:
            logging.error(f"Error in AI gap analysis: {str(e)}")
            raise
   
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
        self.findings = pd.DataFrame(report['findings'])
        self.metrics = pd.DataFrame.from_dict(report['metrics'], orient='index')

    def create_severity_distribution(self) -> go.Figure:
        """Create severity distribution chart"""
        summary = self.report['summary']

        fig = go.Figure(data=[
            go.Bar(
                name='Gaps',
                x=['High', 'Medium', 'Low'],
                y=[summary['high_severity'],
                   summary['medium_severity'],
                   summary['low_severity']],
                marker_color=['#ff4d4d', '#ffa64d', '#4da6ff']
            )
        ])

        fig.update_layout(
            title='Gap Severity Distribution',
            xaxis_title='Severity Level',
            yaxis_title='Number of Gaps',
            template='plotly_white'
        )

        return fig

    def create_coverage_radar(self) -> go.Figure:
        """Create enhanced radar chart for coverage metrics"""
        metrics = self.report['metrics']

        # Extract values with proper error handling
        values = [
            round(metrics['compliance'].get('regulatory_coverage', 0), 1),
            round(metrics['operational'].get('process_adherence', 0), 1),
            round(metrics['risk'].get('control_coverage', 0), 1),
            round(metrics['risk'].get('risk_assessment_completion', 0), 1)
        ]

        categories = [
            'Regulatory Coverage',
            'Process Adherence',
            'Control Effectiveness',
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
            title='Coverage Analysis',
            height=400,
            template='plotly_white'
        )

        # Add percentage annotations
        for i, value in enumerate(values):
            angle = (i / len(values)) * 2 * np.pi
            fig.add_annotation(
                x=0.5 + 0.45 * np.cos(angle),
                y=0.5 + 0.45 * np.sin(angle),
                text=f'{value:.1f}%',
                showarrow=False,
                font=dict(size=12)
            )

        return fig

    def create_gap_heatmap(self) -> go.Figure:
        """Create heatmap of gaps by category and severity"""
        if len(self.findings) == 0:
            # Create empty heatmap if no findings
            fig = go.Figure(data=go.Heatmap(
                z=[[0]],
                x=['No Data'],
                y=['No Data'],
                colorscale='RdYlBu_r'
            ))
        else:
            # Pivot findings data
            heatmap_data = pd.crosstab(
                self.findings['category'],
                self.findings['severity']
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

        return fig

    def display_recommendations_table(self) -> None:
        """Display detailed recommendations table"""
        recommendations = self.report.get('recommendations', [])

        if not recommendations:
            st.info("No recommendations data available")
            return

        df = pd.DataFrame(recommendations)

        # Format the DataFrame for display
        display_df = df[[
            'priority', 'description', 'target_date', 'status', 'impact', 'control_name'
        ]].copy()

        # Apply styling
        styled_df = display_df.style.apply(lambda x: [
            'background-color: rgba(255,77,77,0.3)' if v == 'High' else
            'background-color: rgba(255,166,77,0.3)' if v == 'Medium' else
            'background-color: rgba(77,166,255,0.3)' if v == 'Low' else ''
            for v in x
        ], subset=['priority'])

        st.dataframe(styled_df)

    def create_timeline_view(self) -> go.Figure:
        """Create enhanced timeline view of recommendations"""
        recommendations = self.report.get('recommendations', [])

        logging.info(f"Creating timeline view with {len(recommendations)} recommendations")

        if not recommendations:
            logging.warning("No recommendations data available for timeline")
            fig = go.Figure()
            fig.update_layout(
                title='Recommendation Timeline (No Data)',
                xaxis_title='Timeline',
                yaxis_title='Priority',
                height=400,
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                annotations=[{
                    'text': 'No recommendations data available',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': dict(color='white', size=14)
                }]
            )
            return fig

        try:
            df = pd.DataFrame(recommendations)
            logging.info(f"Created DataFrame with columns: {df.columns.tolist()}")
            logging.info(f"DataFrame shape: {df.shape}")

            # Ensure target_date is datetime
            df['target_date'] = pd.to_datetime(df['target_date'])

            # Log unique priorities
            logging.info(f"Unique priorities found: {df['priority'].unique().tolist()}")

            color_map = {
                'High': '#ff4d4d',
                'Medium': '#ffa64d',
                'Low': '#4da6ff'
            }

            fig = go.Figure()

            for priority in ['High', 'Medium', 'Low']:
                mask = df['priority'] == priority
                if any(mask):
                    logging.info(f"Adding {sum(mask)} recommendations for priority {priority}")
                    fig.add_trace(go.Scatter(
                        x=df[mask]['target_date'],
                        y=[priority] * sum(mask),
                        mode='markers+text',
                        name=priority,
                        marker=dict(
                            size=15,
                            color=color_map[priority],
                            symbol='circle'
                        ),
                        text=df[mask]['description'],
                        textposition='middle right',
                        hovertemplate=(
                                '<b>%{text}</b><br>' +
                                'Target Date: %{x|%Y-%m-%d}<br>' +
                                'Priority: ' + priority + '<br>' +
                                '<extra></extra>'
                        )
                    ))

            fig.update_layout(
                title='Recommendation Timeline',
                xaxis_title='Target Date',
                yaxis_title='Priority Level',
                height=400,
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(
                    categoryorder='array',
                    categoryarray=['Low', 'Medium', 'High']
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    font=dict(color='white')
                ),
                margin=dict(r=250)
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

            return fig

        except Exception as e:
            logging.error(f"Error creating timeline visualization: {str(e)}")
            logging.error(traceback.format_exc())
            # Return empty figure with error message
            fig = go.Figure()
            fig.update_layout(
                title='Error Creating Timeline',
                annotations=[{
                    'text': f'Error: {str(e)}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': dict(color='red', size=14)
                }]
            )
            return fig

    def generate_interactive_dashboard(self) -> Dict[str, go.Figure]:
        """Generate all visualizations for dashboard"""
        return {
            'severity_distribution': self.create_severity_distribution(),
            'coverage_radar': self.create_coverage_radar(),
            'gap_heatmap': self.create_gap_heatmap(),
            'timeline_view': self.create_timeline_view()
        }



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


class FXTradingGapAnalyzer:
    """Comprehensive gap analyzer for FX trading processes"""

    def __init__(self, uri: str, user: str, password: str, api_key: str = None):
        """Initialize the analyzer with Neo4j and OpenAI credentials"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.findings = []
        self._initialize_metrics()

        # Initialize AI analyzer with API key from environment or session state
        api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            logging.warning("No OpenAI API key found - AI analysis will be disabled")
            self.ai_analyzer = None
        else:
            try:
                self.ai_analyzer = AIGapAnalyzer(api_key)
                logging.info("AI analyzer initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize AI analyzer: {str(e)}")
                self.ai_analyzer = None

    def analyze_recommendations(self) -> List[Dict]:
        """Analyze and fetch recommendations from Neo4j"""
        try:
            with self.driver.session() as session:
                # Modified query to get all recommendations regardless of Control relationship
                result = session.run("""
                    MATCH (r:Recommendation)
                    OPTIONAL MATCH (r)-[:IMPROVES]->(c:Control)
                    RETURN 
                        r.id as id,
                        r.description as description,
                        r.priority as priority,
                        r.target_date as target_date,
                        r.status as status,
                        r.impact as impact,
                        CASE WHEN c IS NOT NULL THEN c.name ELSE null END as control_name
                    ORDER BY 
                        CASE r.priority 
                            WHEN 'High' THEN 1 
                            WHEN 'Medium' THEN 2 
                            ELSE 3 
                        END,
                        r.target_date
                """)

                recommendations = []
                for record in result:
                    recommendations.append({
                        'id': record['id'],
                        'description': record['description'],
                        'priority': record['priority'],
                        'target_date': record['target_date'].isoformat() if record['target_date'] else None,
                        'status': record['status'],
                        'impact': record['impact'],
                        'control_name': record['control_name'] or 'No Control Associated'
                    })

                # Debug logging
                logging.info(f"Found {len(recommendations)} recommendations")
                for rec in recommendations:
                    logging.info(f"Recommendation: {rec['id']} - {rec['description']} - {rec['target_date']}")

                return recommendations

        except Exception as e:
            logging.error(f"Error analyzing recommendations: {str(e)}")
            logging.error(traceback.format_exc())
            return []

    def calculate_metrics(self) -> None:
        """Calculate actual metrics from Neo4j data"""
        try:
            with self.driver.session() as session:
                # Calculate control effectiveness
                control_result = session.run("""
                    MATCH (c:Control)
                    WITH avg(c.effectiveness_score) as avg_effectiveness
                    RETURN avg_effectiveness
                """)
                control_effectiveness = control_result.single()['avg_effectiveness'] or 0.0

                # Calculate process adherence
                process_result = session.run("""
                    MATCH (a:Activity)
                    WITH avg(a.completion_rate) as avg_adherence
                    RETURN avg_adherence
                """)
                process_adherence = process_result.single()['avg_adherence'] or 0.0

                # Calculate regulatory coverage
                regulatory_result = session.run("""
                    MATCH (g:Guideline {type: 'Regulatory'})-[:APPLIES_TO]->(a:Activity)
                    WITH count(DISTINCT a) as covered_activities
                    MATCH (a:Activity)
                    WITH covered_activities, count(a) as total_activities
                    RETURN CASE WHEN total_activities > 0 
                           THEN (toFloat(covered_activities) / total_activities) * 100 
                           ELSE 0.0 END as coverage
                """)
                regulatory_coverage = regulatory_result.single()['coverage'] or 0.0

                # Update metrics
                self.metrics['compliance']['regulatory_coverage'] = regulatory_coverage
                self.metrics['compliance']['control_effectiveness'] = control_effectiveness
                self.metrics['operational']['process_adherence'] = process_adherence
                self.metrics['risk']['control_coverage'] = control_effectiveness

        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            raise

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

    def _get_process_data(self) -> Dict:
        """Retrieve process data from Neo4j"""
        with self.driver.session() as session:
            # Get activities and their relationships
            result = session.run("""
                MATCH (a:Activity)
                OPTIONAL MATCH (a)-[:MONITORS]-(c:Control)
                RETURN {
                    activities: collect(distinct {
                        id: a.id,
                        name: a.name,
                        completion_rate: a.completion_rate,
                        controls: collect(c.name)
                    })
                } as process_data
            """)
            return result.single()['process_data']

    def _get_guidelines_data(self) -> Dict:
        """Retrieve guidelines data from Neo4j"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (g:Guideline)
                RETURN {
                    guidelines: collect({
                        id: g.id,
                        name: g.name,
                        type: g.type,
                        severity: g.severity,
                        implementation_status: g.implementation_status
                    })
                } as guidelines_data
            """)
            return result.single()['guidelines_data']

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

                # Update metrics
                total_guidelines = session.run("""
                    MATCH (g:Guideline {type: 'Regulatory'})
                    RETURN count(*) as total
                """).single()['total']

                if total_guidelines > 0:
                    compliant_count = total_guidelines - len(findings)
                    self.metrics['compliance']['regulatory_coverage'] = (compliant_count / total_guidelines) * 100

                return findings
        except Exception as e:
            logger.error(f"Error in regulatory analysis: {str(e)}")
            raise

    def analyze_process_execution(self) -> List[GapFindings]:
        """Analyze process execution gaps"""
        try:
            with self.driver.session() as session:
                # Calculate process coverage
                process_metrics = session.run("""
                    MATCH (a:Activity)
                    WITH count(a) as total_activities
                    MATCH (a:Activity)
                    WHERE a.completion_rate >= 95
                    WITH count(a) as compliant_activities, total_activities
                    RETURN toFloat(compliant_activities) / total_activities * 100 as coverage
                """).single()

                self.metrics['operational']['process_adherence'] = process_metrics['coverage']

                # Analyze SLA breaches
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
            raise

    def analyze_control_effectiveness(self) -> List[GapFindings]:
        """Analyze control effectiveness gaps"""
        try:
            with self.driver.session() as session:
                # Calculate overall control effectiveness
                effectiveness_result = session.run("""
                    MATCH (c:Control)
                    WITH collect(c.effectiveness_score) as scores
                    RETURN reduce(s = 0, x IN scores | s + x) / size(scores) as avg_effectiveness
                """).single()

                self.metrics['risk']['control_effectiveness'] = effectiveness_result['avg_effectiveness']

                # Analyze control execution gaps
                result = session.run("""
                    MATCH (c:Control)
                    OPTIONAL MATCH (ce:ControlExecution)-[:EXECUTES]->(c)
                    WITH c, 
                         count(ce) as execution_count,
                         sum(CASE WHEN ce.result = 'Pass' THEN 1 ELSE 0 END) as passed_count,
                         c.effectiveness_score as effectiveness_score
                    WHERE execution_count = 0 OR (passed_count / toFloat(execution_count)) < 0.8
                    RETURN c.name as control_name,
                           c.type as control_type,
                           execution_count,
                           effectiveness_score,
                           CASE WHEN execution_count > 0 
                                THEN toFloat(passed_count) / execution_count * 100 
                                ELSE 0 END as success_rate
                """)

                findings = []
                for record in result:
                    findings.append(GapFindings(
                        category='Control Effectiveness',
                        severity='High',
                        description=f"Ineffective control: {record['control_name']}",
                        current_state=(f"Success rate: {record['success_rate']:.1f}%, "
                                       f"Effectiveness score: {record['effectiveness_score']}"),
                        expected_state="Success rate >= 80% and Effectiveness score >= 85",
                        impact='High - Control Risk',
                        recommendations=[
                            'Review control design',
                            'Implement automated controls',
                            'Enhance monitoring capabilities'
                        ]
                    ))

                return findings

        except Exception as e:
            logger.error(f"Error in control effectiveness analysis: {str(e)}")
            raise

    def get_recommendations(self) -> List[Dict]:
        """Get prioritized recommendations"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (r:Recommendation)
                    RETURN r.description as description,
                           r.priority as priority,
                           r.target_date as target_date,
                           r.impact as impact
                    ORDER BY 
                        CASE r.priority 
                            WHEN 'High' THEN 1 
                            WHEN 'Medium' THEN 2 
                            ELSE 3 
                        END
                """)

                recommendations = []
                for record in result:
                    recommendations.append({
                        'description': record['description'],
                        'priority': record['priority'],
                        'target_date': record['target_date'],
                        'impact': record['impact']
                    })

                return recommendations
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

    def calculate_stp_rate(self) -> float:
        """Calculate Straight Through Processing rate"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH p=(:Event)-[:NEXT*]->(:Event)
                    WITH length(p) as path_length,
                         count(*) as path_count
                    WHERE path_length >= 2
                    RETURN sum(path_count) as total_paths,
                           sum(CASE WHEN path_length = 2 THEN path_count ELSE 0 END) as direct_paths
                """).single()

                if result['total_paths'] > 0:
                    return (result['direct_paths'] / result['total_paths']) * 100
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating STP rate: {str(e)}")
            return 0.0

    def generate_gap_report(self) -> Dict:
        """Generate comprehensive gap analysis report with optional AI insights"""
        try:
            # Calculate base metrics and findings
            self.calculate_metrics()
            regulatory_findings = self.analyze_regulatory_compliance()

            # Initialize report structure
            report = {
                'summary': {
                    'total_gaps': len(regulatory_findings),
                    'high_severity': len([f for f in regulatory_findings if f.severity == 'High']),
                    'medium_severity': len([f for f in regulatory_findings if f.severity == 'Medium']),
                    'low_severity': len([f for f in regulatory_findings if f.severity == 'Low'])
                },
                'metrics': self.metrics,
                'findings': [self._format_finding(f) for f in regulatory_findings],
                'recommendations': self.analyze_recommendations(),
                'timestamp': datetime.now().isoformat()
            }

            # Add AI insights if available
            if self.ai_analyzer:
                try:
                    process_data = self._get_process_data()
                    guidelines = self._get_guidelines_data()
                    ai_analysis = self.ai_analyzer.analyze_process_gaps(process_data, guidelines)

                    report['ai_findings'] = ai_analysis['gaps']
                    report['metrics']['ai_insights'] = ai_analysis['metrics']

                    # Merge AI recommendations with existing ones
                    ai_recommendations = self.ai_analyzer.analyze_recommendations(
                        report['findings']
                    )
                    report['recommendations'].extend(ai_recommendations)

                except Exception as e:
                    logging.error(f"AI analysis failed: {str(e)}")
                    # Continue without AI insights
                    pass

            return report

        except Exception as e:
            logging.error(f"Error generating gap report: {str(e)}")
            raise

    def _format_finding(self, finding: GapFindings) -> Dict:
        """Format finding for report"""
        return {
            'category': finding.category,
            'severity': finding.severity,
            'description': finding.description,
            'gap': {
                'current': finding.current_state,
                'expected': finding.expected_state
            },
            'impact': finding.impact,
            'recommendations': finding.recommendations
        }

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

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


def handle_graph_analytics():
    """Handle graph analytics operations"""
    st.subheader("Graph Analytics")

    if not hasattr(st.session_state, 'neo4j_connected') or not st.session_state.neo4j_connected:
        st.warning("Please connect to Neo4j and import data first")
        return

    analysis_tabs = st.tabs(["Process Flow", "Comprehensive Gap Analysis"])

    with analysis_tabs[1]:
        st.subheader("Comprehensive Gap Analysis")

        credentials = st.session_state.neo4j_credentials

        analysis_type = st.multiselect(
            "Select Analysis Types",
            ["Regulatory Compliance", "Process Execution", "Control Effectiveness"],
            default=["Regulatory Compliance"]
        )

        if st.button("Run Comprehensive Analysis"):
            try:
                with st.spinner("Running comprehensive gap analysis..."):
                    analyzer = FXTradingGapAnalyzer(
                        uri=credentials['uri'],
                        user=credentials['user'],
                        password=credentials['password']
                    )

                    # Generate report
                    report = analyzer.generate_gap_report()

                    # Create visualizer
                    visualizer = GapAnalysisVisualizer(report)
                    dashboard = visualizer.generate_interactive_dashboard()

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Gaps", report['summary']['total_gaps'])
                    with col2:
                        st.metric("High Severity", report['summary']['high_severity'])
                    with col3:
                        st.metric("Process Coverage",
                                  f"{report['metrics']['operational']['process_adherence']:.1f}%")
                    with col4:
                        st.metric("Control Effectiveness",
                                  f"{report['metrics']['risk']['control_coverage']:.1f}%")

                    # Display AI Insights if available
                    if 'ai_findings' in report:
                        st.subheader("🤖 AI-Powered Insights")
                        for finding in report['ai_findings']:
                            with st.expander(f"{finding['category']} - {finding['severity']}"):
                                st.write(f"**Description:** {finding['description']}")
                                st.write(f"**Impact:** {finding['impact']}")
                                if finding.get('related_controls'):
                                    st.write("**Related Controls:**")
                                    for control in finding['related_controls']:
                                        st.write(f"- {control}")

                    # Display visualizations in two columns
                    st.subheader("Analysis Visualizations")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(dashboard['severity_distribution'],
                                        use_container_width=True)
                        st.plotly_chart(dashboard['gap_heatmap'],
                                        use_container_width=True)

                    with col2:
                        st.plotly_chart(dashboard['coverage_radar'],
                                        use_container_width=True)
                        st.plotly_chart(dashboard['timeline_view'],
                                        use_container_width=True)

                    # Display detailed recommendations
                    st.subheader("Detailed Recommendations")
                    visualizer.display_recommendations_table()

                    analyzer.close()

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.error(traceback.format_exc())


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

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'APA Analytics'

    st.sidebar.title("IRMAI")

    current_page = st.sidebar.radio(
        label="",
        options=["APA Analytics", "Digital Twin"],
        label_visibility="collapsed"
    )

    st.session_state.current_page = current_page

    if st.session_state.current_page == "Digital Twin":
        st.title("Digital Twin")
        tab1, tab2 = st.tabs(["Import Data", "Graph Analytics"])

        with tab1:
            handle_data_import()
        with tab2:
            handle_graph_analytics()
    else:
        st.title("APA Analytics")
        create_ocpm_ui()


if __name__ == "__main__":
    main()