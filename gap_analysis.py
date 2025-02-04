import os
import logging
from datetime import datetime
from xml.etree import ElementTree as ET
import PyPDF2
import spacy
from typing import Dict, List, Set, Optional


class BPMNProcessAnalyzer:
    """Analyzes actual process from BPMN"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_process(self, bpmn_graph: ET.Element) -> Dict:
        """Extract detailed process information from BPMN"""
        try:
            ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
            process_details = {
                'activities': {},
                'sequence_flows': [],
                'gateways': [],
                'events': [],
                'metrics': {
                    'total_activities': 0,
                    'total_gateways': 0,
                    'total_events': 0
                }
            }

            # Extract activities and their connections
            for task in bpmn_graph.findall('.//bpmn:task', ns):
                activity_id = task.get('id')
                activity_name = task.get('name', '')

                # Find incoming flows
                incoming_flows = []
                for flow in bpmn_graph.findall('.//bpmn:sequenceFlow[@targetRef="{}"]'.format(activity_id), ns):
                    incoming_flows.append({
                        'id': flow.get('id'),
                        'source': flow.get('sourceRef'),
                        'source_name': self._get_node_name(bpmn_graph, flow.get('sourceRef'), ns)
                    })

                # Find outgoing flows
                outgoing_flows = []
                for flow in bpmn_graph.findall('.//bpmn:sequenceFlow[@sourceRef="{}"]'.format(activity_id), ns):
                    outgoing_flows.append({
                        'id': flow.get('id'),
                        'target': flow.get('targetRef'),
                        'target_name': self._get_node_name(bpmn_graph, flow.get('targetRef'), ns)
                    })

                process_details['activities'][activity_name] = {
                    'id': activity_id,
                    'type': 'task',
                    'incoming': incoming_flows,
                    'outgoing': outgoing_flows
                }

            # Extract gateways
            for gateway in bpmn_graph.findall('.//bpmn:*Gateway', ns):
                gateway_info = {
                    'id': gateway.get('id'),
                    'name': gateway.get('name', ''),
                    'type': gateway.tag.split('}')[-1],
                    'incoming': self._get_gateway_connections(gateway, 'incoming', bpmn_graph, ns),
                    'outgoing': self._get_gateway_connections(gateway, 'outgoing', bpmn_graph, ns)
                }
                process_details['gateways'].append(gateway_info)

            # Extract events (start, end, intermediate)
            for event_type in ['startEvent', 'endEvent', 'intermediateEvent']:
                for event in bpmn_graph.findall(f'.//bpmn:{event_type}', ns):
                    event_info = {
                        'id': event.get('id'),
                        'name': event.get('name', ''),
                        'type': event_type
                    }
                    process_details['events'].append(event_info)

            # Update metrics
            process_details['metrics'].update({
                'total_activities': len(process_details['activities']),
                'total_gateways': len(process_details['gateways']),
                'total_events': len(process_details['events'])
            })

            return process_details

        except Exception as e:
            self.logger.error(f"Error analyzing process: {str(e)}")
            raise

    def _get_node_name(self, bpmn_graph, node_id, ns):
        """Get name of a BPMN node by ID"""
        for node in bpmn_graph.findall('.//*[@id="{}"]'.format(node_id), ns):
            return node.get('name', '')
        return ''

    def _get_gateway_connections(self, gateway, direction, bpmn_graph, ns):
        """Get gateway connections (incoming or outgoing)"""
        connections = []
        gateway_id = gateway.get('id')

        if direction == 'incoming':
            flows = bpmn_graph.findall('.//bpmn:sequenceFlow[@targetRef="{}"]'.format(gateway_id), ns)
            for flow in flows:
                connections.append({
                    'id': flow.get('id'),
                    'source': flow.get('sourceRef'),
                    'source_name': self._get_node_name(bpmn_graph, flow.get('sourceRef'), ns)
                })
        else:
            flows = bpmn_graph.findall('.//bpmn:sequenceFlow[@sourceRef="{}"]'.format(gateway_id), ns)
            for flow in flows:
                connections.append({
                    'id': flow.get('id'),
                    'target': flow.get('targetRef'),
                    'target_name': self._get_node_name(bpmn_graph, flow.get('targetRef'), ns)
                })

        return connections


class GuidelinesAnalyzer:
    """Analyzes FX trading guidelines using NLP"""

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.logger = logging.getLogger(__name__)

        # Keywords for classification
        self.mandatory_keywords = ['must', 'shall', 'required', 'mandatory']
        self.recommended_keywords = ['should', 'recommended', 'advisable']
        self.control_keywords = ['control', 'monitor', 'verify', 'validate', 'check', 'ensure']
        self.risk_keywords = ['risk', 'exposure', 'threat', 'vulnerability']

    def _is_mandatory_requirement(self, sent) -> bool:
        """Check if sentence contains mandatory requirement"""
        text = sent.text.lower()
        return any(keyword in text for keyword in self.mandatory_keywords)

    def _is_recommended_activity(self, sent) -> bool:
        """Check if sentence contains recommended activity"""
        text = sent.text.lower()
        return any(keyword in text for keyword in self.recommended_keywords)

    def _is_control_requirement(self, sent) -> bool:
        """Check if sentence contains control requirement"""
        text = sent.text.lower()
        return any(keyword in text for keyword in self.control_keywords)

    def _is_risk_requirement(self, sent) -> bool:
        """Check if sentence contains risk requirement"""
        text = sent.text.lower()
        return any(keyword in text for keyword in self.risk_keywords)

    def _is_regulatory_requirement(self, sent) -> bool:
        """Check if sentence contains regulatory requirement"""
        text = sent.text.lower()
        return 'regulatory' in text or 'regulation' in text or 'compliance' in text

    def analyze_guidelines(self, guidelines_text: str) -> Dict:
        """Extract structured requirements from guidelines"""
        try:
            doc = self.nlp(guidelines_text)
            requirements = {
                'mandatory_activities': [],
                'recommended_activities': [],
                'controls': [],
                'risks': [],
                'regulatory_requirements': []
            }

            for sent in doc.sents:
                text = sent.text.strip()
                if len(text) < 10:  # Skip very short sentences
                    continue

                if self._is_mandatory_requirement(sent):
                    requirement = self._extract_requirement(sent)
                    if requirement:
                        requirements['mandatory_activities'].append(requirement)

                elif self._is_recommended_activity(sent):
                    activity = self._extract_activity(sent)
                    if activity:
                        requirements['recommended_activities'].append(activity)

                elif self._is_control_requirement(sent):
                    control = self._extract_control(sent)
                    if control:
                        requirements['controls'].append(control)

                elif self._is_risk_requirement(sent):
                    # Fixed: Changed _extract_risk to _extract_risk_requirement
                    risk = self._extract_risk_requirement(sent)
                    if risk:
                        requirements['risks'].append(risk)

                if self._is_regulatory_requirement(sent):
                    reg = self._extract_regulatory_requirement(sent)
                    if reg:
                        requirements['regulatory_requirements'].append(reg)

            self.logger.info(f"Extracted {len(requirements['mandatory_activities'])} mandatory activities")
            self.logger.info(f"Extracted {len(requirements['recommended_activities'])} recommended activities")
            self.logger.info(f"Extracted {len(requirements['controls'])} controls")
            self.logger.info(f"Extracted {len(requirements['risks'])} risks")

            return requirements

        except Exception as e:
            self.logger.error(f"Error analyzing guidelines: {str(e)}")
            raise

    def _extract_requirement(self, sent) -> Optional[Dict]:
        """Extract structured requirement from sentence"""
        verb = None
        objects = []
        context = sent.text

        for token in sent:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                verb = token.lemma_
            elif token.dep_ in ['dobj', 'pobj'] and not token.is_stop:
                objects.append(token.text)

        if verb and objects:
            return {
                'action': verb,
                'objects': objects,
                'context': context,
                'text': sent.text
            }
        return None

    def _extract_activity(self, sent) -> Optional[Dict]:
        """Extract activity from sentence"""
        return self._extract_requirement(sent)  # Same structure for now

    def _extract_control(self, sent) -> Optional[Dict]:
        """Extract control from sentence"""
        control = self._extract_requirement(sent)
        if control:
            control['type'] = 'control'
        return control

    def _extract_risk_requirement(self, sent) -> Optional[Dict]:
        """Extract risk requirement from sentence"""
        risk = self._extract_requirement(sent)
        if risk:
            risk['type'] = 'risk'
            # Determine severity
            text = sent.text.lower()
            if any(word in text for word in ['high', 'critical', 'severe']):
                risk['severity'] = 'high'
            elif any(word in text for word in ['medium', 'moderate']):
                risk['severity'] = 'medium'
            else:
                risk['severity'] = 'low'
        return risk

    def _extract_regulatory_requirement(self, sent) -> Optional[Dict]:
        """Extract regulatory requirement from sentence"""
        reg = self._extract_requirement(sent)
        if reg:
            reg['type'] = 'regulatory'
        return reg


def generate_gap_report(process_details: Dict, requirements: Dict) -> str:
    """Generate comprehensive gap analysis report"""
    report = []

    # Process Overview
    report.append("=== FX Trading Process Gap Analysis Report ===")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Current Process Details
    report.append("1. Current Process Overview")
    report.append(f"   Total Activities: {process_details['metrics']['total_activities']}")
    report.append(f"   Total Gateways: {process_details['metrics']['total_gateways']}")
    report.append(f"   Total Events: {process_details['metrics']['total_events']}\n")

    # Activity Details
    report.append("2. Activity Analysis")
    for name, activity in process_details['activities'].items():
        report.append(f"   Activity: {name}")
        report.append(f"     ID: {activity['id']}")
        report.append(f"     Type: {activity['type']}")
        report.append(f"     Incoming Connections: {len(activity['incoming'])}")
        report.append("     - Incoming From:")
        for flow in activity['incoming']:
            report.append(f"       * {flow['source_name']}")
        report.append(f"     Outgoing Connections: {len(activity['outgoing'])}")
        report.append("     - Outgoing To:")
        for flow in activity['outgoing']:
            report.append(f"       * {flow['target_name']}")
        report.append("")

    # Guideline Requirements
    report.append("3. Guideline Requirements")
    report.append("   Mandatory Activities:")
    for req in requirements['mandatory_activities']:
        report.append(f"   - Action: {req['action']}")
        report.append(f"     Objects: {', '.join(req['objects'])}")
        report.append(f"     Context: {req['text']}\n")

    # Gap Analysis
    report.append("4. Gap Analysis")

    # Activity Gaps
    actual_activities = set(process_details['activities'].keys())
    required_activities = set(req['action'] for req in requirements['mandatory_activities'])

    missing_activities = required_activities - actual_activities
    if missing_activities:
        report.append("   Missing Required Activities:")
        for activity in missing_activities:
            report.append(f"   - {activity}")
            matching_reqs = [req for req in requirements['mandatory_activities']
                             if req['action'] == activity]
            for req in matching_reqs:
                report.append(f"     Context: {req['text']}")
        report.append("")

    # Control Gaps
    report.append("   Control Gaps:")
    for control in requirements['controls']:
        implemented = False
        # Check if control is implemented in process
        if not implemented:
            report.append(f"   - Missing Control: {control['action']}")
            report.append(f"     Context: {control['text']}")

    return "\n".join(report)


def main():
    try:
        # Initialize analyzers
        process_analyzer = BPMNProcessAnalyzer()
        guidelines_analyzer = GuidelinesAnalyzer()

        # Load BPMN
        bpmn_path = "staging/fx_trade_process.bpmn"
        guidelines_path = "staging/tradingguidelinesNov2010.pdf"

        # Validate file existence
        if not os.path.exists(bpmn_path):
            raise FileNotFoundError(f"BPMN file not found: {bpmn_path}")
        if not os.path.exists(guidelines_path):
            raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")

        # Parse BPMN
        bpmn_tree = ET.parse(bpmn_path)
        bpmn_graph = bpmn_tree.getroot()

        # Analyze process (removed event_log_path parameter)
        process_details = process_analyzer.analyze_process(bpmn_graph)

        # Analyze guidelines
        with open(guidelines_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            guidelines_text = ''
            for page in pdf_reader.pages:
                guidelines_text += page.extract_text()

        requirements = guidelines_analyzer.analyze_guidelines(guidelines_text)

        # Generate report
        report = generate_gap_report(process_details, requirements)

        # Save report
        report_path = "fx_trade_gap_analysis.txt"
        with open(report_path, "w") as f:
            f.write(report)

        print("Gap analysis completed. Report saved to fx_trade_gap_analysis.txt")

    except Exception as e:
        logging.error(f"Error in gap analysis: {str(e)}", exc_info=True)
        print(f"Error occurred: {str(e)}")
        print("Check gap_analysis.log for details")


if __name__ == "__main__":
    main()