"""
Archiver for the Autonomous Quantitative Research Agency

This module handles self-documentation, interpretability reports, and maintains
a living archive of the agency's intellectual lineage. It tracks strategy
evolution, generates comprehensive reports, and builds a knowledge base of
successful quantitative approaches.
"""

import os
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import markdown
import pdfkit
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from .config import config
from .evaluator import ValidationResult, EvaluationMetrics
from .evolution_engine import StrategyGenome, EvolutionResult


@dataclass
class StrategyLineage:
    """Complete lineage record of a strategy's evolution"""
    strategy_id: str
    original_hypothesis: str
    creation_date: datetime
    current_generation: int
    fitness_history: List[float] = field(default_factory=list)
    evolution_events: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    mathematical_properties: Dict[str, Any] = field(default_factory=dict)
    computational_characteristics: Dict[str, Any] = field(default_factory=dict)
    risk_profile: Dict[str, Any] = field(default_factory=dict)
    market_regime_performance: Dict[str, Any] = field(default_factory=dict)
    interpretability_score: float = 0.0
    elegance_score: float = 0.0
    robustness_score: float = 0.0


@dataclass
class KnowledgeBaseEntry:
    """Entry in the agency's knowledge base"""
    concept_id: str
    concept_type: str  # 'indicator', 'signal_logic', 'risk_management', 'parameter_set'
    description: str
    mathematical_formulation: str
    empirical_performance: Dict[str, Any]
    usage_frequency: int
    success_rate: float
    discovery_date: datetime
    related_concepts: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


@dataclass
class ArchiveReport:
    """Comprehensive archive report"""
    report_id: str
    generation: int
    timestamp: datetime
    summary: Dict[str, Any]
    top_strategies: List[Dict[str, Any]]
    evolution_insights: Dict[str, Any]
    knowledge_discoveries: List[Dict[str, Any]]
    risk_assessments: Dict[str, Any]
    recommendations: List[str]


class ReportGenerator(ABC):
    """Abstract base class for report generators"""

    @abstractmethod
    def generate(self, data: Dict[str, Any], output_path: Path) -> bool:
        """Generate a report from data"""
        pass


class MarkdownReportGenerator(ReportGenerator):
    """Generate markdown reports"""

    def generate(self, data: Dict[str, Any], output_path: Path) -> bool:
        """Generate markdown report"""

        template = """
# Autonomous Quantitative Research Agency - Archive Report

**Report ID:** {{ report_id }}
**Generation:** {{ generation }}
**Timestamp:** {{ timestamp }}

## Executive Summary

- **Total Strategies:** {{ summary.total_strategies }}
- **Active Strategies:** {{ summary.active_strategies }}
- **Elite Strategies:** {{ summary.elite_strategies }}
- **Average Fitness:** {{ "%.4f"|format(summary.avg_fitness) }}
- **Best Fitness:** {{ "%.4f"|format(summary.best_fitness) }}

## Top Performing Strategies

{% for strategy in top_strategies %}
### {{ strategy.name }}

- **Fitness Score:** {{ "%.4f"|format(strategy.fitness) }}
- **Generation:** {{ strategy.generation }}
- **Mathematical Beauty:** {{ "%.3f"|format(strategy.mathematical_beauty) }}
- **Robustness:** {{ "%.3f"|format(strategy.robustness) }}

**Key Parameters:**
```json
{{ strategy.parameters | tojson(indent=2) }}
```

**Performance Metrics:**
- Sharpe Ratio: {{ "%.3f"|format(strategy.performance.sharpe_ratio) }}
- Max Drawdown: {{ "%.2%"|format(strategy.performance.max_drawdown) }}
- Win Rate: {{ "%.1%"|format(strategy.performance.win_rate) }}
- Profit Factor: {{ "%.3f"|format(strategy.performance.profit_factor) }}

---
{% endfor %}

## Evolution Insights

### Population Dynamics
- **Diversity Score:** {{ "%.3f"|format(evolution_insights.diversity_score) }}
- **Convergence Score:** {{ "%.3f"|format(evolution_insights.convergence_score) }}
- **Innovation Rate:** {{ "%.1%"|format(evolution_insights.innovation_rate) }}

### Fitness Distribution
```
Min: {{ "%.4f"|format(evolution_insights.fitness_min) }}
Mean: {{ "%.4f"|format(evolution_insights.fitness_mean) }}
Max: {{ "%.4f"|format(evolution_insights.fitness_max) }}
Std: {{ "%.4f"|format(evolution_insights.fitness_std) }}
```

## Knowledge Discoveries

{% for discovery in knowledge_discoveries %}
### {{ discovery.concept_type.title() }}: {{ discovery.concept_id }}

{{ discovery.description }}

**Mathematical Formulation:**
```
{{ discovery.mathematical_formulation }}
```

**Performance:**
- Success Rate: {{ "%.1%"|format(discovery.success_rate) }}
- Usage Frequency: {{ discovery.usage_frequency }}

---
{% endfor %}

## Risk Assessment

### Market Regime Performance
{% for regime, performance in risk_assessments.market_regimes.items() %}
- **{{ regime.title() }}:** {{ "%.3f"|format(performance.fitness) }} ({{ performance.confidence }} confidence)
{% endfor %}

### Systemic Risks
- **Overfitting Risk:** {{ risk_assessments.systemic.overfitting_risk }}
- **Market Impact Risk:** {{ risk_assessments.systemic.market_impact_risk }}
- **Model Risk:** {{ risk_assessments.systemic.model_risk }}

## Recommendations

{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}

---
*Report generated by Autonomous Quantitative Research Agency*
"""

        try:
            jinja_template = Template(template)
            report_content = jinja_template.render(**data)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            return True

        except Exception as e:
            logging.error(f"Failed to generate markdown report: {e}")
            return False


class PDFReportGenerator(ReportGenerator):
    """Generate PDF reports"""

    def generate(self, data: Dict[str, Any], output_path: Path) -> bool:
        """Generate PDF report"""

        try:
            doc = SimpleDocTemplate(str(output_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
            )
            story.append(Paragraph("Autonomous Quantitative Research Agency - Archive Report", title_style))
            story.append(Spacer(1, 12))

            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            summary_text = f"""
            Total Strategies: {data['summary']['total_strategies']}<br/>
            Active Strategies: {data['summary']['active_strategies']}<br/>
            Elite Strategies: {data['summary']['elite_strategies']}<br/>
            Average Fitness: {data['summary']['avg_fitness']:.4f}<br/>
            Best Fitness: {data['summary']['best_fitness']:.4f}
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 12))

            # Top Strategies
            story.append(Paragraph("Top Performing Strategies", styles['Heading2']))

            for strategy in data['top_strategies'][:3]:  # Top 3 for PDF
                story.append(Paragraph(f"<b>{strategy['name']}</b>", styles['Heading3']))

                strategy_text = f"""
                Fitness Score: {strategy['fitness']:.4f}<br/>
                Generation: {strategy['generation']}<br/>
                Mathematical Beauty: {strategy['mathematical_beauty']:.3f}<br/>
                Robustness: {strategy['robustness']:.3f}
                """
                story.append(Paragraph(strategy_text, styles['Normal']))
                story.append(Spacer(1, 6))

            # Build PDF
            doc.build(story)
            return True

        except Exception as e:
            logging.error(f"Failed to generate PDF report: {e}")
            return False


class StrategyArchiver:
    """Main archiver for strategy documentation and lineage tracking"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.archive_dir = Path(config.archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.lineage_dir = self.archive_dir / "lineages"
        self.reports_dir = self.archive_dir / "reports"
        self.knowledge_dir = self.archive_dir / "knowledge_base"
        self.visualizations_dir = self.archive_dir / "visualizations"

        for dir_path in [self.lineage_dir, self.reports_dir, self.knowledge_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Report generators
        self.report_generators = {
            'markdown': MarkdownReportGenerator(),
            'pdf': PDFReportGenerator()
        }

        # Knowledge base
        self.knowledge_base: Dict[str, KnowledgeBaseEntry] = {}
        self._load_knowledge_base()

        # Strategy lineages
        self.strategy_lineages: Dict[str, StrategyLineage] = {}
        self._load_lineages()

    def archive_strategy_genome(self, genome: StrategyGenome, validation_result: ValidationResult):
        """
        Archive a strategy genome and its validation results

        Args:
            genome: Strategy genome to archive
            validation_result: Validation results for the strategy
        """
        try:
            # Create or update lineage
            if genome.strategy_name not in self.strategy_lineages:
                lineage = StrategyLineage(
                    strategy_id=genome.strategy_name,
                    original_hypothesis=genome.hypothesis_id,
                    creation_date=datetime.now(),
                    current_generation=genome.generation
                )
                self.strategy_lineages[genome.strategy_name] = lineage
            else:
                lineage = self.strategy_lineages[genome.strategy_name]
                lineage.current_generation = genome.generation

            # Update lineage data
            lineage.fitness_history.append(genome.fitness_score)
            lineage.evolution_events.extend(genome.mutation_history)
            lineage.evolution_events.extend(genome.crossover_history)

            # Update performance metrics
            lineage.performance_metrics = self._extract_performance_metrics(validation_result)

            # Update mathematical properties
            lineage.mathematical_properties = {
                'complexity_score': genome.complexity_score,
                'mathematical_beauty': genome.mathematical_beauty,
                'computational_efficiency': genome.computational_efficiency
            }

            # Calculate interpretability and elegance scores
            lineage.interpretability_score = self._calculate_interpretability_score(genome, validation_result)
            lineage.elegance_score = genome.mathematical_beauty
            lineage.robustness_score = validation_result.evaluation_metrics.robustness_score

            # Save lineage
            self._save_lineage(lineage)

            # Extract and archive knowledge
            self._extract_knowledge_from_genome(genome, validation_result)

            self.logger.info(f"Archived strategy genome: {genome.strategy_name}")

        except Exception as e:
            self.logger.error(f"Failed to archive strategy genome {genome.strategy_name}: {e}")

    def archive_evolution_result(self, evolution_result: EvolutionResult):
        """
        Archive evolution cycle results

        Args:
            evolution_result: Results from evolution cycle
        """
        try:
            # Generate comprehensive report
            report_data = self._prepare_report_data(evolution_result)

            # Generate reports in multiple formats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"evolution_report_gen_{evolution_result.generation}_{timestamp}"

            for format_name, generator in self.report_generators.items():
                output_path = self.reports_dir / f"{base_filename}.{format_name}"
                success = generator.generate(report_data, output_path)
                if success:
                    self.logger.info(f"Generated {format_name} report: {output_path}")
                else:
                    self.logger.error(f"Failed to generate {format_name} report")

            # Generate visualizations
            self._generate_evolution_visualizations(evolution_result)

            # Save raw evolution data
            evolution_file = self.reports_dir / f"evolution_data_gen_{evolution_result.generation}.json"
            with open(evolution_file, 'w') as f:
                json.dump(asdict(evolution_result), f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to archive evolution result: {e}")

    def generate_interpretability_report(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Generate detailed interpretability report for a strategy

        Args:
            strategy_name: Name of the strategy to analyze

        Returns:
            Interpretability report data
        """
        try:
            if strategy_name not in self.strategy_lineages:
                return None

            lineage = self.strategy_lineages[strategy_name]

            report = {
                'strategy_id': strategy_name,
                'interpretability_score': lineage.interpretability_score,
                'elegance_score': lineage.elegance_score,
                'robustness_score': lineage.robustness_score,
                'mathematical_properties': lineage.mathematical_properties,
                'performance_metrics': lineage.performance_metrics,
                'evolution_history': lineage.evolution_events[-10:],  # Last 10 events
                'fitness_trajectory': lineage.fitness_history,
                'market_regime_performance': lineage.market_regime_performance,
                'risk_profile': lineage.risk_profile,
                'recommendations': self._generate_strategy_recommendations(lineage)
            }

            return report

        except Exception as e:
            self.logger.error(f"Failed to generate interpretability report for {strategy_name}: {e}")
            return None

    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Get summary of the knowledge base"""

        concept_types = {}
        total_concepts = len(self.knowledge_base)

        for concept in self.knowledge_base.values():
            concept_type = concept.concept_type
            if concept_type not in concept_types:
                concept_types[concept_type] = {
                    'count': 0,
                    'avg_success_rate': 0,
                    'total_usage': 0
                }

            concept_types[concept_type]['count'] += 1
            concept_types[concept_type]['avg_success_rate'] += concept.success_rate
            concept_types[concept_type]['total_usage'] += concept.usage_frequency

        # Calculate averages
        for concept_type in concept_types:
            count = concept_types[concept_type]['count']
            concept_types[concept_type]['avg_success_rate'] /= count

        return {
            'total_concepts': total_concepts,
            'concept_types': concept_types,
            'most_successful_concepts': self._get_top_concepts_by_success(5),
            'most_used_concepts': self._get_top_concepts_by_usage(5)
        }

    def _prepare_report_data(self, evolution_result: EvolutionResult) -> Dict[str, Any]:
        """Prepare data for report generation"""

        # Get top strategies
        top_strategies = []
        for genome in evolution_result.elite_strategies[:5]:
            lineage = self.strategy_lineages.get(genome.strategy_name)
            strategy_data = {
                'name': genome.strategy_name,
                'fitness': genome.fitness_score,
                'generation': genome.generation,
                'mathematical_beauty': genome.mathematical_beauty,
                'robustness': lineage.robustness_score if lineage else 0.0,
                'parameters': genome.parameters,
                'performance': lineage.performance_metrics if lineage else {}
            }
            top_strategies.append(strategy_data)

        # Evolution insights
        evolution_insights = evolution_result.evolution_metrics.copy()
        evolution_insights.update({
            'innovation_rate': len(evolution_result.new_hypotheses) / max(1, evolution_result.population_size),
            'pruning_rate': len(evolution_result.pruned_strategies) / max(1, evolution_result.population_size)
        })

        # Knowledge discoveries (simplified)
        knowledge_discoveries = []
        for concept in list(self.knowledge_base.values())[-5:]:  # Last 5 discoveries
            knowledge_discoveries.append({
                'concept_id': concept.concept_id,
                'concept_type': concept.concept_type,
                'description': concept.description,
                'mathematical_formulation': concept.mathematical_formulation,
                'success_rate': concept.success_rate,
                'usage_frequency': concept.usage_frequency
            })

        # Risk assessments (placeholder)
        risk_assessments = {
            'market_regimes': {
                'bull': {'fitness': 0.85, 'confidence': 'high'},
                'bear': {'fitness': 0.72, 'confidence': 'medium'},
                'sideways': {'fitness': 0.91, 'confidence': 'high'}
            },
            'systemic': {
                'overfitting_risk': 'low',
                'market_impact_risk': 'medium',
                'model_risk': 'low'
            }
        }

        # Recommendations
        recommendations = [
            "Continue evolving top-performing strategies",
            "Increase exploration rate for innovation",
            "Focus on robustness testing across market regimes",
            "Archive successful parameter combinations",
            "Monitor for emerging market patterns"
        ]

        return {
            'report_id': f"evolution_{evolution_result.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generation': evolution_result.generation,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_strategies': evolution_result.population_size,
                'active_strategies': evolution_result.population_size - len(evolution_result.pruned_strategies),
                'elite_strategies': len(evolution_result.elite_strategies),
                'avg_fitness': evolution_result.evolution_metrics.get('mean_fitness', 0),
                'best_fitness': evolution_result.evolution_metrics.get('max_fitness', 0)
            },
            'top_strategies': top_strategies,
            'evolution_insights': evolution_insights,
            'knowledge_discoveries': knowledge_discoveries,
            'risk_assessments': risk_assessments,
            'recommendations': recommendations
        }

    def _generate_evolution_visualizations(self, evolution_result: EvolutionResult):
        """Generate visualizations for evolution results"""

        try:
            # Fitness distribution plot
            fitness_scores = [genome.fitness_score for genome in evolution_result.elite_strategies]

            plt.figure(figsize=(10, 6))
            plt.hist(fitness_scores, bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'Fitness Distribution - Generation {evolution_result.generation}')
            plt.xlabel('Fitness Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

            viz_file = self.visualizations_dir / f"fitness_dist_gen_{evolution_result.generation}.png"
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()

            # Evolution metrics over time (if we have history)
            if len(self.strategy_lineages) > 0:
                self._generate_lineage_visualization()

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")

    def _generate_lineage_visualization(self):
        """Generate visualization of strategy lineages"""

        try:
            # Collect lineage data
            lineage_data = []
            for lineage in self.strategy_lineages.values():
                if len(lineage.fitness_history) > 1:
                    lineage_data.append({
                        'strategy': lineage.strategy_id,
                        'generations': list(range(len(lineage.fitness_history))),
                        'fitness': lineage.fitness_history
                    })

            if lineage_data:
                plt.figure(figsize=(12, 8))

                for data in lineage_data[:10]:  # Top 10 lineages
                    plt.plot(data['generations'], data['fitness'],
                           marker='o', markersize=3, linewidth=1.5,
                           label=data['strategy'][:20] + '...' if len(data['strategy']) > 20 else data['strategy'])

                plt.title('Strategy Lineage Evolution')
                plt.xlabel('Generation')
                plt.ylabel('Fitness Score')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                viz_file = self.visualizations_dir / "strategy_lineages.png"
                plt.savefig(viz_file, dpi=150, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.error(f"Failed to generate lineage visualization: {e}")

    def _extract_performance_metrics(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Extract performance metrics from validation result"""

        metrics = validation_result.evaluation_metrics

        return {
            'sharpe_ratio': getattr(metrics, 'sharpe_ratio', 0),
            'max_drawdown': getattr(metrics, 'max_drawdown', 0),
            'win_rate': getattr(metrics, 'win_rate', 0),
            'profit_factor': getattr(metrics, 'profit_factor', 0),
            'total_return': getattr(metrics, 'total_return', 0),
            'volatility': getattr(metrics, 'volatility', 0),
            'consistency_score': metrics.consistency_score,
            'robustness_score': metrics.robustness_score,
            'overall_score': metrics.overall_score
        }

    def _calculate_interpretability_score(self, genome: StrategyGenome, validation_result: ValidationResult) -> float:
        """Calculate interpretability score for a strategy"""

        # Based on complexity, parameter count, and performance consistency
        complexity_penalty = genome.complexity_score
        consistency_bonus = validation_result.evaluation_metrics.consistency_score
        robustness_bonus = validation_result.evaluation_metrics.robustness_score

        # Lower complexity and higher consistency/robustness = better interpretability
        interpretability = (consistency_bonus + robustness_bonus) / 2 - complexity_penalty * 0.3
        interpretability = max(0, min(1, interpretability))  # Clamp to [0, 1]

        return interpretability

    def _extract_knowledge_from_genome(self, genome: StrategyGenome, validation_result: ValidationResult):
        """Extract knowledge concepts from successful strategies"""

        try:
            # Extract parameter knowledge
            if validation_result.evaluation_metrics.overall_score > 0.7:  # Only from good strategies
                for param_name, param_value in genome.parameters.items():
                    concept_id = f"param_{param_name}_{hash(str(param_value)) % 10000}"

                    if concept_id not in self.knowledge_base:
                        concept = KnowledgeBaseEntry(
                            concept_id=concept_id,
                            concept_type='parameter_set',
                            description=f"Parameter setting: {param_name} = {param_value}",
                            mathematical_formulation=f"{param_name} = {param_value}",
                            empirical_performance={
                                'fitness_contribution': validation_result.evaluation_metrics.overall_score,
                                'usage_context': genome.strategy_name
                            },
                            usage_frequency=1,
                            success_rate=validation_result.evaluation_metrics.overall_score,
                            discovery_date=datetime.now()
                        )
                        self.knowledge_base[concept_id] = concept
                    else:
                        # Update existing concept
                        concept = self.knowledge_base[concept_id]
                        concept.usage_frequency += 1
                        concept.success_rate = (concept.success_rate + validation_result.evaluation_metrics.overall_score) / 2

                # Save updated knowledge base
                self._save_knowledge_base()

        except Exception as e:
            self.logger.error(f"Failed to extract knowledge from genome: {e}")

    def _generate_strategy_recommendations(self, lineage: StrategyLineage) -> List[str]:
        """Generate recommendations for strategy improvement"""

        recommendations = []

        # Based on fitness trajectory
        if len(lineage.fitness_history) > 5:
            recent_trend = np.polyfit(range(len(lineage.fitness_history[-5:])), lineage.fitness_history[-5:], 1)[0]
            if recent_trend < -0.01:
                recommendations.append("Strategy fitness declining - consider parameter adjustment or structural changes")
            elif recent_trend > 0.01:
                recommendations.append("Strategy showing improvement - continue current evolution path")

        # Based on robustness
        if lineage.robustness_score < 0.6:
            recommendations.append("Low robustness score - increase out-of-sample testing and regime analysis")

        # Based on complexity
        if lineage.mathematical_properties.get('complexity_score', 0) > 0.7:
            recommendations.append("High complexity - consider simplification for better interpretability")

        # Based on elegance
        if lineage.elegance_score < 0.5:
            recommendations.append("Low mathematical elegance - explore more elegant formulations")

        if not recommendations:
            recommendations.append("Strategy performing well - maintain current approach")

        return recommendations

    def _get_top_concepts_by_success(self, count: int) -> List[Dict[str, Any]]:
        """Get top concepts by success rate"""

        sorted_concepts = sorted(self.knowledge_base.values(),
                               key=lambda x: x.success_rate,
                               reverse=True)
        return [asdict(concept) for concept in sorted_concepts[:count]]

    def _get_top_concepts_by_usage(self, count: int) -> List[Dict[str, Any]]:
        """Get top concepts by usage frequency"""

        sorted_concepts = sorted(self.knowledge_base.values(),
                               key=lambda x: x.usage_frequency,
                               reverse=True)
        return [asdict(concept) for concept in sorted_concepts[:count]]

    def _save_lineage(self, lineage: StrategyLineage):
        """Save strategy lineage to disk"""

        try:
            lineage_file = self.lineage_dir / f"{lineage.strategy_id}.json"
            with open(lineage_file, 'w') as f:
                json.dump(asdict(lineage), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save lineage {lineage.strategy_id}: {e}")

    def _load_lineages(self):
        """Load existing lineages from disk"""

        try:
            for lineage_file in self.lineage_dir.glob("*.json"):
                with open(lineage_file, 'r') as f:
                    lineage_data = json.load(f)

                # Convert back to datetime
                lineage_data['creation_date'] = datetime.fromisoformat(lineage_data['creation_date'])

                lineage = StrategyLineage(**lineage_data)
                self.strategy_lineages[lineage.strategy_id] = lineage

        except Exception as e:
            self.logger.error(f"Failed to load lineages: {e}")

    def _save_knowledge_base(self):
        """Save knowledge base to disk"""

        try:
            kb_file = self.knowledge_dir / "knowledge_base.json"
            kb_data = {concept_id: asdict(concept) for concept_id, concept in self.knowledge_base.items()}

            with open(kb_file, 'w') as f:
                json.dump(kb_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save knowledge base: {e}")

    def _load_knowledge_base(self):
        """Load knowledge base from disk"""

        try:
            kb_file = self.knowledge_dir / "knowledge_base.json"
            if kb_file.exists():
                with open(kb_file, 'r') as f:
                    kb_data = json.load(f)

                for concept_id, concept_data in kb_data.items():
                    # Convert discovery_date back to datetime
                    concept_data['discovery_date'] = datetime.fromisoformat(concept_data['discovery_date'])
                    concept = KnowledgeBaseEntry(**concept_data)
                    self.knowledge_base[concept_id] = concept

        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")