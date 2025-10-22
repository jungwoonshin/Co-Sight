#!/usr/bin/env python3
"""
GAIA Results Processor and Analysis Tools

This module provides tools for processing, analyzing, and visualizing GAIA benchmark results
from CoSight evaluations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import argparse
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.common.logger_util import logger


@dataclass
class AnalysisConfig:
    """Configuration for results analysis"""
    results_file: str
    output_dir: str
    generate_plots: bool = True
    generate_report: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300


class GAIAResultsProcessor:
    """Processor for GAIA benchmark results with comprehensive analysis"""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the results processor"""
        self.config = config
        self.results_file = Path(config.results_file)
        self.output_dir = Path(config.output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results_data = self._load_results()
        self.df = self._create_dataframe()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded results from {self.results_file}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from results"""
        if 'results' not in self.results_data:
            raise ValueError("Results data does not contain 'results' key")
        
        df = pd.DataFrame(self.results_data['results'])
        
        # Convert numeric columns
        numeric_columns = ['execution_time', 'confidence_score', 'retry_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        boolean_columns = ['is_correct']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Add derived columns
        df['has_error'] = df['error_message'].notna()
        df['confidence_level'] = pd.cut(df['confidence_score'], 
                                      bins=[0, 0.3, 0.7, 1.0], 
                                      labels=['Low', 'Medium', 'High'])
        
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        stats = {}
        
        # Basic metrics
        stats['total_cases'] = len(self.df)
        stats['correct_cases'] = self.df['is_correct'].sum()
        stats['error_cases'] = self.df['has_error'].sum()
        stats['accuracy'] = stats['correct_cases'] / stats['total_cases'] if stats['total_cases'] > 0 else 0
        stats['success_rate'] = (stats['total_cases'] - stats['error_cases']) / stats['total_cases'] if stats['total_cases'] > 0 else 0
        
        # Execution time statistics
        execution_times = self.df['execution_time'].dropna()
        if len(execution_times) > 0:
            stats['execution_time'] = {
                'mean': execution_times.mean(),
                'median': execution_times.median(),
                'std': execution_times.std(),
                'min': execution_times.min(),
                'max': execution_times.max(),
                'q25': execution_times.quantile(0.25),
                'q75': execution_times.quantile(0.75)
            }
        
        # Confidence statistics
        confidence_scores = self.df['confidence_score'].dropna()
        if len(confidence_scores) > 0:
            stats['confidence'] = {
                'mean': confidence_scores.mean(),
                'median': confidence_scores.median(),
                'std': confidence_scores.std(),
                'min': confidence_scores.min(),
                'max': confidence_scores.max()
            }
        
        # Accuracy by confidence level
        confidence_accuracy = self.df.groupby('confidence_level')['is_correct'].agg(['count', 'sum', 'mean'])
        stats['accuracy_by_confidence'] = confidence_accuracy.to_dict()
        
        # Retry statistics
        retry_stats = self.df['retry_count'].value_counts().to_dict()
        stats['retry_distribution'] = retry_stats
        
        return stats
    
    def generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate detailed performance analysis"""
        analysis = {}
        
        # Execution time analysis
        execution_times = self.df['execution_time'].dropna()
        if len(execution_times) > 0:
            analysis['execution_time_performance'] = {
                'fast_cases': len(execution_times[execution_times <= execution_times.quantile(0.25)]),
                'slow_cases': len(execution_times[execution_times >= execution_times.quantile(0.75)]),
                'outliers': len(execution_times[execution_times > execution_times.mean() + 2 * execution_times.std()])
            }
        
        # Accuracy vs execution time correlation
        if 'execution_time' in self.df.columns and 'is_correct' in self.df.columns:
            correlation = self.df[['execution_time', 'is_correct']].corr().iloc[0, 1]
            analysis['accuracy_time_correlation'] = correlation
        
        # Confidence vs accuracy correlation
        if 'confidence_score' in self.df.columns and 'is_correct' in self.df.columns:
            correlation = self.df[['confidence_score', 'is_correct']].corr().iloc[0, 1]
            analysis['confidence_accuracy_correlation'] = correlation
        
        # Error analysis
        error_cases = self.df[self.df['has_error']]
        if len(error_cases) > 0:
            analysis['error_analysis'] = {
                'error_rate': len(error_cases) / len(self.df),
                'avg_execution_time_on_error': error_cases['execution_time'].mean(),
                'avg_execution_time_on_success': self.df[~self.df['has_error']]['execution_time'].mean()
            }
        
        return analysis
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        if not self.config.generate_plots:
            return
        
        logger.info("Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Accuracy Distribution
        self._plot_accuracy_distribution()
        
        # 2. Execution Time Distribution
        self._plot_execution_time_distribution()
        
        # 3. Confidence Score Distribution
        self._plot_confidence_distribution()
        
        # 4. Accuracy vs Confidence
        self._plot_accuracy_vs_confidence()
        
        # 5. Execution Time vs Accuracy
        self._plot_execution_time_vs_accuracy()
        
        # 6. Retry Distribution
        self._plot_retry_distribution()
        
        # 7. Error Analysis
        self._plot_error_analysis()
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def _plot_accuracy_distribution(self):
        """Plot accuracy distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall accuracy
        accuracy_counts = self.df['is_correct'].value_counts()
        ax1.pie(accuracy_counts.values, labels=['Incorrect', 'Correct'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Accuracy Distribution')
        
        # Accuracy by confidence level
        if 'confidence_level' in self.df.columns:
            accuracy_by_confidence = self.df.groupby('confidence_level')['is_correct'].mean()
            accuracy_by_confidence.plot(kind='bar', ax=ax2, color=['red', 'orange', 'green'])
            ax2.set_title('Accuracy by Confidence Level')
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Confidence Level')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'accuracy_distribution.{self.config.plot_format}', 
                   dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_time_distribution(self):
        """Plot execution time distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        execution_times = self.df['execution_time'].dropna()
        
        # Histogram
        ax1.hist(execution_times, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title('Execution Time Distribution')
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_ylabel('Frequency')
        
        # Box plot by correctness
        if 'is_correct' in self.df.columns:
            self.df.boxplot(column='execution_time', by='is_correct', ax=ax2)
            ax2.set_title('Execution Time by Correctness')
            ax2.set_xlabel('Correct')
            ax2.set_ylabel('Execution Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'execution_time_distribution.{self.config.plot_format}', 
                   dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self):
        """Plot confidence score distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        confidence_scores = self.df['confidence_score'].dropna()
        
        # Histogram
        ax1.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Confidence Score Distribution')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        
        # Box plot by correctness
        if 'is_correct' in self.df.columns:
            self.df.boxplot(column='confidence_score', by='is_correct', ax=ax2)
            ax2.set_title('Confidence Score by Correctness')
            ax2.set_xlabel('Correct')
            ax2.set_ylabel('Confidence Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confidence_distribution.{self.config.plot_format}', 
                   dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_vs_confidence(self):
        """Plot accuracy vs confidence correlation"""
        plt.figure(figsize=(8, 6))
        
        # Scatter plot
        plt.scatter(self.df['confidence_score'], self.df['is_correct'], alpha=0.6)
        
        # Add trend line
        z = np.polyfit(self.df['confidence_score'].dropna(), 
                      self.df['is_correct'].astype(int), 1)
        p = np.poly1d(z)
        plt.plot(self.df['confidence_score'].dropna(), 
                p(self.df['confidence_score'].dropna()), "r--", alpha=0.8)
        
        plt.title('Accuracy vs Confidence Score')
        plt.xlabel('Confidence Score')
        plt.ylabel('Correct (1) / Incorrect (0)')
        
        # Calculate and display correlation
        correlation = self.df[['confidence_score', 'is_correct']].corr().iloc[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'accuracy_vs_confidence.{self.config.plot_format}', 
                   dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_time_vs_accuracy(self):
        """Plot execution time vs accuracy"""
        plt.figure(figsize=(8, 6))
        
        # Box plot
        self.df.boxplot(column='execution_time', by='is_correct')
        plt.title('Execution Time vs Accuracy')
        plt.xlabel('Correct')
        plt.ylabel('Execution Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'execution_time_vs_accuracy.{self.config.plot_format}', 
                   dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_retry_distribution(self):
        """Plot retry distribution"""
        plt.figure(figsize=(8, 6))
        
        retry_counts = self.df['retry_count'].value_counts().sort_index()
        retry_counts.plot(kind='bar')
        plt.title('Retry Distribution')
        plt.xlabel('Number of Retries')
        plt.ylabel('Number of Cases')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'retry_distribution.{self.config.plot_format}', 
                   dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(self):
        """Plot error analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error rate pie chart
        error_counts = self.df['has_error'].value_counts()
        ax1.pie(error_counts.values, labels=['Success', 'Error'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Success vs Error Rate')
        
        # Execution time comparison
        if 'execution_time' in self.df.columns:
            success_times = self.df[~self.df['has_error']]['execution_time']
            error_times = self.df[self.df['has_error']]['execution_time']
            
            ax2.hist(success_times.dropna(), bins=20, alpha=0.7, label='Success', color='green')
            ax2.hist(error_times.dropna(), bins=20, alpha=0.7, label='Error', color='red')
            ax2.set_title('Execution Time: Success vs Error')
            ax2.set_xlabel('Execution Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'error_analysis.{self.config.plot_format}', 
                   dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report"""
        if not self.config.generate_report:
            return ""
        
        logger.info("Generating comprehensive report...")
        
        # Generate statistics
        summary_stats = self.generate_summary_statistics()
        performance_analysis = self.generate_performance_analysis()
        
        # Create report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'gaia_analysis_report_{timestamp}.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# GAIA Benchmark Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Results File:** {self.results_file}\n\n")
            
            # Executive Summary
            f.write(f"## Executive Summary\n\n")
            f.write(f"- **Total Test Cases:** {summary_stats['total_cases']}\n")
            f.write(f"- **Overall Accuracy:** {summary_stats['accuracy']:.2%}\n")
            f.write(f"- **Success Rate:** {summary_stats['success_rate']:.2%}\n")
            f.write(f"- **Error Rate:** {summary_stats['error_cases'] / summary_stats['total_cases']:.2%}\n\n")
            
            # Detailed Statistics
            f.write(f"## Detailed Statistics\n\n")
            
            # Execution Time Statistics
            if 'execution_time' in summary_stats:
                exec_stats = summary_stats['execution_time']
                f.write(f"### Execution Time Statistics\n\n")
                f.write(f"- **Mean:** {exec_stats['mean']:.2f}s\n")
                f.write(f"- **Median:** {exec_stats['median']:.2f}s\n")
                f.write(f"- **Standard Deviation:** {exec_stats['std']:.2f}s\n")
                f.write(f"- **Min:** {exec_stats['min']:.2f}s\n")
                f.write(f"- **Max:** {exec_stats['max']:.2f}s\n")
                f.write(f"- **25th Percentile:** {exec_stats['q25']:.2f}s\n")
                f.write(f"- **75th Percentile:** {exec_stats['q75']:.2f}s\n\n")
            
            # Confidence Statistics
            if 'confidence' in summary_stats:
                conf_stats = summary_stats['confidence']
                f.write(f"### Confidence Score Statistics\n\n")
                f.write(f"- **Mean:** {conf_stats['mean']:.3f}\n")
                f.write(f"- **Median:** {conf_stats['median']:.3f}\n")
                f.write(f"- **Standard Deviation:** {conf_stats['std']:.3f}\n")
                f.write(f"- **Min:** {conf_stats['min']:.3f}\n")
                f.write(f"- **Max:** {conf_stats['max']:.3f}\n\n")
            
            # Accuracy by Confidence Level
            if 'accuracy_by_confidence' in summary_stats:
                f.write(f"### Accuracy by Confidence Level\n\n")
                conf_acc = summary_stats['accuracy_by_confidence']
                for level in ['Low', 'Medium', 'High']:
                    if level in conf_acc['mean']:
                        f.write(f"- **{level} Confidence:** {conf_acc['mean'][level]:.2%} "
                               f"({conf_acc['sum'][level]}/{conf_acc['count'][level]} cases)\n")
                f.write("\n")
            
            # Performance Analysis
            f.write(f"## Performance Analysis\n\n")
            
            if 'execution_time_performance' in performance_analysis:
                perf = performance_analysis['execution_time_performance']
                f.write(f"### Execution Time Performance\n\n")
                f.write(f"- **Fast Cases (≤25th percentile):** {perf['fast_cases']}\n")
                f.write(f"- **Slow Cases (≥75th percentile):** {perf['slow_cases']}\n")
                f.write(f"- **Outliers (>2σ):** {perf['outliers']}\n\n")
            
            # Correlations
            if 'accuracy_time_correlation' in performance_analysis:
                f.write(f"### Correlations\n\n")
                f.write(f"- **Accuracy vs Execution Time:** {performance_analysis['accuracy_time_correlation']:.3f}\n")
                if 'confidence_accuracy_correlation' in performance_analysis:
                    f.write(f"- **Confidence vs Accuracy:** {performance_analysis['confidence_accuracy_correlation']:.3f}\n")
                f.write("\n")
            
            # Error Analysis
            if 'error_analysis' in performance_analysis:
                error_analysis = performance_analysis['error_analysis']
                f.write(f"### Error Analysis\n\n")
                f.write(f"- **Error Rate:** {error_analysis['error_rate']:.2%}\n")
                f.write(f"- **Avg Execution Time (Error Cases):** {error_analysis['avg_execution_time_on_error']:.2f}s\n")
                f.write(f"- **Avg Execution Time (Success Cases):** {error_analysis['avg_execution_time_on_success']:.2f}s\n\n")
            
            # Retry Statistics
            if 'retry_distribution' in summary_stats:
                f.write(f"## Retry Statistics\n\n")
                retry_dist = summary_stats['retry_distribution']
                for retries, count in sorted(retry_dist.items()):
                    f.write(f"- **{retries} Retries:** {count} cases\n")
                f.write("\n")
            
            # Recommendations
            f.write(f"## Recommendations\n\n")
            
            # Generate recommendations based on analysis
            recommendations = self._generate_recommendations(summary_stats, performance_analysis)
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n")
            
            # Generated Files
            f.write(f"## Generated Files\n\n")
            f.write(f"The following files were generated during this analysis:\n\n")
            f.write(f"- **Visualizations:** Multiple plots in {self.config.plot_format} format\n")
            f.write(f"- **Data Export:** CSV file with detailed results\n")
            f.write(f"- **This Report:** Comprehensive analysis report\n\n")
        
        logger.info(f"Comprehensive report saved to {report_file}")
        return str(report_file)
    
    def _generate_recommendations(self, summary_stats: Dict, performance_analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Accuracy recommendations
        if summary_stats['accuracy'] < 0.7:
            recommendations.append("Consider improving the answer extraction logic to increase accuracy")
        
        # Execution time recommendations
        if 'execution_time' in summary_stats:
            exec_stats = summary_stats['execution_time']
            if exec_stats['mean'] > 60:  # More than 1 minute average
                recommendations.append("Consider optimizing execution time - average is quite high")
            
            if exec_stats['std'] > exec_stats['mean'] * 0.5:  # High variance
                recommendations.append("High variance in execution times suggests inconsistent performance")
        
        # Confidence recommendations
        if 'confidence' in summary_stats:
            conf_stats = summary_stats['confidence']
            if conf_stats['mean'] < 0.5:
                recommendations.append("Low average confidence scores suggest uncertainty in predictions")
        
        # Error recommendations
        error_rate = summary_stats['error_cases'] / summary_stats['total_cases']
        if error_rate > 0.1:  # More than 10% errors
            recommendations.append("High error rate suggests need for better error handling")
        
        # Correlation recommendations
        if 'confidence_accuracy_correlation' in performance_analysis:
            corr = performance_analysis['confidence_accuracy_correlation']
            if corr < 0.3:
                recommendations.append("Low correlation between confidence and accuracy suggests confidence scoring needs improvement")
        
        return recommendations
    
    def export_results(self) -> str:
        """Export results to various formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export to CSV
        csv_file = self.output_dir / f'gaia_results_processed_{timestamp}.csv'
        self.df.to_csv(csv_file, index=False)
        
        # Export summary statistics
        summary_stats = self.generate_summary_statistics()
        stats_file = self.output_dir / f'gaia_summary_stats_{timestamp}.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to {csv_file} and {stats_file}")
        return str(csv_file)


def main():
    """Main function for results processing"""
    parser = argparse.ArgumentParser(description='Process and analyze GAIA benchmark results')
    parser.add_argument('--results_file', required=True, help='Path to GAIA results JSON file')
    parser.add_argument('--output_dir', default='./gaia_analysis', help='Output directory for analysis')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--no_report', action='store_true', help='Skip generating report')
    parser.add_argument('--plot_format', default='png', choices=['png', 'pdf', 'svg'], help='Plot format')
    parser.add_argument('--plot_dpi', type=int, default=300, help='Plot DPI')
    
    args = parser.parse_args()
    
    # Validate results file
    if not os.path.exists(args.results_file):
        print(f"Error: Results file does not exist: {args.results_file}")
        return 1
    
    try:
        # Create configuration
        config = AnalysisConfig(
            results_file=args.results_file,
            output_dir=args.output_dir,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            plot_format=args.plot_format,
            plot_dpi=args.plot_dpi
        )
        
        # Create processor
        processor = GAIAResultsProcessor(config)
        
        # Generate analysis
        processor.generate_visualizations()
        report_file = processor.generate_comprehensive_report()
        csv_file = processor.export_results()
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results exported to: {csv_file}")
        if report_file:
            print(f"Report generated: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
