"""
Script di analisi completo per i risultati della simulazione
Genera grafici e report dettagliati come nel sistema del collega
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Setup stile grafici
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class SimulationAnalyzer:
    """
    Analizzatore risultati simulazione
    """
    
    def __init__(self, metrics_dir: str = "/mnt/user-data/outputs"):
        """
        Args:
            metrics_dir: Directory con i file CSV delle metriche
        """
        self.metrics_dir = Path(metrics_dir)
        self.inv_df = None
        self.sched_df = None
        
        logger.info(f"SimulationAnalyzer initialized: {metrics_dir}")
    
    def load_data(self):
        """Carica i dati CSV"""
        logger.info("Loading data...")
        
        # Invocations
        inv_path = self.metrics_dir / "invocations.csv"
        if inv_path.exists():
            self.inv_df = pd.read_csv(inv_path)
            logger.info(f"  ✓ Loaded invocations: {len(self.inv_df)} records")
            
            # Calcola response time se non presente
            if 't_wait' in self.inv_df.columns and 't_exec' in self.inv_df.columns:
                self.inv_df['response_time'] = self.inv_df['t_wait'] + self.inv_df['t_exec']
                self.inv_df['response_time_ms'] = self.inv_df['response_time'] * 1000
                self.inv_df['t_wait_ms'] = self.inv_df['t_wait'] * 1000
        else:
            logger.warning(f"  ⚠️  Invocations file not found: {inv_path}")
        
        # Scheduling
        sched_path = self.metrics_dir / "scheduling.csv"
        if sched_path.exists():
            self.sched_df = pd.read_csv(sched_path)
            logger.info(f"  ✓ Loaded scheduling: {len(self.sched_df)} records")
        else:
            logger.warning(f"  ⚠️  Scheduling file not found: {sched_path}")
    
    def generate_all_plots(self, output_dir: str = "/mnt/user-data/outputs/plots"):
        """Genera tutti i grafici"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n📊 Generating plots in {output_dir}...")
        
        if self.inv_df is not None:
            # 1. Response time over time
            self.plot_response_time_over_time(output_path)
            
            # 2. Latency distribution
            self.plot_latency_distribution(output_path)
            
            # 3. Queue time analysis
            self.plot_queue_time_analysis(output_path)
            
            # 4. Throughput over time
            self.plot_throughput_over_time(output_path)
            
            # 5. Per-function statistics
            self.plot_per_function_stats(output_path)
            
            # 6. SLA compliance
            self.plot_sla_compliance(output_path)
        
        logger.info("✅ All plots generated")
    
    def plot_response_time_over_time(self, output_dir: Path):
        """Grafico response time nel tempo con P50, P95, P99"""
        if 'response_time_ms' not in self.inv_df.columns:
            return
        
        logger.info("  • Generating response time over time plot...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Crea time bins
        df = self.inv_df.copy()
        df['time_bin'] = (df['t_start'] // 10) * 10  # Bins di 10s
        
        # Calcola percentili per bin
        rt_stats = df.groupby('time_bin')['response_time_ms'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('p50', lambda x: x.quantile(0.50)),
            ('p95', lambda x: x.quantile(0.95)),
            ('p99', lambda x: x.quantile(0.99))
        ]).reset_index()
        
        # Plot 1: Percentili
        ax = axes[0]
        ax.plot(rt_stats['time_bin'], rt_stats['mean'], label='Mean', linewidth=2, alpha=0.7)
        ax.plot(rt_stats['time_bin'], rt_stats['p50'], label='P50', linewidth=2)
        ax.plot(rt_stats['time_bin'], rt_stats['p95'], label='P95', linewidth=2)
        ax.plot(rt_stats['time_bin'], rt_stats['p99'], label='P99', linewidth=2)
        
        # SLA threshold
        ax.axhline(y=100, color='red', linestyle='--', label='SLA (100ms)', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response Time (ms)')
        ax.set_title('Response Time Over Time (10s bins)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Throughput (requests/bin)
        ax = axes[1]
        ax.bar(rt_stats['time_bin'], rt_stats['count'], width=8, alpha=0.6, color='steelblue')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Requests per 10s')
        ax.set_title('Throughput Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "response_time_over_time.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_latency_distribution(self, output_dir: Path):
        """Distribuzione latenze (response time e queue time)"""
        if 'response_time_ms' not in self.inv_df.columns:
            return
        
        logger.info("  • Generating latency distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Response time distribution
        ax = axes[0]
        ax.hist(self.inv_df['response_time_ms'], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=100, color='red', linestyle='--', label='SLA (100ms)', linewidth=2)
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Response Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Queue time distribution
        if 't_wait_ms' in self.inv_df.columns:
            ax = axes[1]
            ax.hist(self.inv_df['t_wait_ms'], bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Queue Wait Time (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Queue Time Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "latency_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_queue_time_analysis(self, output_dir: Path):
        """Analisi tempi in coda"""
        if 't_wait_ms' not in self.inv_df.columns:
            return
        
        logger.info("  • Generating queue time analysis...")
        
        df = self.inv_df.copy()
        df['time_bin'] = (df['t_start'] // 10) * 10
        
        queue_stats = df.groupby('time_bin')['t_wait_ms'].agg([
            ('mean', 'mean'),
            ('p95', lambda x: x.quantile(0.95)),
            ('max', 'max')
        ]).reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(queue_stats['time_bin'], queue_stats['mean'], label='Mean queue time', linewidth=2)
        ax.plot(queue_stats['time_bin'], queue_stats['p95'], label='P95 queue time', linewidth=2)
        ax.fill_between(queue_stats['time_bin'], 0, queue_stats['mean'], alpha=0.3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Queue Wait Time (ms)')
        ax.set_title('Queue Time Analysis Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "queue_time_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_throughput_over_time(self, output_dir: Path):
        """Throughput nel tempo"""
        logger.info("  • Generating throughput plot...")
        
        df = self.inv_df.copy()
        df['time_bin'] = (df['t_start'] // 5) * 5  # Bins di 5s
        
        throughput = df.groupby('time_bin').size().reset_index(name='requests')
        throughput['rps'] = throughput['requests'] / 5.0  # RPS
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(throughput['time_bin'], throughput['rps'], linewidth=2, marker='o', markersize=4)
        ax.fill_between(throughput['time_bin'], 0, throughput['rps'], alpha=0.3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Requests Per Second (RPS)')
        ax.set_title('Throughput Over Time (5s bins)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "throughput_over_time.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_per_function_stats(self, output_dir: Path):
        """Statistiche per funzione"""
        if 'response_time_ms' not in self.inv_df.columns:
            return
        
        logger.info("  • Generating per-function statistics...")
        
        # Statistiche per funzione
        fn_stats = self.inv_df.groupby('function_name').agg({
            'response_time_ms': ['count', 'mean', 'std', lambda x: x.quantile(0.95)],
            't_wait_ms': ['mean', lambda x: x.quantile(0.95)] if 't_wait_ms' in self.inv_df.columns else 'count'
        }).reset_index()
        
        fn_stats.columns = ['function_name', 'count', 'mean_rt', 'std_rt', 'p95_rt', 'mean_wait', 'p95_wait']
        fn_stats = fn_stats.sort_values('count', ascending=False)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Invocation count
        ax = axes[0]
        ax.bar(range(len(fn_stats)), fn_stats['count'], alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(fn_stats)))
        ax.set_xticklabels(fn_stats['function_name'], rotation=45, ha='right')
        ax.set_ylabel('Number of Invocations')
        ax.set_title('Invocations per Function')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: P95 response time
        ax = axes[1]
        bars = ax.bar(range(len(fn_stats)), fn_stats['p95_rt'], alpha=0.7, edgecolor='black')
        
        # Colora in rosso se sopra SLA
        for i, (idx, row) in enumerate(fn_stats.iterrows()):
            if row['p95_rt'] > 100:
                bars[i].set_color('red')
        
        ax.axhline(y=100, color='red', linestyle='--', label='SLA (100ms)', linewidth=2)
        ax.set_xticks(range(len(fn_stats)))
        ax.set_xticklabels(fn_stats['function_name'], rotation=45, ha='right')
        ax.set_ylabel('P95 Response Time (ms)')
        ax.set_title('P95 Response Time per Function')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "per_function_stats.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_sla_compliance(self, output_dir: Path):
        """SLA compliance over time"""
        if 'response_time_ms' not in self.inv_df.columns:
            return
        
        logger.info("  • Generating SLA compliance plot...")
        
        df = self.inv_df.copy()
        df['time_bin'] = (df['t_start'] // 10) * 10
        df['sla_violation'] = df['response_time_ms'] > 100
        
        sla_stats = df.groupby('time_bin').agg({
            'sla_violation': ['sum', 'count']
        }).reset_index()
        
        sla_stats.columns = ['time_bin', 'violations', 'total']
        sla_stats['compliance_rate'] = (1 - sla_stats['violations'] / sla_stats['total']) * 100
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Compliance rate
        ax = axes[0]
        ax.plot(sla_stats['time_bin'], sla_stats['compliance_rate'], linewidth=2, marker='o', markersize=4)
        ax.axhline(y=99, color='orange', linestyle='--', label='99% SLA target', linewidth=2)
        ax.axhline(y=95, color='red', linestyle='--', label='95% minimum', linewidth=2)
        ax.fill_between(sla_stats['time_bin'], 0, sla_stats['compliance_rate'], alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('SLA Compliance (%)')
        ax.set_title('SLA Compliance Over Time (P95 < 100ms)')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Violations count
        ax = axes[1]
        ax.bar(sla_stats['time_bin'], sla_stats['violations'], width=8, alpha=0.6, color='red', edgecolor='black')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('SLA Violations (count)')
        ax.set_title('SLA Violations Over Time')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "sla_compliance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_path: str = "/mnt/user-data/outputs/report.txt"):
        """Genera report testuale"""
        logger.info(f"\n📄 Generating text report...")
        
        if self.inv_df is None:
            logger.warning("  ⚠️  No data to generate report")
            return
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SIMULATION RESULTS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Global stats
            f.write("GLOBAL STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total invocations:     {len(self.inv_df)}\n")
            
            if 'response_time_ms' in self.inv_df.columns:
                f.write(f"Mean response time:    {self.inv_df['response_time_ms'].mean():.2f} ms\n")
                f.write(f"P50 response time:     {self.inv_df['response_time_ms'].quantile(0.50):.2f} ms\n")
                f.write(f"P95 response time:     {self.inv_df['response_time_ms'].quantile(0.95):.2f} ms\n")
                f.write(f"P99 response time:     {self.inv_df['response_time_ms'].quantile(0.99):.2f} ms\n")
                f.write(f"Max response time:     {self.inv_df['response_time_ms'].max():.2f} ms\n")
            
            if 't_wait_ms' in self.inv_df.columns:
                f.write(f"\nMean queue time:       {self.inv_df['t_wait_ms'].mean():.2f} ms\n")
                f.write(f"P95 queue time:        {self.inv_df['t_wait_ms'].quantile(0.95):.2f} ms\n")
            
            # SLA compliance
            if 'response_time_ms' in self.inv_df.columns:
                sla_violations = (self.inv_df['response_time_ms'] > 100).sum()
                sla_compliance = (1 - sla_violations / len(self.inv_df)) * 100
                
                f.write(f"\nSLA COMPLIANCE (P95 < 100ms)\n")
                f.write("-"*70 + "\n")
                f.write(f"Compliance rate:       {sla_compliance:.2f}%\n")
                f.write(f"Violations:            {sla_violations} / {len(self.inv_df)}\n")
            
            # Per-function stats
            f.write(f"\nPER-FUNCTION STATISTICS\n")
            f.write("-"*70 + "\n")
            
            for fn_name in sorted(self.inv_df['function_name'].unique()):
                fn_df = self.inv_df[self.inv_df['function_name'] == fn_name]
                f.write(f"\n{fn_name}:\n")
                f.write(f"  Invocations:         {len(fn_df)}\n")
                
                if 'response_time_ms' in fn_df.columns:
                    f.write(f"  Mean response time:  {fn_df['response_time_ms'].mean():.2f} ms\n")
                    f.write(f"  P95 response time:   {fn_df['response_time_ms'].quantile(0.95):.2f} ms\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"  ✓ Report saved to {output_path}")


def main():
    """Main entry point"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    # Crea analyzer
    analyzer = SimulationAnalyzer()
    
    # Carica dati
    analyzer.load_data()
    
    # Genera plots
    analyzer.generate_all_plots()
    
    # Genera report
    analyzer.generate_report()
    
    logger.info("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()