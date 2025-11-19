import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MetricsAnalyzer:
    """
    Analizza ed estrae metriche dettagliate dalla simulazione faas-sim
    """
    
    def __init__(self, env):
        self.env = env
        self.metrics = env.metrics
    
    def extract_all_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Estrae tutte le metriche disponibili dalla simulazione
        """
        metrics = {}
        
        # Lista di metriche comuni in faas-sim
        metric_names = [
            'invocations',
            'schedule', 
            'scale',
            'deployments',
            'replicas',
            'node_resources',
            'function_resources',
            'network_traffic',
            'image_pulls',
            'container_startup'
        ]
        
        for name in metric_names:
            try:
                df = self.metrics.extract_dataframe(name)
                if df is not None and len(df) > 0:
                    metrics[name] = df
                    logger.debug(f"Extracted {name}: {len(df)} records")
            except Exception as e:
                logger.debug(f"Metric {name} not available: {e}")
        
        return metrics
    
    def analyze_invocations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizza metriche di invocazione (latency, throughput)
        
        IMPORTANTE: faas-sim registra le invocazioni con colonne:
        - t_wait: tempo di attesa in coda
        - t_exec: tempo di esecuzione
        - t_start: timestamp di inizio
        
        Dobbiamo calcolare: response_time = t_wait + t_exec
        """
        if df is None or len(df) == 0:
            return {}
        
        # Crea una copia per evitare SettingWithCopyWarning
        df = df.copy()
        
        # Calcola response_time dalle colonne di faas-sim
        if 't_wait' in df.columns and 't_exec' in df.columns:
            df['response_time'] = df['t_wait'] + df['t_exec']
            df['queue_time'] = df['t_wait']
            df['execution_time'] = df['t_exec']
            
            logger.debug(
                f"Calculated response_time: "
                f"mean={df['response_time'].mean():.3f}s, "
                f"p95={df['response_time'].quantile(0.95):.3f}s"
            )
        else:
            logger.warning(
                f"Missing t_wait/t_exec columns! "
                f"Available: {list(df.columns)}"
            )
        
        # Calcola start_time e end_time per throughput
        if 't_start' in df.columns and 'response_time' in df.columns:
            df['start_time'] = df['t_start']
            df['end_time'] = df['t_start'] + df['response_time']
        
        analysis = {
            'total_invocations': len(df),
            'functions': {}
        }
        
        # Analizza per funzione
        if 'function_name' in df.columns:
            for fn_name in df['function_name'].unique():
                fn_df = df[df['function_name'] == fn_name]
                
                fn_analysis = {
                    'count': len(fn_df),
                }
                
                # Response time (t_wait + t_exec)
                if 'response_time' in fn_df.columns:
                    fn_analysis['response_time'] = {
                        'mean': fn_df['response_time'].mean(),
                        'median': fn_df['response_time'].median(),
                        'min': fn_df['response_time'].min(),
                        'max': fn_df['response_time'].max(),
                        'std': fn_df['response_time'].std(),
                        'p95': fn_df['response_time'].quantile(0.95),
                        'p99': fn_df['response_time'].quantile(0.99)
                    }
                
                # Execution time (t_exec)
                if 'execution_time' in fn_df.columns:
                    fn_analysis['execution_time'] = {
                        'mean': fn_df['execution_time'].mean(),
                        'median': fn_df['execution_time'].median(),
                        'min': fn_df['execution_time'].min(),
                        'max': fn_df['execution_time'].max()
                    }
                
                # Queue time (t_wait)
                if 'queue_time' in fn_df.columns:
                    fn_analysis['queue_time'] = {
                        'mean': fn_df['queue_time'].mean(),
                        'median': fn_df['queue_time'].median(),
                        'min': fn_df['queue_time'].min(),
                        'max': fn_df['queue_time'].max()
                    }
                
                # Throughput (invocations per second)
                if 'start_time' in fn_df.columns and 'end_time' in fn_df.columns:
                    duration = fn_df['end_time'].max() - fn_df['start_time'].min()
                    if duration > 0:
                        fn_analysis['throughput'] = len(fn_df) / duration
                
                analysis['functions'][fn_name] = fn_analysis
        
        return analysis
    
    def analyze_scheduling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizza decisioni di scheduling
        """
        if df is None or len(df) == 0:
            return {}
        
        analysis = {
            'total_schedules': len(df),
            'nodes': {}
        }
        
        # Trova colonna del nodo
        node_col = None
        for col in ['node', 'node_name', 'target_node', 'scheduled_to']:
            if col in df.columns:
                node_col = col
                break
        
        if node_col:
            for node_name in df[node_col].unique():
                node_df = df[df[node_col] == node_name]
                
                node_analysis = {
                    'pods_scheduled': len(node_df)
                }
                
                # Scheduling time
                if 'scheduling_time' in node_df.columns:
                    node_analysis['scheduling_time'] = {
                        'mean': node_df['scheduling_time'].mean(),
                        'max': node_df['scheduling_time'].max()
                    }
                
                # Pod names
                pod_col = 'pod' if 'pod' in df.columns else 'pod_name'
                if pod_col in node_df.columns:
                    node_analysis['pods'] = list(node_df[pod_col].unique())
                
                analysis['nodes'][node_name] = node_analysis
        
        return analysis
    
    def analyze_resources(self, node_resources: pd.DataFrame = None, 
                         function_resources: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analizza utilizzo risorse (CPU, RAM)
        """
        analysis = {}
        
        if node_resources is not None and len(node_resources) > 0:
            analysis['node_resources'] = {}
            
            if 'node' in node_resources.columns:
                for node_name in node_resources['node'].unique():
                    node_df = node_resources[node_resources['node'] == node_name]
                    
                    node_analysis = {}
                    
                    # CPU usage
                    if 'cpu_usage' in node_df.columns:
                        node_analysis['cpu'] = {
                            'mean': node_df['cpu_usage'].mean(),
                            'max': node_df['cpu_usage'].max(),
                            'min': node_df['cpu_usage'].min()
                        }
                    
                    # Memory usage
                    if 'memory_usage' in node_df.columns:
                        node_analysis['memory'] = {
                            'mean': node_df['memory_usage'].mean(),
                            'max': node_df['memory_usage'].max(),
                            'min': node_df['memory_usage'].min()
                        }
                    
                    analysis['node_resources'][node_name] = node_analysis
        
        if function_resources is not None and len(function_resources) > 0:
            analysis['function_resources'] = {}
            
            if 'function_name' in function_resources.columns:
                for fn_name in function_resources['function_name'].unique():
                    fn_df = function_resources[function_resources['function_name'] == fn_name]
                    
                    fn_analysis = {}
                    
                    if 'cpu_usage' in fn_df.columns:
                        fn_analysis['cpu'] = {
                            'mean': fn_df['cpu_usage'].mean(),
                            'max': fn_df['cpu_usage'].max()
                        }
                    
                    if 'memory_usage' in fn_df.columns:
                        fn_analysis['memory'] = {
                            'mean': fn_df['memory_usage'].mean(),
                            'max': fn_df['memory_usage'].max()
                        }
                    
                    analysis['function_resources'][fn_name] = fn_analysis
        
        return analysis
    
    def analyze_scaling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizza eventi di scaling
        """
        if df is None or len(df) == 0:
            return {}
        
        analysis = {
            'total_events': len(df),
            'functions': {}
        }
        
        if 'function_name' in df.columns:
            for fn_name in df['function_name'].unique():
                fn_df = df[df['function_name'] == fn_name]
                
                fn_analysis = {
                    'scale_events': len(fn_df)
                }
                
                # Scale up/down
                if 'action' in fn_df.columns:
                    scale_up = len(fn_df[fn_df['action'] == 'scale_up'])
                    scale_down = len(fn_df[fn_df['action'] == 'scale_down'])
                    fn_analysis['scale_up'] = scale_up
                    fn_analysis['scale_down'] = scale_down
                
                # Replica count over time
                if 'replica_count' in fn_df.columns:
                    fn_analysis['replicas'] = {
                        'min': fn_df['replica_count'].min(),
                        'max': fn_df['replica_count'].max(),
                        'mean': fn_df['replica_count'].mean()
                    }
                
                analysis['functions'][fn_name] = fn_analysis
        
        return analysis
    
    def print_summary(self):
        """
        Stampa un riepilogo completo di tutte le metriche
        """
        logger.info("\n" + "="*70)
        logger.info("DETAILED METRICS ANALYSIS")
        logger.info("="*70)
        
        # Estrai tutte le metriche
        all_metrics = self.extract_all_metrics()
        
        logger.info(f"\nAvailable metrics: {list(all_metrics.keys())}")
        
        # 1. INVOCATIONS
        if 'invocations' in all_metrics:
            logger.info("\n" + "-"*70)
            logger.info("INVOCATIONS ANALYSIS")
            logger.info("-"*70)
            
            inv_analysis = self.analyze_invocations(all_metrics['invocations'])
            logger.info(f"Total invocations: {inv_analysis.get('total_invocations', 0)}")
            
            for fn_name, fn_data in inv_analysis.get('functions', {}).items():
                logger.info(f"\n  {fn_name}:")
                logger.info(f"    Count: {fn_data.get('count', 0)}")
                
                if 'response_time' in fn_data:
                    rt = fn_data['response_time']
                    logger.info(f"    Response Time:")
                    logger.info(f"      Mean:   {rt.get('mean', 0):.3f}s")
                    logger.info(f"      Median: {rt.get('median', 0):.3f}s")
                    logger.info(f"      Min:    {rt.get('min', 0):.3f}s")
                    logger.info(f"      Max:    {rt.get('max', 0):.3f}s")
                    logger.info(f"      P95:    {rt.get('p95', 0):.3f}s")
                    logger.info(f"      P99:    {rt.get('p99', 0):.3f}s")
                
                if 'execution_time' in fn_data:
                    et = fn_data['execution_time']
                    logger.info(f"    Execution Time:")
                    logger.info(f"      Mean: {et.get('mean', 0):.3f}s")
                    logger.info(f"      Median: {et.get('median', 0):.3f}s")
                    logger.info(f"      Max:  {et.get('max', 0):.3f}s")
                
                if 'queue_time' in fn_data:
                    qt = fn_data['queue_time']
                    logger.info(f"    Queue Time:")
                    logger.info(f"      Mean: {qt.get('mean', 0):.3f}s")
                    logger.info(f"      Median: {qt.get('median', 0):.3f}s")
                    logger.info(f"      Max:  {qt.get('max', 0):.3f}s")
                
                if 'throughput' in fn_data:
                    logger.info(f"    Throughput: {fn_data['throughput']:.2f} req/s")
        
        # 2. SCHEDULING
        if 'schedule' in all_metrics:
            logger.info("\n" + "-"*70)
            logger.info("SCHEDULING ANALYSIS")
            logger.info("-"*70)
            
            sched_analysis = self.analyze_scheduling(all_metrics['schedule'])
            logger.info(f"Total scheduling decisions: {sched_analysis.get('total_schedules', 0)}")
            
            for node_name, node_data in sched_analysis.get('nodes', {}).items():
                logger.info(f"\n  {node_name}:")
                logger.info(f"    Pods scheduled: {node_data.get('pods_scheduled', 0)}")
                
                if 'pods' in node_data:
                    for pod in node_data['pods'][:5]:  # Mostra solo i primi 5
                        logger.info(f"      - {pod}")
                    if len(node_data['pods']) > 5:
                        logger.info(f"      ... and {len(node_data['pods']) - 5} more")
                
                if 'scheduling_time' in node_data:
                    st = node_data['scheduling_time']
                    logger.info(f"    Scheduling time: {st.get('mean', 0):.3f}s (mean)")
        
        # 3. SCALING
        if 'scale' in all_metrics:
            logger.info("\n" + "-"*70)
            logger.info("SCALING ANALYSIS")
            logger.info("-"*70)
            
            scale_analysis = self.analyze_scaling(all_metrics['scale'])
            logger.info(f"Total scaling events: {scale_analysis.get('total_events', 0)}")
            
            for fn_name, fn_data in scale_analysis.get('functions', {}).items():
                logger.info(f"\n  {fn_name}:")
                logger.info(f"    Scale up events: {fn_data.get('scale_up', 0)}")
                logger.info(f"    Scale down events: {fn_data.get('scale_down', 0)}")
                
                if 'replicas' in fn_data:
                    rep = fn_data['replicas']
                    logger.info(f"    Replica count: {rep.get('min', 0)} - {rep.get('max', 0)} (avg: {rep.get('mean', 0):.1f})")
        
        # 4. RESOURCES
        node_res = all_metrics.get('node_resources')
        fn_res = all_metrics.get('function_resources')
        
        if node_res is not None or fn_res is not None:
            logger.info("\n" + "-"*70)
            logger.info("RESOURCE USAGE ANALYSIS")
            logger.info("-"*70)
            
            res_analysis = self.analyze_resources(node_res, fn_res)
            
            if 'node_resources' in res_analysis:
                logger.info("\n  Node Resources:")
                for node_name, node_data in res_analysis['node_resources'].items():
                    logger.info(f"\n    {node_name}:")
                    
                    if 'cpu' in node_data:
                        cpu = node_data['cpu']
                        logger.info(f"      CPU usage: {cpu.get('mean', 0):.1f}% (max: {cpu.get('max', 0):.1f}%)")
                    
                    if 'memory' in node_data:
                        mem = node_data['memory']
                        logger.info(f"      Memory usage: {mem.get('mean', 0):.1f}% (max: {mem.get('max', 0):.1f}%)")
            
            if 'function_resources' in res_analysis:
                logger.info("\n  Function Resources:")
                for fn_name, fn_data in res_analysis['function_resources'].items():
                    logger.info(f"\n   {fn_name}:")
                    
                    if 'cpu' in fn_data:
                        cpu = fn_data['cpu']
                        logger.info(f"      CPU: {cpu.get('mean', 0):.1f}% (max: {cpu.get('max', 0):.1f}%)")
                    
                    if 'memory' in fn_data:
                        mem = fn_data['memory']
                        logger.info(f"      Memory: {mem.get('mean', 0):.1f}% (max: {mem.get('max', 0):.1f}%)")
        
        # 5. RAW DATA (per debug)
        logger.info("\n" + "-"*70)
        logger.info("RAW DATAFRAMES (first 5 rows)")
        logger.info("-"*70)
        
        for name, df in all_metrics.items():
            logger.info(f"\n  {name} ({len(df)} records):")
            logger.info(f"  Columns: {list(df.columns)}")
            if len(df) > 0:
                logger.info(f"\n{df.head().to_string()}")
        
        logger.info("\n" + "="*70)