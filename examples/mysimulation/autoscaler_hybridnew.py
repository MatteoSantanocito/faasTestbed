"""
Hybrid Autoscaler - Replica logica del testbed
Combines reactive (P95-based) + proactive (ML-based) scaling
"""

import logging
from sim.core import Environment
from sim.faas import FunctionState

logger = logging.getLogger(__name__)


class HybridAutoscaler:
    """
    Autoscaler ibrido che replica la logica del testbed:
    
    REACTIVE:
    - Scale UP se P95 > 100ms
    - Scale DOWN se P95 < 30ms AND low load
    
    PROACTIVE (quando implementato ML):
    - Scale UP se ML predice spike (prob > 0.65)
    - Scale UP urgente se ML prob > 0.80
    - Scale DOWN se ML prob < 0.20 AND low load
    """
    
    def __init__(self, env: Environment, 
                 check_interval=10.0,
                 cooldown_period=50.0,
                 enable_proactive=False):
        """
        Args:
            env: Simulation environment
            check_interval: Intervallo controllo (secondi)
            cooldown_period: Periodo cooldown tra scale (secondi)
            enable_proactive: Abilita ML proattivo
        """
        self.env = env
        self.check_interval = check_interval
        self.cooldown_period = cooldown_period
        self.enable_proactive = enable_proactive
        
        # Thresholds (dal tuo codice testbed)
        self.rt_p95_scale_up = 100  # ms
        self.rt_p95_scale_down = 30  # ms
        
        # ML thresholds (se proactive enabled)
        self.ml_prob_urgent = 0.80
        self.ml_prob_prepare = 0.65
        self.ml_prob_safe = 0.20
        
        # Cooldown tracking
        self.last_scale_time = {}
        
        logger.info("🤖 HybridAutoscaler initialized:")
        logger.info(f"   Mode: {'HYBRID (reactive + proactive)' if enable_proactive else 'REACTIVE only'}")
        logger.info(f"   Check interval: {check_interval}s")
        logger.info(f"   Cooldown: {cooldown_period}s")
        logger.info(f"   Thresholds: P95={self.rt_p95_scale_up}ms (up), {self.rt_p95_scale_down}ms (down)")
    
    def run(self):
        """Main autoscaler loop"""
        logger.info("HybridAutoscaler started")
        
        while True:
            # Wait for next check
            yield self.env.timeout(self.check_interval)
            
            # Check ogni deployment
            for deployment in self.env.faas.get_deployments():
                yield from self._check_deployment(deployment)
    
    def _check_deployment(self, deployment):
        """Controlla singolo deployment per scaling"""
        
        fn_name = deployment.name
        
        # Skip se in cooldown
        if not self._can_scale(fn_name):
            logger.debug(f"[{fn_name}] In cooldown, skipping")
            return
        
        # Get replicas
        replicas = self.env.faas.get_replicas(fn_name, FunctionState.RUNNING)
        current_count = len(replicas)
        
        if current_count == 0:
            return
        
        # Calcola metriche
        metrics = self._calculate_metrics(fn_name, replicas)
        
        # ⚠️ CRITICAL: Non scalare se non ci sono richieste!
        # Durante warmup/deploy, RPS=0 → attendi workload reale
        if metrics['rps'] < 0.01:  # Praticamente zero
            return
        
        # Decisione scaling
        decision = self._make_decision(
            fn_name, 
            current_count, 
            metrics,
            deployment.scaling_config
        )
        
        
        # Applica scaling
        if decision['scale']:
            target = decision['target_replicas']
            reason = decision['reason']
            scale_type = decision['type']
            
            if target > current_count:
                # Scale UP
                delta = target - current_count
                yield from self.env.faas.scale_up(fn_name, delta)
                self._mark_cooldown(fn_name)
                
            elif target < current_count:
                # Scale DOWN
                delta = current_count - target
                yield from self.env.faas.scale_down(fn_name, delta)
                self._mark_cooldown(fn_name)
    
    def _calculate_metrics(self, fn_name, replicas):
        """
        Calcola metriche REALI dalle invocazioni recenti
        
        Legge dal metrics dataframe gli ultimi 60s di dati
        """
        if not replicas:
            return {
                'p95_response_time': 50.0,
                'rps': 0,
                'failure_rate': 0.0,
                'avg_cpu': 0.0
            }
        
        try:
            # Estrai invocazioni recenti (ultimi 60s)
            inv_df = self.env.metrics.extract_dataframe('invocations')
            
            if inv_df is None or len(inv_df) == 0:
                # Nessun dato ancora - attendi prima di scalare
                return {
                    'p95_response_time': 50.0,  # ← Valore BASSO (no trigger)
                    'rps': 0,
                    'failure_rate': 0.0,
                    'avg_cpu': 0.5
                }
            
            # Filtra per function e ultimi 60s
            window_start = max(0, self.env.now - 60)
            recent = inv_df[
                (inv_df['function_name'] == fn_name) &
                (inv_df['t_start'] >= window_start)
            ]
            
            if len(recent) == 0:
                # Nessuna invocazione recente - NON scalare!
                # Attendi che arrivino dati reali
                num_replicas = len(replicas)
                return {
                    'p95_response_time': 50.0,  # ← Valore BASSO (no trigger scaling)
                    'rps': 0,
                    'failure_rate': 0.0,
                    'avg_cpu': 0.3
                }
            
            # Calcola response time REALE
            if 't_wait' in recent.columns and 't_exec' in recent.columns:
                response_times = (recent['t_wait'] + recent['t_exec']) * 1000  # → ms
                p95_real = response_times.quantile(0.95)
            else:
                p95_real = 100.0
            
            # Calcola RPS negli ultimi 60s
            window_duration = min(60, self.env.now - window_start)
            rps_real = len(recent) / window_duration if window_duration > 0 else 0
            
            return {
                'p95_response_time': p95_real,
                'rps': rps_real,
                'failure_rate': 0.0,
                'avg_cpu': 0.5
            }
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for {fn_name}: {e}")
            # Fallback: stima conservativa che TRIGGERA scaling
            num_replicas = len(replicas)
            return {
                'p95_response_time': 120.0,  # Valore alto per triggerare scaling
                'rps': num_replicas * 10,
                'failure_rate': 0.0,
                'avg_cpu': 0.5
            }
    
    def _make_decision(self, fn_name, current_count, metrics, scaling_config):
        """
        Decisione scaling (replica logica testbed)
        
        Returns:
            dict con: scale (bool), target_replicas (int), reason (str), type (str)
        """
        p95 = metrics['p95_response_time']
        rps = metrics['rps']
        
        min_replicas = scaling_config.scale_min
        max_replicas = scaling_config.scale_max
        
        # ========== REACTIVE SCALE UP ==========
        if p95 > self.rt_p95_scale_up:
            if current_count < max_replicas:
                return {
                    'scale': True,
                    'target_replicas': current_count + 1,
                    'reason': f'P95={p95:.1f}ms >> threshold',
                    'type': 'reactive'
                }
            else:
                return {
                    'scale': False,
                    'target_replicas': current_count,
                    'reason': f'Already at max ({max_replicas})',
                    'type': 'none'
                }
        
        # ========== REACTIVE SCALE DOWN ==========
        if p95 < self.rt_p95_scale_down:
            if current_count > min_replicas:
                # Assicurati che non ci sia alto carico
                capacity_threshold = 0.7
                if rps < (capacity_threshold * current_count * 10):
                    return {
                        'scale': True,
                        'target_replicas': current_count - 1,
                        'reason': f'P95={p95:.1f}ms << threshold, low load',
                        'type': 'reactive'
                    }
        
        # ========== PROACTIVE (ML-BASED) ==========
        if self.enable_proactive:
            # TODO: Integra predizioni ML
            # ml_prediction = self._get_ml_prediction()
            # if ml_prediction['spike_prob'] > self.ml_prob_urgent:
            #     ... scale up proattivo
            pass
        
        # NO ACTION
        return {
            'scale': False,
            'target_replicas': current_count,
            'reason': 'Metrics OK',
            'type': 'none'
        }
    
    def _can_scale(self, fn_name):
        """Check cooldown period"""
        if fn_name not in self.last_scale_time:
            return True
        
        elapsed = self.env.now - self.last_scale_time[fn_name]
        return elapsed >= self.cooldown_period
    
    def _mark_cooldown(self, fn_name):
        """Mark scaling time for cooldown"""
        self.last_scale_time[fn_name] = self.env.now
    
    def _get_ml_prediction(self):
        """
        Get ML prediction (placeholder)
        
        Nel testbed reale:
        - Legge da /tmp/prediction_output_classifier.json
        - Contiene: spike_probability, risk_level, etc.
        
        Nella simulazione:
        - TODO: Implementare ML predictor che analizza trend
        - Oppure: replay predictions da testbed reale
        """
        return {
            'spike_probability': 0.0,
            'risk_level': 'low',
            'horizon_seconds': 30
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("HYBRID AUTOSCALER TEST")
    print("="*70)
    
    # Test configuration
    from sim.core import Environment
    from sim.topology import Topology
    
    t = Topology()
    env = Environment(t)
    
    # Test reactive mode
    autoscaler_reactive = HybridAutoscaler(
        env, 
        check_interval=10,
        enable_proactive=False
    )
    
    print("\n✅ Reactive autoscaler created")
    
    # Test hybrid mode
    autoscaler_hybrid = HybridAutoscaler(
        env,
        check_interval=10,
        enable_proactive=True
    )
    
    print("✅ Hybrid autoscaler created")
    print("\n" + "="*70)