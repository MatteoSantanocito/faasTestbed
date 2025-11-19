"""
Custom Hybrid Autoscaler che combina:
1. Componente REATTIVA: basata su soglie (latenza, RPS, code)
2. Componente PROATTIVA: integrazione ML predictions (preparato per future estensioni)
3. Meccanismi di stabilità: cooldown, hysteresis, debouncing

Ispirato al CPA del collega (Custom Pod Autoscaler)
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
from sim.core import Environment

logger = logging.getLogger(__name__)


@dataclass
class AutoscalerConfig:
    """Configurazione autoscaler"""
    # Parametri reattivi
    latency_p95_threshold_ms: float = 90.0      # Soglia P95 latenza (ms)
    latency_p95_scale_down_ms: float = 30.0     # Soglia scale-down
    rps_per_replica_target: float = 10.0        # Target RPS per replica
    error_rate_threshold: float = 0.01          # 1% error rate
    
    # Parametri temporali
    check_interval: float = 10.0                # Ogni quanto controlla (s)
    cooldown_period: float = 50.0               # Cooldown tra scaling (s)
    scale_down_cooldown: float = 120.0          # Cooldown più lungo per scale-down
    
    # Parametri predittivi (per futuro ML)
    enable_proactive: bool = False
    prediction_horizon: float = 39.0            # Orizzonte predizione (s)
    confidence_threshold_urgent: float = 0.70   # Soglia per scale-up urgente
    confidence_threshold_elevated: float = 0.60 # Soglia per scale-up preventivo
    
    # Parametri di stabilità
    min_observations_for_scale_down: int = 3    # Conferma trend prima di scale-down


class HybridAutoscaler:
    """
    Autoscaler ibrido che combina approccio reattivo e proattivo
    
    REACTIVE MODE:
    - Monitora latenza P95, RPS, error rate
    - Scala immediatamente se soglie superate
    
    PROACTIVE MODE (opzionale):
    - Integra predizioni ML per anticipare spike
    - Scala preventivamente basandosi su confidence
    """
    
    def __init__(
        self,
        env: Environment,
        config: Optional[AutoscalerConfig] = None
    ):
        self.env = env
        self.config = config or AutoscalerConfig()
        
        # Stato per decisioni
        self.last_scale_time: Dict[str, float] = {}
        self.observation_history: Dict[str, list] = {}  # Per trend analysis
        self.request_counts: Dict[str, int] = {}
        self.last_check_time: float = 0
        
        # ML predictions (placeholder per futuro)
        self.ml_predictions: Dict[str, Dict] = {}
        
        mode = "HYBRID (reactive + proactive)" if self.config.enable_proactive else "REACTIVE only"
        logger.info(f"🤖 HybridAutoscaler initialized in {mode} mode")
        logger.info(f"   Thresholds: P95={self.config.latency_p95_threshold_ms}ms, "
                   f"RPS/replica={self.config.rps_per_replica_target}")
    
    def run(self):
        """Background process che monitora e scala"""
        logger.info("HybridAutoscaler started")
        
        self.last_check_time = self.env.now
        
        while True:
            yield self.env.timeout(self.config.check_interval)
            yield from self._evaluate_and_scale()
    
    def _evaluate_and_scale(self):
        """Valuta metriche e decide scaling"""
        current_time = self.env.now
        time_elapsed = current_time - self.last_check_time
        
        if time_elapsed <= 0:
            return
        
        faas = self.env.faas
        deployments = faas.get_deployments()
        
        for deployment in deployments:
            fn_name = deployment.fn.name
            
            # Raccogli metriche
            metrics = self._collect_metrics(fn_name, time_elapsed)
            if metrics is None:
                continue
            
            # Storicizza osservazione
            if fn_name not in self.observation_history:
                self.observation_history[fn_name] = []
            self.observation_history[fn_name].append(metrics)
            
            # Mantieni solo ultimi N
            max_history = 30
            if len(self.observation_history[fn_name]) > max_history:
                self.observation_history[fn_name] = self.observation_history[fn_name][-max_history:]
            
            # ========== DECISIONE DI SCALING ==========
            decision = self._make_scaling_decision(fn_name, metrics, deployment)
            
            if decision['action'] == 'scale_up':
                yield from self._scale_up(
                    fn_name, 
                    decision['count'], 
                    deployment,
                    reason=decision['reason']
                )
            elif decision['action'] == 'scale_down':
                yield from self._scale_down(
                    fn_name, 
                    decision['count'], 
                    deployment,
                    reason=decision['reason']
                )
        
        self.last_check_time = current_time
    
    def _collect_metrics(self, fn_name: str, time_elapsed: float) -> Optional[Dict]:
        """
        Raccoglie metriche per una funzione
        
        Returns:
            Dict con: replicas, rps, p95_latency, error_rate, avg_queue_time
        """
        faas = self.env.faas
        replicas = faas.get_replicas(fn_name)
        
        if not replicas:
            return None
        
        current_replicas = len(replicas)
        
        # Conta richieste nel periodo
        new_count = self._count_function_invocations(fn_name)
        old_count = self.request_counts.get(fn_name, 0)
        requests_in_period = max(0, new_count - old_count)
        self.request_counts[fn_name] = new_count
        
        # Calcola RPS
        rps = requests_in_period / time_elapsed if time_elapsed > 0 else 0
        
        # Calcola latenza P95
        p95_latency = self._get_p95_latency(fn_name)
        
        # Calcola error rate (placeholder, faas-sim non simula errori per default)
        error_rate = 0.0
        
        # Calcola tempo in coda
        avg_queue_time = self._get_avg_queue_time(fn_name)
        
        return {
            'replicas': current_replicas,
            'rps': rps,
            'p95_latency_ms': p95_latency * 1000,  # Converti in ms
            'error_rate': error_rate,
            'avg_queue_time': avg_queue_time,
            'rps_per_replica': rps / current_replicas if current_replicas > 0 else 0
        }
    
    def _make_scaling_decision(self, fn_name: str, metrics: Dict, deployment) -> Dict:
        """
        Combina logica reattiva e proattiva per decidere scaling
        
        Returns:
            Dict con: action ('scale_up'/'scale_down'/'none'), count, reason
        """
        current_replicas = metrics['replicas']
        
        # ========== CHECK COOLDOWN ==========
        if self._in_cooldown(fn_name):
            return {'action': 'none', 'count': 0, 'reason': 'cooldown'}
        
        # ========== COMPONENTE PROATTIVA (se abilitata) ==========
        if self.config.enable_proactive:
            proactive_decision = self._evaluate_proactive(fn_name, metrics, deployment)
            if proactive_decision['action'] != 'none':
                return proactive_decision
        
        # ========== COMPONENTE REATTIVA ==========
        
        # 🔼 SCALE UP: Latenza alta
        if metrics['p95_latency_ms'] > self.config.latency_p95_threshold_ms:
            # Scala aggressivamente se latenza molto alta
            if metrics['p95_latency_ms'] > self.config.latency_p95_threshold_ms * 1.5:
                count = 2
                reason = f"P95={metrics['p95_latency_ms']:.1f}ms >> threshold (critical)"
            else:
                count = 1
                reason = f"P95={metrics['p95_latency_ms']:.1f}ms > threshold"
            
            required = min(current_replicas + count, deployment.scaling_config.scale_max)
            return {
                'action': 'scale_up',
                'count': required - current_replicas,
                'reason': reason
            }
        
        # 🔼 SCALE UP: RPS troppo alto per replica
        target_rps = self.config.rps_per_replica_target
        if metrics['rps_per_replica'] > target_rps * 1.5:
            # Calcola repliche necessarie
            required = int((metrics['rps'] / target_rps) + 0.99)
            required = min(required, deployment.scaling_config.scale_max)
            required = max(required, deployment.scaling_config.scale_min)
            
            if required > current_replicas:
                return {
                    'action': 'scale_up',
                    'count': required - current_replicas,
                    'reason': f"RPS/replica={metrics['rps_per_replica']:.1f} > target"
                }
        
        # 🔼 SCALE UP: Code lunghe
        if metrics['avg_queue_time'] > 2.0:  # >2s di attesa in coda
            return {
                'action': 'scale_up',
                'count': 1,
                'reason': f"High queue time={metrics['avg_queue_time']:.2f}s"
            }
        
        # 🔽 SCALE DOWN: Sotto-utilizzo prolungato
        # Richiede conferma su multiple osservazioni
        if self._should_scale_down(fn_name, metrics, deployment):
            return {
                'action': 'scale_down',
                'count': 1,
                'reason': f"Low utilization: P95={metrics['p95_latency_ms']:.1f}ms, RPS/rep={metrics['rps_per_replica']:.1f}"
            }
        
        return {'action': 'none', 'count': 0, 'reason': 'stable'}
    
    def _evaluate_proactive(self, fn_name: str, metrics: Dict, deployment) -> Dict:
        """
        Valuta decisioni proattive basate su ML predictions
        (Placeholder per futuro ML integration)
        """
        # Questo è un placeholder per future estensioni con ML
        # Nel sistema del collega, qui verrebbe letto il file JSON con predizioni
        
        # Esempio di come potrebbe funzionare:
        # prediction = self.ml_predictions.get(fn_name, {})
        # confidence = prediction.get('spike_probability', 0.0)
        #
        # if confidence > self.config.confidence_threshold_urgent:
        #     return {'action': 'scale_up', 'count': 2, 'reason': f'ML: spike predicted (conf={confidence:.2f})'}
        # elif confidence > self.config.confidence_threshold_elevated:
        #     return {'action': 'scale_up', 'count': 1, 'reason': f'ML: elevated risk (conf={confidence:.2f})'}
        
        return {'action': 'none', 'count': 0, 'reason': 'proactive: no action'}
    
    def _should_scale_down(self, fn_name: str, metrics: Dict, deployment) -> bool:
        """
        Decide se scalare in basso basandosi su trend sostenuto
        """
        # Condizioni per scale-down:
        # 1. Latenza bassa
        # 2. RPS basso
        # 3. Nessuna coda
        # 4. Trend confermato su multiple osservazioni
        
        if metrics['replicas'] <= deployment.scaling_config.scale_min:
            return False
        
        if metrics['p95_latency_ms'] > self.config.latency_p95_scale_down_ms:
            return False
        
        if metrics['avg_queue_time'] > 0.1:
            return False
        
        target_rps = self.config.rps_per_replica_target
        if metrics['rps_per_replica'] > target_rps * 0.3:  # Almeno 30% utilizzo
            return False
        
        # Controlla cooldown specifico per scale-down
        if fn_name in self.last_scale_time:
            time_since = self.env.now - self.last_scale_time[fn_name]
            if time_since < self.config.scale_down_cooldown:
                return False
        
        # Conferma trend su ultime N osservazioni
        history = self.observation_history.get(fn_name, [])
        if len(history) < self.config.min_observations_for_scale_down:
            return False
        
        recent = history[-self.config.min_observations_for_scale_down:]
        all_low = all(
            obs['p95_latency_ms'] < self.config.latency_p95_scale_down_ms and
            obs['rps_per_replica'] < target_rps * 0.3
            for obs in recent
        )
        
        return all_low
    
    def _count_function_invocations(self, fn_name: str) -> int:
        """Conta invocazioni totali per funzione"""
        try:
            metrics_df = self.env.metrics.extract_dataframe('invocations')
            if metrics_df is None or len(metrics_df) == 0:
                return 0
            fn_invocations = metrics_df[metrics_df['function_name'] == fn_name]
            return len(fn_invocations)
        except:
            return 0
    
    def _get_p95_latency(self, fn_name: str) -> float:
        """Calcola P95 latency dalle metriche recenti"""
        try:
            metrics_df = self.env.metrics.extract_dataframe('invocations')
            if metrics_df is None or len(metrics_df) == 0:
                return 0.0
            
            fn_df = metrics_df[metrics_df['function_name'] == fn_name]
            if len(fn_df) == 0:
                return 0.0
            
            # Ultimi 100 sample
            window = min(100, len(fn_df))
            recent = fn_df.tail(window)
            
            # Response time = t_wait + t_exec (se disponibili)
            if 't_wait' in recent.columns and 't_exec' in recent.columns:
                response_times = recent['t_wait'] + recent['t_exec']
            elif 't_exec' in recent.columns:
                response_times = recent['t_exec']
            else:
                return 0.0
            
            p95 = response_times.quantile(0.95)
            return float(p95)
        except Exception as e:
            logger.debug(f"Error calculating P95: {e}")
            return 0.0
    
    def _get_avg_queue_time(self, fn_name: str) -> float:
        """Calcola tempo medio in coda"""
        try:
            metrics_df = self.env.metrics.extract_dataframe('invocations')
            if metrics_df is None or len(metrics_df) == 0:
                return 0.0
            
            fn_df = metrics_df[metrics_df['function_name'] == fn_name]
            if len(fn_df) == 0:
                return 0.0
            
            window = min(100, len(fn_df))
            recent = fn_df.tail(window)
            
            if 't_wait' in recent.columns:
                return float(recent['t_wait'].mean())
            return 0.0
        except:
            return 0.0
    
    def _in_cooldown(self, fn_name: str) -> bool:
        """Check se in cooldown"""
        if fn_name not in self.last_scale_time:
            return False
        time_since = self.env.now - self.last_scale_time[fn_name]
        return time_since < self.config.cooldown_period
    
    def _scale_up(self, fn_name: str, count: int, deployment, reason: str):
        """Scala in alto"""
        if count <= 0:
            return
        
        logger.info(
            f"🔼 [HYBRID] Scaling UP {fn_name}: +{count} replicas "
            f"(t={self.env.now:.1f}s) - {reason}"
        )
        
        faas = self.env.faas
        yield from faas.scale_up(fn_name, count)
        self.last_scale_time[fn_name] = self.env.now
    
    def _scale_down(self, fn_name: str, count: int, deployment, reason: str):
        """Scala in basso"""
        if count <= 0:
            return
        
        logger.info(
            f"🔽 [HYBRID] Scaling DOWN {fn_name}: -{count} replicas "
            f"(t={self.env.now:.1f}s) - {reason}"
        )
        
        faas = self.env.faas
        yield from faas.scale_down(fn_name, count)
        self.last_scale_time[fn_name] = self.env.now
    
    def set_ml_prediction(self, fn_name: str, prediction: Dict):
        """
        Imposta predizione ML per una funzione
        (Per futuro uso con modelli ML)
        
        Args:
            fn_name: Nome funzione
            prediction: Dict con 'spike_probability', 'confidence', etc.
        """
        self.ml_predictions[fn_name] = prediction