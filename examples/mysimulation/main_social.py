"""
Social Network Testbed Replica - Main Simulation
Replica ESATTA del testbed reale con DeathStarBench
"""

import logging
import argparse
import sys
import os
import random

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.core import Environment
from sim.faassim import Simulation
from sim.benchmark import Benchmark
from sim.docker import ImageProperties
from sim.faas import (
    FunctionDeployment, Function, FunctionImage, 
    FunctionContainer, ScalingConfiguration, FunctionRequest,
    KubernetesResourceConfiguration
)

from collections import Counter

from topology_testbed_single import SingleNodeTopology
from social_network_microservices import SocialNetworkMicroservices
from workload_k6_replica import K6WorkloadGenerator
from microservice_chain_simulator import MicroserviceSimulatorFactory
from autoscaler_hybridnew import HybridAutoscaler, AutoscalerConfig

logger = logging.getLogger(__name__)


class SocialNetworkBenchmark(Benchmark):
    """
    Benchmark che replica esattamente il testbed
    """
    
    def __init__(self, config):
        self.config = config
        self.workload_gen = None
        self.requests = None
        
    def setup(self, env: Environment):
        """Setup: registra immagini e genera workload"""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK SETUP")
        logger.info("="*70)
        
        # ========== REGISTRA IMMAGINI ==========
        logger.info("\nRegistering container images...")
        containers = env.container_registry
        
        for spec in SocialNetworkMicroservices.get_all_services():
            # Ogni servizio ha un'immagine amd64
            image_size = 100 * 1024 * 1024  # 100MB base
            
            containers.put(ImageProperties(
                spec.name,
                image_size,
                arch='amd64'
            ))
            
            logger.info(f"  ✓ {spec.name}")
        
        logger.info(f"\n✅ Registered {len(SocialNetworkMicroservices.SERVICES)} microservices")
        
        # ========== GENERA WORKLOAD ==========
        logger.info("\nGenerating workload pattern...")
        
        self.workload_gen = K6WorkloadGenerator(
            rps_max=self.config.rps_max,
            duration_minutes=self.config.duration_minutes,
            minutes_per_simulated_day=self.config.minutes_per_simulated_day,
            random_seed=self.config.seed,
            workload_scale=getattr(self.config, 'workload_scale', 1.0)
        )
        
        # Genera richieste
        self.requests = self.workload_gen.generate_request_times(
            start_time=0.0,
            end_time=self.config.duration_minutes * 60
        )
        
        logger.info(f"✅ Generated {len(self.requests)} requests")
        logger.info("="*70 + "\n")
    
    def run(self, env: Environment):
        """Run: deploy microservizi e genera richieste"""
        
        # ========== DEPLOY MICROSERVIZI ==========
        logger.info("="*70)
        logger.info("DEPLOYING MICROSERVICES")
        logger.info("="*70 + "\n")
        
        deployments = self._prepare_deployments(env)
        
        for deployment in deployments:
            yield from env.faas.deploy(deployment)
            logger.info(f"  ✓ Deployed {deployment.fn.name}")
        
        # ========== WAIT FOR NGINX ==========
        logger.info("\n⏳ Waiting for nginx-thrift to be ready...")
        yield env.process(env.faas.poll_available_replica('nginx-thrift'))
        logger.info("  ✓ nginx-thrift ready\n")
        
        # ========== WARM-UP PHASE ==========
        logger.info("🔥 STARTING WARM-UP (30s @ 50 RPS)...")
        
        # STEP 1: Aspetta che TUTTI i servizi abbiano almeno 1 replica
        logger.info("   Step 1: Waiting for all services to be ready...")
        all_services = [
            'nginx-thrift', 'home-timeline', 'user-timeline', 'compose-post',
            'post-storage', 'user-service', 'social-graph', 
            'text-service', 'media-service', 'unique-id-service',  
            'user-mention-service', 'url-shorten-service', 'write-home-timeline'  
        ]
        
        for service_name in all_services:
            yield env.process(env.faas.poll_available_replica(service_name))
            logger.info(f"     ✓ {service_name} ready")
        
        logger.info("   Step 2: All services ready! Starting warm-up traffic...\n")
        
        # STEP 2: Warm-up PROGRESSIVO (come nel testbed reale)
        warmup_counter = 0
        warmup_start = env.now

        logger.info("   [0-60s] Progressive warm-up (50→150 RPS)...")

        # Fase 1: Warm-up gentile (30s @ 50 RPS)
        phase1_end = warmup_start + 30
        while env.now < phase1_end:
            request = FunctionRequest('nginx-thrift')
            env.process(env.faas.invoke(request))
            warmup_counter += 1
            yield env.timeout(1.0 / 50.0)

        # Fase 2: Ramp-up (30s @ 150 RPS) - simula inizio traffico reale
        phase2_end = warmup_start + 60
        while env.now < phase2_end:
            request = FunctionRequest('nginx-thrift')
            env.process(env.faas.invoke(request))
            warmup_counter += 1
            yield env.timeout(1.0 / 150.0)  # 150 RPS

        logger.info(f"✅ Warm-up completed ({warmup_counter} requests)")
        logger.info(f"   System pre-scaled to handle {150} RPS\n")
                
        # ========== GENERATE REQUESTS ==========
        logger.info("="*70)
        logger.info(f"STARTING WORKLOAD GENERATION ({len(self.requests)} requests)")
        logger.info("="*70 + "\n")
        
        request_count = 0
        last_log_time = env.now
        last_request_count = 0
        
        # ⬅️ NEW: Contatore per le richieste in questo intervallo
        interval_type_counts = Counter()
        
        for timestamp, endpoint in self.requests:
            # Wait until timestamp
            wait_time = timestamp - env.now
            if wait_time > 0:
                yield env.timeout(wait_time)
            
            # Get call chain & Invoke
            if endpoint == 'homepage':
                entry_service = 'nginx-thrift'
            else:
                entry_service = endpoint  # 'user-timeline', 'home-timeline', 'compose-post'
            request = FunctionRequest(entry_service)
            env.process(env.faas.invoke(request))
            
            request_count += 1
            
            # Conta il tipo di richiesta
            # Abbrevia i nomi per risparmiare spazio nel log
            short_name = {
                'homepage': 'home', 
                'user-timeline': 'u-tl', 
                'home-timeline': 'h-tl', 
                'compose-post': 'POST'  # Evidenziamo le scritture!
            }.get(endpoint, endpoint)
            
            interval_type_counts[short_name] += 1
            
            # Log progress every 60s CON MONITORING
            if env.now - last_log_time >= 60:
                # 1. Calcoli Metriche Traffico
                interval_duration = env.now - last_log_time
                requests_in_interval = request_count - last_request_count
                current_real_rps = requests_in_interval / interval_duration if interval_duration > 0 else 0
                
                progress = (env.now / (self.config.duration_minutes * 60)) * 100
                
                # 2. Calcolo Orario Simulato
                cycle_sec = self.config.minutes_per_simulated_day * 60
                day_progress = (env.now % cycle_sec) / cycle_sec
                sim_h = int(day_progress * 24)
                sim_m = int((day_progress * 24 * 60) % 60)
                sim_time_str = f"{sim_h:02d}:{sim_m:02d}"

                # 3. Calcolo P95
                current_p95 = self._calculate_current_p95(env)
                
                # 4. Conteggio Repliche Compatto
                deployments = env.faas.get_deployments()
                shortcuts = {
                    'nginx-thrift': 'ngx', 'compose-post': 'cmp', 
                    'post-storage': 'sto', 'home-timeline': 'hom', 'user-timeline': 'usr'
                }
                parts = []
                for dep in deployments:
                    name = dep.fn.name
                    if name in shortcuts:
                        count = len(env.faas.get_replicas(name))
                        parts.append(f"{shortcuts[name]}={count}")
                summary_replicas = ", ".join(parts)
                
                # ⬅️ NEW: Formatta il breakdown delle richieste
                # Es: "POST=430, home=1200..."
                req_breakdown = ", ".join([f"{k}={v}" for k, v in interval_type_counts.most_common()])
                
                # ⬇️ LOG COMPLETO CON BREAKDOWN
                logger.info(
                    f"  {sim_time_str} (t={env.now:>4.0f}s) | "
                    f"RPS:{current_real_rps:>3.0f} | "
                    f"P95:{current_p95:>4.0f}ms | "
                    f"Reqs:[{req_breakdown}] | "    # <--- QUI VEDI COSA ARRIVA
                    f"Pods:[{summary_replicas}]"
                )
                
                last_log_time = env.now
                last_request_count = request_count
                interval_type_counts.clear() # ⬅️ NEW: Resetta per il prossimo minuto
        
    def _calculate_current_p95(self, env: Environment) -> float:
        """Calcola P95 response time negli ultimi 60s"""
        try:
            inv_df = env.metrics.extract_dataframe('invocations')
            
            if inv_df is None or len(inv_df) == 0:
                return 0.0
            
            # Filtra ultimi 60s
            window_start = max(0, env.now - 60)
            recent = inv_df[inv_df['t_start'] >= window_start]
            
            if len(recent) == 0:
                return 0.0
            
            # Calcola response time
            if 't_wait' in recent.columns and 't_exec' in recent.columns:
                response_times = (recent['t_wait'] + recent['t_exec']) * 1000  # → ms
                return response_times.quantile(0.95)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error calculating P95: {e}")
            return 0.0
    
    def _prepare_deployments(self, env: Environment) -> list:
        """Prepara deployments per tutti i microservizi"""
        
        deployments = []
        
        for spec in SocialNetworkMicroservices.get_all_services():
            # Function image
            fn_image = FunctionImage(image=spec.name)
            
            # Function
            function = Function(
                name=spec.name,
                fn_images=[fn_image]
            )
            
            # Container con risorse MINIME (replica testbed reale!)
            # Testbed reale: ~64m CPU per microservizio leggero
            resource_config = KubernetesResourceConfiguration.create_from_str(
                cpu='64m',       # 64 milliCPU = 0.064 cores (come testbed!)
                memory='128Mi'   # 128 MB RAM
            )
            fn_container = FunctionContainer(
                fn_image=fn_image,
                resource_config=resource_config
            )
            
            # Scaling config
            scaling_config = ScalingConfiguration()
            scaling_config.scale_min = spec.scale_min
            scaling_config.scale_max = spec.scale_max
            scaling_config.scale_zero = False
            
            # Deployment
            deployment = FunctionDeployment(
                function,
                [fn_container],
                scaling_config
            )
            
            deployments.append(deployment)
        
        return deployments


class TestbedReplicaSimulation:
    """
    Simulazione completa che replica il testbed
    """
    
    def __init__(self,
                 duration_minutes=120,
                 rps_max=200,
                 enable_autoscaling=True,
                 minutes_per_simulated_day=60,
                 seed=42,
                 workload_scale=0.02):
        """
        Args:
            workload_scale: Scala workload per match testbed reale
                           Default 0.02 = 2% del max teorico (~17k requests)
        """
        
        self.duration_minutes = duration_minutes
        self.rps_max = rps_max
        self.enable_autoscaling = enable_autoscaling
        self.minutes_per_simulated_day = minutes_per_simulated_day
        self.seed = seed
        self.workload_scale = workload_scale
        
    def run(self):
        """Esegui simulazione completa"""
        
        # ========== BANNER ==========
        logger.info("\n" + "="*70)
        logger.info("SOCIAL NETWORK TESTBED REPLICA SIMULATION")
        logger.info("="*70)
        logger.info("\n📋 Configuration:")
        logger.info(f"  • Duration:       {self.duration_minutes}min ({self.duration_minutes*60}s)")
        logger.info(f"  • RPS max:        {self.rps_max}")
        logger.info(f"  • Autoscaling:    {'ENABLED' if self.enable_autoscaling else 'DISABLED'}")
        logger.info(f"  • Sim day:        {self.minutes_per_simulated_day}min real = 1 day sim")
        logger.info(f"  • Random seed:    {self.seed}")
        if self.workload_scale != 1.0:
            expected_requests = int(890000 * self.workload_scale)
            logger.info(f"  • Workload scale: {self.workload_scale:.2f} (~{expected_requests:,} requests)")
        logger.info("="*70 + "\n")
        
        # ========== TOPOLOGY ==========
        logger.info("📡 Creating topology...")
        topo_builder = SingleNodeTopology()
        topology = topo_builder.create()
        topo_builder.print_summary()
        
        # ========== BENCHMARK ==========
        benchmark = SocialNetworkBenchmark(self)
        
        # ========== SIMULATION ==========
        logger.info("🔧 Creating simulation...")
        sim = Simulation(topology, benchmark)
        
        # Register simulator factory
        sim.env.simulator_factory = MicroserviceSimulatorFactory()
        logger.info("  ✓ Microservice chain simulator registered")
        
        if self.enable_autoscaling:
            logger.info("\n🤖 Setting up autoscaler...")
            
            # 1. Crea la configurazione con i parametri ottimizzati
            scaler_config = AutoscalerConfig(
                check_interval=5.0,           # Controllo frequente (5s)
                cooldown_period=15.0,         # Scale UP veloce (15s)
                scale_down_cooldown=60.0,     # Scale DOWN lento (60s)
                latency_p95_scale_down_ms=45.0, # Soglia P95 per scale down
                rps_per_replica_target=40.0,  # Target 40 RPS (più realistico)
                enable_proactive=False        # Solo reactive per ora
            )
            
            # 2. Istanzia l'autoscaler passando il config
            autoscaler = HybridAutoscaler(
                sim.env,
                config=scaler_config
            )
            
            sim.env.process(autoscaler.run())
            logger.info("  ✓ Autoscaler registered as background process")
        
        # ========== RUN ==========
        logger.info("\n" + "="*70)
        logger.info("🚀 STARTING SIMULATION")
        logger.info("="*70 + "\n")
        
        sim.run()
        
        # ========== RESULTS ==========
        logger.info("\n" + "="*70)
        logger.info("✅ SIMULATION COMPLETED")
        logger.info("="*70)
        
        self._analyze_results(sim.env)
        
        return sim.env
    
    def _analyze_results(self, env):
        """Analizza e stampa risultati"""
        
        logger.info("\n" + "="*70)
        logger.info("📈 ANALYZING RESULTS")
        logger.info("="*70 + "\n")
        
        try:
            # Estrai metriche invocazioni
            inv_df = env.metrics.extract_dataframe('invocations')
            
            if inv_df is None or len(inv_df) == 0:
                logger.warning("No invocation data found!")
                return
            
            # Calcola response time
            if 't_wait' in inv_df.columns and 't_exec' in inv_df.columns:
                inv_df['response_time'] = inv_df['t_wait'] + inv_df['t_exec']
            else:
                logger.warning("Missing timing columns in invocations dataframe")
                return
            
            # Global stats
            total = len(inv_df)
            mean_rt = inv_df['response_time'].mean() * 1000  # seconds → ms
            p50_rt = inv_df['response_time'].quantile(0.50) * 1000
            p95_rt = inv_df['response_time'].quantile(0.95) * 1000
            p99_rt = inv_df['response_time'].quantile(0.99) * 1000
            
            mean_wait = inv_df['t_wait'].mean() * 1000 if 't_wait' in inv_df.columns else 0
            p95_wait = inv_df['t_wait'].quantile(0.95) * 1000 if 't_wait' in inv_df.columns else 0
            
            logger.info("📊 Global Statistics:")
            logger.info(f"  • Total invocations:  {total}")
            logger.info(f"  • Mean response time: {mean_rt:.2f}ms")
            logger.info(f"  • P50 response time:  {p50_rt:.2f}ms")
            logger.info(f"  • P95 response time:  {p95_rt:.2f}ms")
            logger.info(f"  • P99 response time:  {p99_rt:.2f}ms")
            logger.info(f"  • Mean queue time:    {mean_wait:.2f}ms")
            logger.info(f"  • P95 queue time:     {p95_wait:.2f}ms")
            
            # SLA compliance (P95 < 100ms)
            sla_threshold = 100  # ms
            violations = (inv_df['response_time'] * 1000 > sla_threshold).sum()
            compliance = (1 - violations / total) * 100
            
            logger.info(f"\n🎯 SLA Compliance (P95 < {sla_threshold}ms):")
            logger.info(f"  • Compliance rate:    {compliance:.2f}%")
            logger.info(f"  • Violations:         {violations}/{total}")
            
            # Per-function stats
            if 'function_name' in inv_df.columns:
                logger.info(f"\n📋 Per-Function Statistics:")
                
                for fn_name in sorted(inv_df['function_name'].unique()):
                    fn_df = inv_df[inv_df['function_name'] == fn_name]
                    fn_count = len(fn_df)
                    fn_p95 = fn_df['response_time'].quantile(0.95) * 1000
                    
                    logger.info(f"  • {fn_name:30s}: {fn_count:>6} invocations, P95={fn_p95:>6.2f}ms")
            
            # Export CSV
            output_dir = "/mnt/user-data/outputs"
            if os.path.exists(output_dir):
                logger.info(f"\n💾 Exporting metrics to {output_dir}...")
                csv_path = os.path.join(output_dir, "invocations_testbed_replica.csv")
                inv_df.to_csv(csv_path, index=False)
                logger.info(f"  ✓ Exported invocations: {total} records → {csv_path}")
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("\n" + "="*70)
        logger.info("🎉 ALL DONE!")
        logger.info("="*70 + "\n")


def main():
    """Entry point"""
    
    parser = argparse.ArgumentParser(
        description='Social Network Testbed Replica Simulation'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=120,
        help='Simulation duration in minutes (default: 120)'
    )
    parser.add_argument(
        '--rps-max',
        type=int,
        default=200,
        help='Maximum RPS (default: 200)'
    )
    parser.add_argument(
        '--no-autoscaling',
        action='store_true',
        help='Disable autoscaling (baseline with fixed replicas)'
    )
    parser.add_argument(
        '--sim-day-minutes',
        type=int,
        default=60,
        help='Real minutes per simulated day (default: 60)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--workload-scale',
        type=float,
        default=0.02,
        help='Workload scale factor (default: 0.02 = 2%% for ~17k requests)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logging.getLogger('sim.faas.system').setLevel(logging.WARNING)
    logging.getLogger('microservice_chain_simulator').setLevel(logging.WARNING)
    logging.getLogger('autoscaler_hybridnew').setLevel(logging.WARNING)
    logging.getLogger('sim.faassim').setLevel(logging.WARNING)
    logging.getLogger('topology_testbed_single').setLevel(logging.WARNING)
    logging.getLogger('workload_k6_replica').setLevel(logging.WARNING)
    
    # Run simulation
    sim = TestbedReplicaSimulation(
        duration_minutes=args.duration,
        rps_max=args.rps_max,
        enable_autoscaling=not args.no_autoscaling,
        minutes_per_simulated_day=args.sim_day_minutes,
        seed=args.seed,
        workload_scale=args.workload_scale
    )
    
    env = sim.run()
    
    return env


if __name__ == '__main__':
    main()