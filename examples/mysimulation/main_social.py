"""
Social Network Testbed Replica - Main Simulation
Replica ESATTA del testbed reale con DeathStarBench
"""

import logging
import argparse
import sys
import os

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

from topology_testbed_single import SingleNodeTopology
from social_network_microservices import SocialNetworkMicroservices
from workload_k6_replica import K6WorkloadGenerator
from microservice_chain_simulator import MicroserviceSimulatorFactory
from autoscaler_hybridnew import HybridAutoscaler

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
        
        # ========== GENERATE REQUESTS ==========
        logger.info("="*70)
        logger.info(f"STARTING WORKLOAD GENERATION ({len(self.requests)} requests)")
        logger.info("="*70 + "\n")
        
        request_count = 0
        last_log_time = 0
        
        for timestamp, endpoint in self.requests:
            # Wait until timestamp
            wait_time = timestamp - env.now
            if wait_time > 0:
                yield env.timeout(wait_time)
            
            # Get call chain for endpoint
            call_chain = SocialNetworkMicroservices.get_call_chain(endpoint)
            
            # Invoke entry point (nginx-thrift)
            entry_service = call_chain[0]
            request = FunctionRequest(entry_service)
            env.process(env.faas.invoke(request))
            
            request_count += 1
            
            # Log progress every 60s CON MONITORING
            if env.now - last_log_time >= 60:
                progress = (env.now / (self.config.duration_minutes * 60)) * 100
                
                # Calcola P95 corrente (ultimi 60s)
                current_p95 = self._calculate_current_p95(env)
                
                # Conta replicas nginx-thrift RUNNING
                try:
                    nginx_replicas = len([r for r in env.faas.get_replicas('nginx-thrift') 
                                         if hasattr(r, 'state') and r.state.name == 'RUNNING'])
                except:
                    nginx_replicas = 1  # Fallback
                
                logger.info(
                    f"  Progress: {progress:>5.1f}% | t={env.now:>6.1f}s | "
                    f"requests={request_count:>6} | P95={current_p95:>6.1f}ms | "
                    f"nginx={nginx_replicas} replicas"
                )
                last_log_time = env.now
        
        logger.info(f"\n✅ Request generation completed at t={env.now:.1f}s")
        logger.info(f"   Total requests sent: {request_count}")
    
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
        
        # Setup autoscaler (se enabled)
        if self.enable_autoscaling:
            logger.info("\n🤖 Setting up autoscaler...")
            autoscaler = HybridAutoscaler(
                sim.env,
                check_interval=10.0,  # Check ogni 10s
                cooldown_period=50.0,  # Cooldown 50s (come testbed)
                enable_proactive=False  # Solo reactive per ora
            )
            sim.env.process(autoscaler.run())
            logger.info("  ✓ Autoscaler registered as background process")
        else:
            logger.info("\n⚠️  Autoscaling DISABLED")
            logger.info("    Running with fixed replicas (scale_min)\n")
        
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