# examples/mysimulation/benchmark_realistic.py

import logging
from typing import List

from sim.core import Environment
from sim.benchmark import Benchmark
from sim.docker import ImageProperties
from sim.faas import (
    FunctionDeployment, Function, FunctionImage, 
    FunctionRequest, ScalingConfiguration, FunctionContainer
)
from sim.requestgen import (
    function_trigger,
    expovariate_arrival_profile,
    constant_rps_profile
)
from skippy.core.utils import parse_size_string

logger = logging.getLogger(__name__)

class RealisticWorkloadBenchmark(Benchmark):
    """
    Benchmark con workload realistici e autoscaling
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'RealisticWorkload'
    
    def setup(self, env: Environment):
        """Setup: Registra immagini funzioni"""
        logger.info("Setting up benchmark...")
        
        containers = env.container_registry
        
        # Python Pi CPU (multi-arch) - DEVE essere disponibile per tutte le arch
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='arm32'))
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='aarch64'))
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='x86'))
        
        # ResNet50 CPU (multi-arch) - DEVE essere disponibile per tutte le arch
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('200M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('200M'), arch='aarch64'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('200M'), arch='x86'))
        
        # ResNet50 GPU (solo aarch64 per TX2 con GPU)
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('250M'), arch='aarch64'))
        
        logger.info("✓ Registered all function images")
    
    def run(self, env: Environment):
        """Run: Workload realistico con multiple fasi"""
        faas = env.faas
        
        logger.info("\n" + "="*70)
        logger.info("DEPLOYING FUNCTIONS")
        logger.info("="*70)
        
        # Prepara deployments
        deployments = self.prepare_deployments()
        
        # Deploy tutte le funzioni
        for deployment in deployments:
            logger.info(f"Deploying {deployment.fn.name}...")
            yield from faas.deploy(deployment)
        
        # Aspetta repliche disponibili
        logger.info("Waiting for replicas...")
        yield env.process(faas.poll_available_replica('python-pi'))
        yield env.process(faas.poll_available_replica('resnet50-cpu'))
        
        logger.info("\n" + "="*70)
        logger.info("STARTING REALISTIC WORKLOAD SIMULATION")
        logger.info("="*70)
        
        # ========== FASE 1: WARM-UP (Carico Basso) ==========
        logger.info("\n[PHASE 1] Warm-up - Low load (10 RPS, 30s)")
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=10))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=300)
        
        logger.info(f"  → Completed at t={env.now:.1f}s")
        logger.info(f"  → Expected: scale_min=1 (no scale-up)")
        
        # ========== FASE 2: RAMP-UP (Step crescenti) ==========
        logger.info("\n[PHASE 2] Ramp-up - Step increase (10→20→40 RPS, 60s)")
        
        # Step 1: 10 RPS
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=10))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=200)
        
        # Step 2: 20 RPS
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=20))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=400)
        
        # Step 3: 40 RPS  
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=40))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=800)
        
        logger.info(f"  → Completed at t={env.now:.1f}s")
        logger.info(f"  → Expected: scale-up to 4-6 replicas")
        
        # ========== FASE 3: PEAK TRAFFIC (Picco) ==========
        logger.info("\n[PHASE 3] Peak traffic - High load (100 RPS, 20s)")
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=100))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=2000)
        
        logger.info(f"  → Completed at t={env.now:.1f}s")
        logger.info(f"  → Expected: scale-up to scale_max=8")
        
        # ========== FASE 4: COOLDOWN (Step decrescenti) ==========
        logger.info("\n[PHASE 4] Cooldown - Step decrease (100→50→20 RPS, 60s)")
        
        # Step 1: 50 RPS
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=50))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=1000)
        
        # Step 2: 20 RPS
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=20))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=400)
        
        # Step 3: 10 RPS
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=10))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=200)
        
        logger.info(f"  → Completed at t={env.now:.1f}s")
        logger.info(f"  → Expected: scale-down to 2-3 replicas")
        
        # ========== FASE 5: IDLE (Quasi Zero Traffico) ==========
        logger.info("\n[PHASE 5] Idle - Minimal load (2 RPS, 60s)")
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=2))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=120)
        
        logger.info(f"  → Completed at t={env.now:.1f}s")
        logger.info(f"  → Expected: scale-down to scale_min=1")
        
        # ========== FASE 6: BURST TEST ==========
        logger.info("\n[PHASE 6] Burst test - Sudden spike (200 RPS, 10s)")
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=200))
        yield from function_trigger(env, deployments[0], ia_gen, max_requests=2000)
        
        logger.info(f"  → Completed at t={env.now:.1f}s")
        logger.info(f"  → Expected: rapid scale-up to scale_max=8")
        
        # ========== FASE 7: RESNET50 WORKLOAD ==========
        logger.info("\n[PHASE 7] ResNet50 workload - ML inference (20 RPS, 30s)")
        ia_gen = expovariate_arrival_profile(constant_rps_profile(rps=20))
        yield from function_trigger(env, deployments[1], ia_gen, max_requests=600)
        
        logger.info(f"  → Completed at t={env.now:.1f}s")
        
        logger.info("\n" + "="*70)
        logger.info("WORKLOAD SIMULATION COMPLETED")
        logger.info("="*70)
        logger.info(f"Total simulation time: {env.now:.1f}s")
        logger.info(f"Total requests generated: ~7000+")
    
    def prepare_deployments(self) -> List[FunctionDeployment]:
        """Prepara deployments con autoscaling"""
        python_pi_fd = self.prepare_python_pi_deployment()
        resnet50_cpu_fd = self.prepare_resnet50_cpu_deployment()
        resnet50_gpu_fd = self.prepare_resnet50_gpu_deployment()
        
        return [python_pi_fd, resnet50_cpu_fd, resnet50_gpu_fd]

    def prepare_python_pi_deployment(self) -> FunctionDeployment:
        """Python Pi con autoscaling aggressivo"""
        python_pi_image = FunctionImage(image='python-pi-cpu')  # ← SENZA arch!
        python_pi_fn = Function('python-pi', fn_images=[python_pi_image])
        python_pi_container = FunctionContainer(python_pi_image)
        
        scaling_config = ScalingConfiguration()
        scaling_config.scale_min = 1
        scaling_config.scale_max = 8
        
        python_pi_fd = FunctionDeployment(
            python_pi_fn,
            [python_pi_container],
            scaling_config
        )
        
        return python_pi_fd

    def prepare_resnet50_cpu_deployment(self) -> FunctionDeployment:
        """ResNet50 CPU con autoscaling moderato"""
        resnet50_cpu_image = FunctionImage(image='resnet50-inference-cpu')
        resnet50_cpu_fn = Function('resnet50-cpu', fn_images=[resnet50_cpu_image])
        resnet50_cpu_container = FunctionContainer(resnet50_cpu_image)
        
        scaling_config = ScalingConfiguration()
        scaling_config.scale_min = 1
        scaling_config.scale_max = 5
        
        resnet50_cpu_fd = FunctionDeployment(
            resnet50_cpu_fn,
            [resnet50_cpu_container],
            scaling_config
        )
        
        return resnet50_cpu_fd

    def prepare_resnet50_gpu_deployment(self) -> FunctionDeployment:
        """ResNet50 GPU con autoscaling limitato"""
        resnet50_gpu_image = FunctionImage(image='resnet50-inference-gpu')
        resnet50_gpu_fn = Function('resnet50-gpu', fn_images=[resnet50_gpu_image])
        resnet50_gpu_container = FunctionContainer(resnet50_gpu_image)
        
        scaling_config = ScalingConfiguration()
        scaling_config.scale_min = 1
        scaling_config.scale_max = 2
        
        resnet50_gpu_fd = FunctionDeployment(
            resnet50_gpu_fn,
            [resnet50_gpu_container],
            scaling_config
        )
        
        # ✅ GPU labels
        resnet50_gpu_fd.labels = {
            'capability.skippy.io/nvidia-gpu': '',
            'capability.skippy.io/nvidia-cuda': '10'
        }
        
        return resnet50_gpu_fd