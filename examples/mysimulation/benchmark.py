# examples/mysimulation/benchmark.py

import logging
from typing import List

from sim.core import Environment
from sim.benchmark import Benchmark
from sim.docker import ImageProperties
from sim.faas import FunctionDeployment, Function, FunctionImage, FunctionRequest, ScalingConfiguration, FunctionContainer
from skippy.core.utils import parse_size_string
from sim.requestgen import (
    function_trigger,
    expovariate_arrival_profile,
    constant_rps_profile
)

logger = logging.getLogger(__name__)



class MySimulationBenchmark(Benchmark):
    """
    Benchmark personalizzato seguendo l'esempio request_gen
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'MySimulation'
    
    def setup(self, env: Environment):
        """
        Setup: Registra immagini nel container registry
        """
        logger.info("Setting up benchmark...")
        
        containers = env.container_registry
        
        # put() prende UN SOLO ImageProperties alla volta
        
        # Python Pi CPU (multi-arch)
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='arm32'))
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='x86'))
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='aarch64'))
        
        # ResNet50 CPU (multi-arch)
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('200M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('200M'), arch='x86'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('200M'), arch='aarch64'))
        
        # ResNet50 GPU (multi-arch) 
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('250M'), arch='aarch64'))
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('250M'), arch='x86'))     
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('250M'), arch='arm32'))   
    
        logger.info("✓ Registered all function images")
        
        
        
        # Log tutte le immagini registrate
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('  %s:%s -> %s', name, tag, images)
    
    def run(self, env: Environment):
        """
        Run: Deploy funzioni e genera workload
        """
        faas = env.faas
        
        logger.info("\n" + "="*60)
        logger.info("DEPLOYING FUNCTIONS")
        logger.info("="*60)
        
        # Prepara deployments
        deployments = self.prepare_deployments()
        
        # Deploy tutte le funzioni
        for deployment in deployments:
            logger.info(f"Deploying {deployment.fn.name}...")
            yield from faas.deploy(deployment)
        
        # Aspetta che le repliche siano disponibili
        logger.info("Waiting for replicas to be available...")
        yield env.process(faas.poll_available_replica('python-pi'))
        
        logger.info("\n" + "="*60)
        logger.info("STARTING WORKLOAD")
        logger.info("="*60)
        
        # ========== GENERA WORKLOAD ==========
        
        # PHASE 1: Python Pi (20 richieste)
        logger.info("\n[PHASE 1] Python Pi - 20 requests")
        for i in range(20):
            if i % 5 == 0:
                logger.info(f"  → Request {i+1}/20")
            faas.invoke(FunctionRequest('python-pi'))
            yield env.timeout(0.5) #QUESTA VERSIONE è troppo semplice 
            # dovrei realizzare degli arrivi Poissoniani (Più Realistico) e faas-sim ha generatori di workload built-in!
        
        # PHASE 2: ResNet50 CPU (10 richieste)
        logger.info("\n[PHASE 2] ResNet50 CPU - 10 requests")
        for i in range(10):
            if i % 3 == 0:
                logger.info(f"  → Request {i+1}/10")
            faas.invoke(FunctionRequest('resnet50-cpu'))
            yield env.timeout(1)
        
        # PHASE 3: ResNet50 GPU (5 richieste)
        logger.info("\n[PHASE 3] ResNet50 GPU - 5 requests")
        for i in range(5):
            logger.info(f"  → Request {i+1}/5")
            faas.invoke(FunctionRequest('resnet50-gpu'))
            yield env.timeout(2)
        
        logger.info("\n" + "="*60)
        logger.info("WORKLOAD COMPLETED")
        logger.info("="*60)
    
    def prepare_deployments(self) -> List[FunctionDeployment]:
        """
        Prepara tutti i FunctionDeployment
        """
        python_pi_fd = self.prepare_python_pi_deployment()
        resnet50_cpu_fd = self.prepare_resnet50_cpu_deployment()
        resnet50_gpu_fd = self.prepare_resnet50_gpu_deployment()
        
        return [python_pi_fd, resnet50_cpu_fd, resnet50_gpu_fd]
    
    def prepare_python_pi_deployment(self) -> FunctionDeployment:
        """
        Prepara deployment per Python Pi
        """
        # DESIGN TIME: Definisci la funzione
        python_pi_image = FunctionImage(image='python-pi-cpu')
        python_pi_fn = Function('python-pi', fn_images=[python_pi_image])
        
        # RUN TIME: Crea container e deployment
        python_pi_container = FunctionContainer(python_pi_image)
        
        scaling_config = ScalingConfiguration()
        scaling_config.scale_min = 1
        scaling_config.scale_max = 8
        
        python_pi_fd = FunctionDeployment(
            python_pi_fn,              # Function object
            [python_pi_container],     # Lista di FunctionContainer
            scaling_config             # ScalingConfiguration
        )
        
        return python_pi_fd
    
    def prepare_resnet50_cpu_deployment(self) -> FunctionDeployment:
        """
        Prepara deployment per ResNet50 CPU
        """
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
        """
        Prepara deployment per ResNet50 GPU
        """
        resnet50_gpu_image = FunctionImage(image='resnet50-inference-gpu')
        resnet50_gpu_fn = Function('resnet50-gpu', fn_images=[resnet50_gpu_image])
        
        resnet50_gpu_container = FunctionContainer(resnet50_gpu_image)
        
        scaling_config = ScalingConfiguration()
        scaling_config.scale_min = 1
        scaling_config.scale_max = 3
        
        resnet50_gpu_fd = FunctionDeployment(
            resnet50_gpu_fn,
            [resnet50_gpu_container],
            scaling_config
        )
        
        # Aggiungi label GPU
        resnet50_gpu_fd.labels = {'capability.skippy.io/gpu': ''}
        
        return resnet50_gpu_fd