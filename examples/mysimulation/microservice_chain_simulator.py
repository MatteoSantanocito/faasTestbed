"""
Microservice Chain Simulator
Gestisce invocazioni a catena tra microservizi con dipendenze
Es: nginx → home-timeline → post-storage + social-graph
"""

import logging
from sim.core import Environment
from sim.faas import FunctionSimulator, FunctionReplica, FunctionRequest, FunctionResponse
from sim.faas.watchdogs import HTTPWatchdog
from social_network_microservices import SocialNetworkMicroservices

logger = logging.getLogger(__name__)


class MicroserviceChainSimulator(FunctionSimulator):
    """
    Simulator che gestisce catene di invocazioni tra microservizi
    
    Quando nginx-thrift riceve una richiesta per /home-timeline:
    1. nginx processa (5ms)
    2. Chiama home-timeline (50ms)
    3. home-timeline chiama post-storage (30ms) + social-graph (35ms) in parallelo
    4. Risponde al client
    
    Total time: 5 + 50 + max(30, 35) = 90ms (+ network + queuing)
    """
    
    def __init__(self, 
                 service_name: str,
                 workers: int,
                 base_execution_time: float,
                 degradation_factor: float = 0.1):
        """
        Args:
            service_name: Nome del microservizio
            workers: Numero worker concorrenti
            base_execution_time: Tempo esecuzione base (ms)
            degradation_factor: Fattore degradazione (0-1)
        """
        super().__init__()
        self.service_name = service_name
        self.base_execution_time = base_execution_time / 1000.0  # ms → seconds
        self.degradation_factor = degradation_factor
        self.workers = workers
        self.queue = None  # Inizializzato in setup()
        
        # Ottieni dipendenze dal catalogo
        self.dependencies = SocialNetworkMicroservices.get_dependencies(service_name)
    
    def deploy(self, env: Environment, replica: FunctionReplica):
        """Deploy phase: image pull"""
        # Simula docker pull (se immagine non in cache)
        import sim.docker as docker
        yield from docker.pull(env, replica.container.image, replica.node.ether_node)

    def startup(self, env: Environment, replica: FunctionReplica):
        """Startup phase: container initialization"""
        # Tempo realistico: 3-8 secondi (misurato nel testbed)
        import random
        startup_delay = random.uniform(3.0, 8.0)
        
        logger.info(
            f"[t={env.now:.1f}s] Starting {self.service_name} "
            f"on {replica.node.name} (startup_delay={startup_delay:.1f}s)"
        )
        
        yield env.timeout(startup_delay)

    def setup(self, env: Environment, replica: FunctionReplica):
        """Setup phase: runtime initialization"""
        import simpy
        self.queue = simpy.Resource(env, capacity=self.workers)
        
        # Runtime setup (più veloce: 0.5-2s)
        import random
        setup_delay = random.uniform(0.5, 2.0)
        
        logger.info(
            f"[t={env.now:.1f}s] Setup {self.service_name}: "
            f"{self.workers} workers (setup_delay={setup_delay:.1f}s)"
        )
        
        yield env.timeout(setup_delay)
    
    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        """
        Esegui funzione con invocazioni a catena verso dipendenze
        """
        # ========== 1. ATTESA IN CODA ==========
        worker_request = self.queue.request()
        t_queue_start = env.now
        
        yield worker_request
        
        t_wait = env.now - t_queue_start
        if not hasattr(request, 't_wait'):
            request.t_wait = t_wait
        
        # ========== 2. CALCOLA TEMPO ESECUZIONE ==========
        exec_time = self._calculate_execution_time(env, replica)
        
        if not hasattr(request, 't_exec'):
            request.t_exec = exec_time
        
        # ========== 3. CONSUMO RISORSE ==========
        env.resource_state.put_resource(replica, 'cpu', 0.25)
        env.resource_state.put_resource(replica, 'memory', 128 * 1024 * 1024)
        
        # ========== 4. ESECUZIONE ==========
        yield env.timeout(exec_time)
        
        
        
        # ========== 5. INVOCAZIONI DIPENDENZE ==========
        if self.dependencies:
            yield from self._invoke_dependencies(env, request)
            
        # ========== 6. RILASCIA WORKER ==========
        self.queue.release(worker_request)
        
        # ========== 7. RILASCIA RISORSE ==========
        env.resource_state.remove_resource(replica, 'cpu', 0.25)
        env.resource_state.remove_resource(replica, 'memory', 128 * 1024 * 1024)
    
    def _calculate_execution_time(self, env: Environment, replica: FunctionReplica) -> float:
        """
        Calcola tempo esecuzione con degradazione
        
        Degradazione lineare basata su concurrent requests:
        exec_time = base_time * (1 + degradation_factor * concurrent_requests / workers)
        """
        # Degradazione basata su richieste concorrenti nella coda
        concurrent = self.queue.count if self.queue else 0
        
        # Degradazione lineare
        degradation = 1.0 + (self.degradation_factor * concurrent / self.workers)
        exec_time = self.base_execution_time * degradation
        
        return exec_time
    
    def _invoke_dependencies(self, env: Environment, parent_request: FunctionRequest):
        """
        Invoca servizi dipendenti
        
        Strategia:
        - Invocazioni in PARALLELO (simula async calls)
        - Attendi che TUTTE completino (simula Promise.all())
        """
        if not self.dependencies:
            return
        
        # Crea richieste per ogni dipendenza
        dep_processes = []
        
        for dep_service in self.dependencies:
            # Crea sub-request
            dep_request = FunctionRequest(dep_service)
            
            # Invoca in parallelo
            dep_process = env.process(env.faas.invoke(dep_request))
            dep_processes.append(dep_process)
        
        # Attendi completamento di tutte le dipendenze
        for process in dep_processes:
            yield process


class MicroserviceSimulatorFactory:
    """
    Factory per creare simulator specifici per ogni microservizio
    """
    
    def __init__(self):
        self.services = SocialNetworkMicroservices.SERVICES
    
    def create(self, env: Environment, fn_container) -> FunctionSimulator:
        """
        Crea simulator appropriato per il microservizio
        
        Args:
            env: Simulation environment
            fn_container: FunctionContainer (not replica!)
            
        Returns:
            Configured MicroserviceChainSimulator
        """
        # Extract service name from container image
        service_name = fn_container.image
        
        # Ottieni spec dal catalogo
        spec = self.services.get(service_name)
        
        if not spec:
            logger.warning(f"No spec found for service '{service_name}', using defaults")
            return MicroserviceChainSimulator(
                service_name=service_name,
                workers=4,
                base_execution_time=50.0,
                degradation_factor=0.15
            )
        
        # Crea simulator configurato
        simulator = MicroserviceChainSimulator(
            service_name=spec.name,
            workers=spec.workers,
            base_execution_time=spec.execution_time_ms,
            degradation_factor=spec.degradation_factor
        )
        
        logger.debug(
            f"Created simulator for '{service_name}': "
            f"workers={spec.workers}, exec={spec.execution_time_ms}ms, "
            f"deps={spec.dependencies}"
        )
        
        return simulator


# Test standalone
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    print("\n" + "="*70)
    print("MICROSERVICE CHAIN SIMULATOR TEST")
    print("="*70)
    
    factory = MicroserviceSimulatorFactory()
    
    print("\n📦 Simulators per servizio:")
    for service_name in SocialNetworkMicroservices.SERVICES.keys():
        spec = SocialNetworkMicroservices.get_service(service_name)
        print(f"\n  {service_name}:")
        print(f"    Workers: {spec.workers}")
        print(f"    Base exec: {spec.execution_time_ms}ms")
        print(f"    Degradation: {spec.degradation_factor*100:.0f}%")
        print(f"    Dependencies: {spec.dependencies or 'none (leaf)'}")
    
    print("\n" + "="*70)