"""
Social Network Microservices (DeathStarBench)
Replica completa: https://github.com/delimitrou/DeathStarBench/tree/master/socialNetwork

8 microservizi con dipendenze reali e invocazioni a catena
"""

import logging
from typing import Dict, List, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MicroserviceSpec:
    """Specifica di un microservizio"""
    name: str
    dependencies: List[str]  # Servizi chiamati
    scale_min: int
    scale_max: int
    execution_time_ms: float  # Tempo base esecuzione
    workers: int  # Concurrent workers (HTTPWatchdog)
    degradation_factor: float  # Degradazione sotto carico
    description: str


class SocialNetworkMicroservices:
    """
    Microservizi Social Network DeathStarBench
    
    Architettura:
    - nginx-thrift: API Gateway (entry point)
    - home-timeline: Legge timeline aggregata
    - user-timeline: Legge timeline utente
    - compose-post: Crea nuovo post
    - post-storage: Storage posts (MongoDB)
    - user-service: Gestione utenti
    - social-graph: Grafo sociale (follower/following)
    - url-shorten: URL shortening
    """
    
    # Definizione completa microservizi
    SERVICES = {
        'nginx-thrift': MicroserviceSpec(
            name='nginx-thrift',
            dependencies=['home-timeline', 'user-timeline', 'compose-post'],
            scale_min=1, scale_max=10,   # ← Parte da 1
            execution_time_ms=2.0,       # ← Molto veloce
            workers=10,                  # Nginx regge bene
            degradation_factor=0.01,
            description='API Gateway'
        ),

        'home-timeline': MicroserviceSpec(
            name='home-timeline',
            dependencies=['post-storage', 'social-graph'],
            scale_min=1, scale_max=20,
            execution_time_ms=15.0,      # ← Ridotto da 35ms (Base latency più bassa)
            workers=2,                   # ← BOTTLE NECK! Solo 2 worker
            degradation_factor=0.10,
            description='Reads home timeline'
        ),

        'user-timeline': MicroserviceSpec(
            name='user-timeline',
            dependencies=['post-storage'],
            scale_min=1, scale_max=15,
            execution_time_ms=10.0,      # ← Ridotto
            workers=2,                   # ← Bottleneck
            degradation_factor=0.08,
            description='Reads user timeline'
        ),

        'compose-post': MicroserviceSpec(
            name='compose-post',
            dependencies=['post-storage', 'user-service', 'url-shorten'],
            scale_min=1, scale_max=15,
            execution_time_ms=25.0,      # ← Ridotto da 60ms. Base latency ~40ms totale
            workers=1,                   # ← CRITICO! 1 solo worker = si intasa subito
            degradation_factor=0.15,
            description='Creates new posts'
        ),

        'post-storage': MicroserviceSpec(
            name='post-storage',
            dependencies=[],
            scale_min=1, scale_max=12,
            execution_time_ms=5.0,       # Mongo veloce
            workers=4,
            degradation_factor=0.05,
            description='Post storage (MongoDB)'
        ),

        'user-service': MicroserviceSpec(
            name='user-service',
            dependencies=[],
            scale_min=1, scale_max=8,
            execution_time_ms=3.0,
            workers=4,
            degradation_factor=0.05,
            description='User info'
        ),

        'social-graph': MicroserviceSpec(
            name='social-graph',
            dependencies=[],
            scale_min=1, scale_max=10,
            execution_time_ms=5.0,
            workers=4,
            degradation_factor=0.05,
            description='Social graph'
        ),

        'url-shorten': MicroserviceSpec(
            name='url-shorten',
            dependencies=[],
            scale_min=1, scale_max=5,
            execution_time_ms=2.0,
            workers=4,
            degradation_factor=0.02,
            description='URL shortening'
        )
    }
    
    @classmethod
    def get_service(cls, name: str) -> MicroserviceSpec:
        """Ottieni specifica servizio"""
        return cls.SERVICES.get(name)
    
    @classmethod
    def get_all_services(cls) -> List[MicroserviceSpec]:
        """Ottieni tutti i servizi"""
        return list(cls.SERVICES.values())
    
    @classmethod
    def get_dependencies(cls, service_name: str) -> List[str]:
        """Ottieni dipendenze di un servizio"""
        spec = cls.SERVICES.get(service_name)
        return spec.dependencies if spec else []
    
    @classmethod
    def get_transitive_dependencies(cls, service_name: str) -> Set[str]:
        """
        Ottieni tutte le dipendenze transitive di un servizio
        Es: nginx-thrift → home-timeline → post-storage
        """
        visited = set()
        to_visit = [service_name]
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            deps = cls.get_dependencies(current)
            to_visit.extend(deps)
        
        # Rimuovi il servizio stesso
        visited.discard(service_name)
        return visited
    
    @classmethod
    def get_call_chain(cls, endpoint: str) -> List[str]:
        """
        Ottieni la catena di chiamate per un endpoint
        
        Endpoint mapping (dal tuo k6 script):
        - homepage (40%): nginx-thrift only
        - user-timeline (30%): nginx → user-timeline → post-storage
        - home-timeline (20%): nginx → home-timeline → post-storage + social-graph
        - compose-post (10%): nginx → compose-post → post-storage + user + url-shorten
        """
        
        chains = {
            'homepage': ['nginx-thrift'],
            
            'user-timeline': [
                'nginx-thrift',
                'user-timeline', 
                'post-storage'
            ],
            
            'home-timeline': [
                'nginx-thrift',
                'home-timeline',
                'post-storage',
                'social-graph'
            ],
            
            'compose-post': [
                'nginx-thrift',
                'compose-post',
                'post-storage',
                'user-service',
                'url-shorten'
            ]
        }
        
        return chains.get(endpoint, ['nginx-thrift'])
    
    @classmethod
    def print_architecture(cls):
        """Stampa architettura microservizi"""
        print("\n" + "="*80)
        print("SOCIAL NETWORK MICROSERVICES ARCHITECTURE")
        print("="*80)
        
        print("\n📦 Services:")
        for spec in cls.get_all_services():
            deps_str = f" → {', '.join(spec.dependencies)}" if spec.dependencies else " (leaf)"
            print(f"\n  • {spec.name}")
            print(f"    {spec.description}")
            print(f"    Scale: [{spec.scale_min}, {spec.scale_max}], "
                  f"Workers: {spec.workers}, "
                  f"Exec: {spec.execution_time_ms}ms")
            print(f"    Dependencies{deps_str}")
        
        print("\n🔗 Call Chains (from k6 endpoints):")
        for endpoint in ['homepage', 'user-timeline', 'home-timeline', 'compose-post']:
            chain = cls.get_call_chain(endpoint)
            print(f"\n  {endpoint}:")
            print(f"    {' → '.join(chain)}")
        
        print("\n" + "="*80)
    
    @classmethod
    def get_endpoint_distribution(cls) -> Dict[str, float]:
        """
        Distribuzione endpoint dal tuo k6 script:
        
        if (rand < 0.40) return GET /
        else if (rand < 0.70) return GET /user-timeline/read
        else if (rand < 0.90) return GET /home-timeline/read
        else return POST /post/compose
        """
        return {
            'homepage': 0.40,       # 40%
            'user-timeline': 0.30,  # 30%
            'home-timeline': 0.20,  # 20%
            'compose-post': 0.10    # 10%
        }


if __name__ == '__main__':
    # Test: stampa architettura
    SocialNetworkMicroservices.print_architecture()
    
    print("\n🧪 Testing dependency resolution:")
    for service in ['nginx-thrift', 'home-timeline', 'compose-post']:
        deps = SocialNetworkMicroservices.get_transitive_dependencies(service)
        print(f"\n{service} → all dependencies: {deps}")