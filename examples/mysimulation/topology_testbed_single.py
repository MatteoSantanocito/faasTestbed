"""
Topologia Single-Node - Replica Esatta Testbed K8s
Hardware: 6 CPU, 16GB RAM (come il tuo laboratorio)
"""

from ether.core import Node, Link, Capacity
from sim.topology import Topology
import logging

logger = logging.getLogger(__name__)


class SingleNodeTopology:
    """
    Replica esatta del testbed reale:
    - 1 nodo Kubernetes
    - 6 CPU cores
    - 16 GB RAM
    """
    
    def __init__(self):
        self.topology = Topology()
        self.k8s_node = None
        
    def create(self):
        """Crea topologia single-node"""
        
        logger.info("Creating single-node topology (testbed replica)...")
        
        # ========== K8S MASTER NODE ==========
        self.k8s_node = Node(
            name='k8s-master',
            capacity=Capacity(
                cpu_millis=6000,        # 6 CPU cores
                memory=16 * 1024**3     # 16 GB RAM
            ),
            arch='amd64',
            labels={
                'type': 'master',
                'node-role.kubernetes.io/master': '',
                'kubernetes.io/hostname': 'k8s-master'
            }
        )
        
        self.topology.add_node(self.k8s_node)
        
        # ========== NETWORK LINK ==========
        # Crea un link di rete locale (come negli esempi faas-sim)
        # Skippy usa questo link per calcolare bandwidth
        local_network = Link(bandwidth=100000)  # 100 Gbps localhost
        local_network.tags = {'name': 'localhost_network'}
        self.topology.add_node(local_network)
        
        # Connetti il nodo K8s al link locale (bidirezionale)
        self.topology.add_edge(self.k8s_node, local_network, latency=0.001)
        self.topology.add_edge(local_network, self.k8s_node, latency=0.001)
        
        logger.info("  ✓ Local network link created (100 Gbps)")
        
        # ========== DOCKER REGISTRY ==========
        self.topology.init_docker_registry()
        
        # Connetti registry al nodo (localhost, latenza minima)
        registry_node = None
        for node in self.topology.get_nodes():
            if node.name == 'registry':
                registry_node = node
                break
        
        if registry_node:
            # Connetti registry al link locale (NON direttamente al nodo k8s)
            # Questo è come fanno gli esempi originali di faas-sim
            self.topology.add_edge(registry_node, local_network, latency=0.1)
            self.topology.add_edge(local_network, registry_node, latency=0.1)
            logger.info("  ✓ Docker registry connected to local network")
        
        logger.info(f"✅ Single-node topology created")
        logger.info(f"   Node: {self.k8s_node.name}")
        logger.info(f"   CPU: {self.k8s_node.capacity.cpu_millis / 1000:.1f} cores")
        logger.info(f"   RAM: {self.k8s_node.capacity.memory / 1024**3:.1f} GB")
        
        return self.topology
    
    def get_node(self):
        """Ritorna il nodo K8s"""
        return self.k8s_node
    
    def print_summary(self):
        """Stampa sommario topologia"""
        print("\n" + "="*60)
        print("TESTBED TOPOLOGY (Single Node)")
        print("="*60)
        print(f"\n📊 Hardware Configuration:")
        print(f"  • Node:   k8s-master")
        print(f"  • CPU:    6 cores")
        print(f"  • RAM:    16 GB")
        print(f"  • Arch:   amd64")
        print(f"\n🌐 Network:")
        print(f"  • Topology:   Star (all nodes → local_network)")
        print(f"  • Bandwidth:  100 Gbps")
        print(f"  • Latency:    0.001ms (node ↔ network), 0.1ms (registry ↔ network)")
        print("="*60 + "\n")


def create_testbed_topology():
    """Helper per creare topologia testbed"""
    topo = SingleNodeTopology()
    return topo.create()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    topo_builder = SingleNodeTopology()
    topology = topo_builder.create()
    topo_builder.print_summary()