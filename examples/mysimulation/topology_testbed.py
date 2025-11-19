"""
Topologia eterogenea edge-cloud CORRETTA
"""

from ether.core import Node, Link, Capacity
from sim.topology import Topology
import logging

logger = logging.getLogger(__name__)


class AdvancedTopology:
    """Topologia edge-cloud eterogenea"""
    
    def __init__(self):
        self.topology = Topology()
        self.nodes_by_type = {
            'edge_rpi': [],
            'edge_tx2': [],
            'cloud': []
        }
        self.edge_switch = None
        self.cloud_gateway = None
        
    def create(self):
        """Crea la topologia completa"""
        
        # ========== NODI EDGE: Raspberry Pi 3 ==========
        logger.info("Creating Raspberry Pi 3 nodes (ARM32, limited resources)...")
        for i in range(3):
            node = Node(
                name=f'rpi3-{i}',
                capacity=Capacity(
                    cpu_millis=4000,
                    memory=1 * 1024**3
                ),
                arch='arm32v7',
                labels={
                    'type': 'edge',
                    'device': 'rpi3',
                    'location': 'edge-site-1',
                    'tier': 'edge'
                }
            )
            self.topology.add_node(node)
            self.nodes_by_type['edge_rpi'].append(node)
        
        # ========== NODI EDGE: Jetson TX2 ==========
        logger.info("Creating Jetson TX2 nodes (ARM64, GPU-capable)...")
        for i in range(2):
            node = Node(
                name=f'jetson-tx2-{i}',
                capacity=Capacity(
                    cpu_millis=8000,
                    memory=8 * 1024**3
                ),
                arch='aarch64',
                labels={
                    'type': 'edge',
                    'device': 'jetson-tx2',
                    'location': 'edge-site-1',
                    'tier': 'edge',
                    'capability.skippy.io/gpu': '',
                    'nvidia.com/gpu': 'pascal'
                }
            )
            self.topology.add_node(node)
            self.nodes_by_type['edge_tx2'].append(node)
        
        # ========== NODO CLOUD ==========
        logger.info("Creating cloud server node (x86, high resources)...")
        cloud_server = Node(
            name='cloud-server-0',
            capacity=Capacity(
                cpu_millis=88000,
                memory=188 * 1024**3
            ),
            arch='amd64',
            labels={
                'type': 'cloud',
                'device': 'server',
                'location': 'datacenter',
                'tier': 'cloud'
            }
        )
        self.topology.add_node(cloud_server)
        self.nodes_by_type['cloud'].append(cloud_server)
        
        # ========== NETWORK TOPOLOGY ==========
        self._create_network_topology()
        
        # ========== DOCKER REGISTRY ==========
        self.topology.init_docker_registry()
        self._connect_registry()
        
        logger.info(f"✅ Topology created: {len(self.topology.get_nodes())} nodes total")
        return self.topology
    
    def _create_network_topology(self):
        """Crea topologia di rete"""
        logger.info("Creating network topology...")
        
        # ========== EDGE LOCAL SWITCH ==========
        self.edge_switch = Link(bandwidth=1000)  # 1 Gbps LAN
        self.edge_switch.tags = {'name': 'edge-switch', 'type': 'lan'}
        self.topology.add_node(self.edge_switch)
        
        # Connetti nodi edge allo switch
        all_edge_nodes = self.nodes_by_type['edge_rpi'] + self.nodes_by_type['edge_tx2']
        
        for node in all_edge_nodes:
            istio_overhead = 0.5
            self.topology.add_edge(node, self.edge_switch, 
                                 latency=1.0 + istio_overhead,
                                 bandwidth=100)
            self.topology.add_edge(self.edge_switch, node, 
                                 latency=1.0 + istio_overhead,
                                 bandwidth=100)
        
        # ========== CLOUD GATEWAY ==========
        self.cloud_gateway = Link(bandwidth=10000)  # 10 Gbps
        self.cloud_gateway.tags = {'name': 'cloud-gateway', 'type': 'wan'}
        self.topology.add_node(self.cloud_gateway)
        
        # Connetti edge switch al cloud gateway (WAN)
        self.topology.add_edge(self.edge_switch, self.cloud_gateway, 
                             latency=20.0, bandwidth=1000)
        self.topology.add_edge(self.cloud_gateway, self.edge_switch, 
                             latency=20.0, bandwidth=1000)
        
        # Connetti server cloud al gateway
        for cloud_node in self.nodes_by_type['cloud']:
            istio_overhead = 0.5
            self.topology.add_edge(cloud_node, self.cloud_gateway,
                                 latency=0.5 + istio_overhead,
                                 bandwidth=10000)
            self.topology.add_edge(self.cloud_gateway, cloud_node,
                                 latency=0.5 + istio_overhead,
                                 bandwidth=10000)
        
        logger.info("  ✓ Edge LAN: 1 Gbps, ~1ms latency")
        logger.info("  ✓ Edge-to-Cloud WAN: ~20ms latency")
        logger.info("  ✓ Istio proxy overhead: ~0.5ms per hop")
    
    def _connect_registry(self):
        """
        Connetti Docker registry alla rete
        FIX: Registry connesso all'EDGE SWITCH, non al cloud gateway!
        Questo permette a tutti i nodi (edge + cloud) di raggiungerlo
        """
        # Trova registry node
        registry_node = None
        for node in self.topology.get_nodes():
            if node.name == 'registry':
                registry_node = node
                break
        
        if not registry_node:
            logger.warning("Registry node not found!")
            return
        
        # ✅ FIX: Connetti all'EDGE SWITCH invece che al cloud gateway
        # Così TUTTI i nodi possono raggiungere il registry
        if self.edge_switch:
            self.topology.add_edge(registry_node, self.edge_switch, latency=0.5)
            self.topology.add_edge(self.edge_switch, registry_node, latency=0.5)
            logger.info("  ✓ Docker registry connected to edge switch (accessible by all nodes)")
        else:
            logger.warning("Edge switch not found, registry may not be reachable!")
    
    def get_edge_nodes(self):
        """Ritorna nodi edge"""
        return self.nodes_by_type['edge_rpi'] + self.nodes_by_type['edge_tx2']
    
    def get_cloud_nodes(self):
        """Ritorna nodi cloud"""
        return self.nodes_by_type['cloud']
    
    def get_gpu_nodes(self):
        """Ritorna nodi con GPU"""
        return self.nodes_by_type['edge_tx2']
    
    def print_summary(self):
        """Stampa sommario topologia"""
        print("\n" + "="*60)
        print("TOPOLOGY SUMMARY")
        print("="*60)
        print(f"\n📊 Node Distribution:")
        print(f"  • Edge (Raspberry Pi 3):  {len(self.nodes_by_type['edge_rpi'])} nodes")
        print(f"  • Edge (Jetson TX2):      {len(self.nodes_by_type['edge_tx2'])} nodes (GPU)")
        print(f"  • Cloud (Server x86):     {len(self.nodes_by_type['cloud'])} nodes")
        
        print(f"\n💾 Total Resources:")
        total_cpu = sum(n.capacity.cpu_millis for n in self.topology.get_nodes() 
                       if hasattr(n, 'capacity'))
        total_mem = sum(n.capacity.memory for n in self.topology.get_nodes() 
                       if hasattr(n, 'capacity'))
        print(f"  • Total CPU:     {total_cpu / 1000:.1f} cores")
        print(f"  • Total Memory:  {total_mem / 1024**3:.1f} GB")
        
        print(f"\n🌐 Network:")
        print(f"  • Edge LAN:       1 Gbps, ~1ms latency")
        print(f"  • Edge-to-Cloud:  ~20ms latency")
        print(f"  • Istio overhead: ~0.5ms per hop")
        print("="*60 + "\n")


def create_advanced_topology():
    """Helper per creare topologia"""
    topo = AdvancedTopology()
    return topo.create()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    topo_builder = AdvancedTopology()
    topology = topo_builder.create()
    topo_builder.print_summary()