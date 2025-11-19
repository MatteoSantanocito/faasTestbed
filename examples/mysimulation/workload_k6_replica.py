"""
K6 Workload Replica Generator
Replica esatta del tuo script k6 con:
- Pattern giornaliero (traffic_profile 24h)
- Eventi virali e breaking news
- Distribuzione endpoint (40% homepage, 30% user-timeline, etc.)
- RPS max 200
"""

import numpy as np
import random
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Evento spike (virale o breaking news)"""
    type: str  # 'viral' o 'breaking_news'
    start_time: float
    duration: float
    multiplier_target: float
    multiplier_current: float = 1.0
    ramp_completed: bool = False


class K6WorkloadGenerator:
    """
    Replica esatta del tuo SocialNetworkTrafficSimulator
    
    Pattern:
    - 24h traffic profile (notte basso, sera alto)
    - Weekend +20%
    - Eventi virali (1.2-1.8x, durata 3-8% sim)
    - Breaking news (1.8-2.5x, durata 5-10% sim)
    - Noise ±10%
    - User clustering +10-30%
    """
    
    def __init__(self, 
                 rps_max=200,
                 duration_minutes=120,
                 minutes_per_simulated_day=60,
                 random_seed=42,
                 workload_scale=1.0):
        """
        Args:
            rps_max: RPS massimo teorico
            duration_minutes: Durata simulazione
            minutes_per_simulated_day: Minuti reali per giorno simulato
            random_seed: Seed riproducibilità
            workload_scale: Scala workload (0.0-1.0), utile per match testbed reale
                           Es: 0.02 per 17k requests invece di 890k
        """
        
        self.rps_max = rps_max
        self.duration_seconds = duration_minutes * 60
        self.seconds_per_day = minutes_per_simulated_day * 60
        self.num_simulated_days = duration_minutes / minutes_per_simulated_day
        self.workload_scale = workload_scale
        
        # Seed per riproducibilità
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Eventi attivi
        self.active_events: List[Event] = []
        
        # Distribuzione endpoint (dal tuo k6)
        self.endpoint_distribution = {
            'homepage': 0.40,
            'user-timeline': 0.30,
            'home-timeline': 0.20,
            'compose-post': 0.10
        }
        
        logger.info(f"K6WorkloadGenerator initialized:") 
        logger.info(f"   RPS max: {rps_max}")
        logger.info(f"   Duration: {duration_minutes}min ({self.duration_seconds}s)")
        logger.info(f"   Simulated days: {self.num_simulated_days:.1f}")
        logger.info(f"   1 simulated day = {minutes_per_simulated_day}min real")
        logger.info(f"   Seed: {random_seed}")
        if workload_scale != 1.0:
            logger.info(f"   ⚠️  Workload scale: {workload_scale:.2f} ({workload_scale*100:.0f}% of max)")
    
    def _get_traffic_profile(self) -> Dict[int, float]:
        """
        Pattern 24h dal tuo codice (traffic_profile)
        0: minimo notturno
        20: picco serale
        """
        return {
            0: 0.25, 1: 0.20, 2: 0.18, 3: 0.20, 4: 0.25, 5: 0.30,
            6: 0.45, 7: 0.60, 8: 0.70, 9: 0.65, 10: 0.60, 11: 0.70,
            12: 0.85, 13: 0.90, 14: 0.75, 15: 0.70, 16: 0.65, 17: 0.75,
            18: 0.90, 19: 0.95, 20: 1.00, 21: 0.95, 22: 0.80, 23: 0.60,
        }
    
    def base_rps_at_time(self, elapsed_seconds: float) -> float:
        """
        Calcola RPS base per il tempo trascorso
        Include: pattern giornaliero + weekend bonus
        """
        # Ora simulata (0-24)
        simulated_hour = (elapsed_seconds % self.seconds_per_day) / self.seconds_per_day * 24
        
        # Giorno simulato (0-6, 0=Lun, 5=Sab, 6=Dom)
        simulated_day = int(elapsed_seconds / self.seconds_per_day) % 7
        
        # Interpolazione lineare tra ore
        traffic_profile = self._get_traffic_profile()
        hour = int(simulated_hour) % 24
        next_hour = (hour + 1) % 24
        fraction = simulated_hour - hour
        
        current_mult = traffic_profile[hour]
        next_mult = traffic_profile[next_hour]
        multiplier = current_mult + (next_mult - current_mult) * fraction
        
        # Weekend +20%
        if simulated_day >= 5:
            multiplier *= 1.2
        
        return self.rps_max * multiplier
    
    def _add_noise(self, rps: float) -> float:
        """Noise gaussiano ±10%"""
        noise = np.random.normal(0, rps * 0.10)
        return max(10, rps + noise)
    
    def _add_user_clustering(self, rps: float) -> float:
        """User clustering: 15% probabilità di +10-30%"""
        if np.random.random() < 0.15:
            return rps * np.random.uniform(1.1, 1.3)
        return rps
    
    def _trigger_viral_event(self, elapsed_seconds: float):
        """
        Trigger evento virale
        
        Probabilità: 0.003 ogni 9s (dal testbed) = 0.003/9 ≈ 0.00033 per secondo
        """
        if np.random.random() < 0.00033:  # ~0.033% per secondo
            duration = np.random.uniform(
                self.duration_seconds * 0.03,  # 3% durata
                self.duration_seconds * 0.08   # 8% durata
            )
            multiplier = np.random.uniform(1.2, 1.8)
            
            event = Event(
                type='viral',
                start_time=elapsed_seconds,
                duration=duration,
                multiplier_target=multiplier
            )
            self.active_events.append(event)
            
            logger.info(f"🔥 VIRAL EVENT at t={elapsed_seconds:.1f}s: "
                       f"duration={duration:.1f}s, multiplier={multiplier:.2f}x")
    
    def _trigger_breaking_news(self, elapsed_seconds: float):
        """
        Trigger breaking news
        
        Probabilità: 0.001 ogni 9s (dal testbed) = 0.001/9 ≈ 0.00011 per secondo
        """
        if np.random.random() < 0.00011:  # ~0.011% per secondo
            duration = np.random.uniform(
                self.duration_seconds * 0.05,  # 5% durata
                self.duration_seconds * 0.10   # 10% durata
            )
            multiplier = np.random.uniform(1.8, 2.5)
            
            event = Event(
                type='breaking_news',
                start_time=elapsed_seconds,
                duration=duration,
                multiplier_target=multiplier
            )
            self.active_events.append(event)
            
            logger.info(f"📰 BREAKING NEWS at t={elapsed_seconds:.1f}s: "
                       f"duration={duration:.1f}s, multiplier={multiplier:.2f}x")
    
    def _update_event_progression(self, event: Event, elapsed_seconds: float):
        """
        Aggiorna progressione evento con ramp-up/down
        - Primi 20%: ramp-up da 1.0 → target
        - 20-80%: plateau al target
        - Ultimi 20%: ramp-down target → 1.0
        """
        time_in_event = elapsed_seconds - event.start_time
        
        # Ramp-up (primi 20%)
        if time_in_event <= event.duration * 0.2 and not event.ramp_completed:
            progress = time_in_event / (event.duration * 0.2)
            event.multiplier_current = 1.0 + (event.multiplier_target - 1.0) * progress
        
        # Plateau
        elif time_in_event <= event.duration * 0.8:
            if not event.ramp_completed:
                event.ramp_completed = True
            event.multiplier_current = event.multiplier_target
        
        # Ramp-down (ultimi 20%)
        elif time_in_event <= event.duration:
            progress = (time_in_event - event.duration * 0.8) / (event.duration * 0.2)
            event.multiplier_current = (
                event.multiplier_target - 
                (event.multiplier_target - 1.0) * progress
            )
    
    def _apply_active_events(self, base_rps: float, elapsed_seconds: float) -> float:
        """
        Applica eventi attivi
        Priority: breaking_news > viral
        """
        # Update eventi
        active = []
        for event in self.active_events:
            time_in_event = elapsed_seconds - event.start_time
            
            if time_in_event > event.duration:
                logger.info(f"Event '{event.type}' ended at t={elapsed_seconds:.1f}s")
                continue
            
            self._update_event_progression(event, elapsed_seconds)
            active.append(event)
        
        self.active_events = active
        
        # Applica moltiplicatore (priority: breaking > viral)
        breaking_news = [e for e in self.active_events if e.type == 'breaking_news']
        viral = [e for e in self.active_events if e.type == 'viral']
        
        if breaking_news:
            return base_rps * breaking_news[0].multiplier_current
        elif viral:
            return base_rps * viral[0].multiplier_current
        
        return base_rps
    
    def get_rps_at_time(self, elapsed_seconds: float) -> float:
        """
        Calcola RPS target per il tempo dato
        Pipeline completa: base → noise → clustering → events
        
        Note: Eventi vengono triggerati in generate_request_times, non qui!
        """
        # 1. Base RPS (pattern giornaliero + weekend)
        base_rps = self.base_rps_at_time(elapsed_seconds)
        
        # 2. Noise
        rps = self._add_noise(base_rps)
        
        # 3. User clustering (solo se nessun evento attivo)
        if not self.active_events:
            rps = self._add_user_clustering(rps)
        
        # 4. Applica eventi attivi (se ce ne sono)
        rps = self._apply_active_events(rps, elapsed_seconds)
        
        # 5. Cap al max +20%
        rps = min(rps, self.rps_max * 1.2)
        
        return max(10, int(rps))
    
    def sample_endpoint(self) -> str:
        """
        Campiona endpoint secondo distribuzione k6
        40% homepage, 30% user-timeline, 20% home-timeline, 10% compose-post
        """
        rand = random.random()
        
        if rand < 0.40:
            return 'homepage'
        elif rand < 0.70:
            return 'user-timeline'
        elif rand < 0.90:
            return 'home-timeline'
        else:
            return 'compose-post'
    
    def generate_request_times(self, 
                               start_time: float = 0.0,
                               end_time: float = None,
                               time_step: float = 1.0) -> List[Tuple[float, str]]:
        """
        Genera lista di (timestamp, endpoint) per tutta la durata
        
        Args:
            start_time: Tempo inizio
            end_time: Tempo fine
            time_step: Step temporale per calcolo RPS (default 1s)
            
        Returns:
            List[(timestamp, endpoint)]
        """
        if end_time is None:
            end_time = self.duration_seconds
        
        requests = []
        current_time = start_time
        
        logger.info(f"\nGenerating requests from t={start_time:.1f}s to t={end_time:.1f}s...")
        
        # Genera richieste con step temporale
        while current_time < end_time:
            # Trigger eventi (1 volta per step, non per ogni richiesta!)
            if not any(e.type == 'viral' for e in self.active_events):
                self._trigger_viral_event(current_time)
            
            if not any(e.type == 'breaking_news' for e in self.active_events):
                self._trigger_breaking_news(current_time)
            
            # RPS target per questo step
            target_rps = self.get_rps_at_time(current_time)
            
            # Applica scala workload
            target_rps = target_rps * self.workload_scale
            
            # Numero richieste in questo step
            num_requests_in_step = int(target_rps * time_step)
            
            # Genera richieste uniformemente distribuite nello step
            for i in range(num_requests_in_step):
                # Timestamp all'interno dello step
                request_time = current_time + (i / num_requests_in_step) * time_step
                
                if request_time >= end_time:
                    break
                
                # Campiona endpoint
                endpoint = self.sample_endpoint()
                requests.append((request_time, endpoint))
            
            current_time += time_step
        
        logger.info(f"Generated {len(requests)} requests for period [{start_time:.1f}, {end_time:.1f}]")
        
        # Statistiche eventi
        total_events = len([e for e in self.active_events])
        logger.info(f"Events remaining active: {total_events}")
        
        return requests
    
    def print_summary(self):
        """Stampa summary workload"""
        print("\n" + "="*70)
        print("K6 WORKLOAD GENERATOR SUMMARY")
        print("="*70)
        print(f"\n⚙️  Configuration:")
        print(f"  • RPS max:       {self.rps_max}")
        print(f"  • Duration:      {self.duration_seconds}s ({self.duration_seconds/60:.1f}min)")
        print(f"  • Simulated days: {self.num_simulated_days:.1f}")
        print(f"\n📊 Endpoint Distribution:")
        for endpoint, prob in self.endpoint_distribution.items():
            print(f"  • {endpoint:20s}: {prob*100:>5.1f}%")
        print(f"\n🔥 Event Probabilities:")
        print(f"  • Viral events:       0.3% per interval")
        print(f"  • Breaking news:      0.1% per interval")
        print("="*70 + "\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test generator
    gen = K6WorkloadGenerator(
        rps_max=200,
        duration_minutes=10,  # 10min test
        minutes_per_simulated_day=60,
        random_seed=42
    )
    
    gen.print_summary()
    
    # Generate requests
    requests = gen.generate_request_times()
    
    print(f"\n📈 Request Statistics:")
    print(f"  Total requests: {len(requests)}")
    print(f"  Avg RPS: {len(requests) / 600:.1f}")
    
    # Endpoint distribution
    from collections import Counter
    endpoint_counts = Counter(ep for _, ep in requests)
    print(f"\n  Endpoint breakdown:")
    for endpoint, count in endpoint_counts.most_common():
        pct = count / len(requests) * 100
        print(f"    {endpoint:20s}: {count:>5} ({pct:>5.1f}%)")