import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RYGate, CXGate, iSwapGate, CUGate
from enum import Enum
import random
from typing import Dict, List, Callable, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScottDomainState(Enum):
    """Scott domain lattice positions for strategy resolution"""
    UNDEFINED = 0      # ⊥ (bottom)
    PARTIAL = 1        # + (partial information)
    RESOLVED = 2       # ⊤ (top/resolved)

class SCGSAgent:
    """
    Shadow Cooperative Game Structure Agent with quantum strategy register
    and Scott domain tracking for adversarial multi-agent systems.
    """
    
    def __init__(self, 
                 agent_id: str,
                 adversarial_reward_fn: Callable[[Dict], float] = None,
                 entanglement_operator: Callable[[float], RYGate] = None,
                 initial_domain_state: ScottDomainState = ScottDomainState.UNDEFINED):
        """
        Initialize SCGS Agent
        
        Args:
            agent_id: Unique identifier for the agent
            adversarial_reward_fn: Function f_i() that computes agent's reward
            entanglement_operator: Function Φ() that creates entanglement bias
            initial_domain_state: Starting position in Scott domain lattice
        """
        self.agent_id = agent_id
        self.domain_state = initial_domain_state
        
        # Quantum strategy register
        self.qreg = QuantumRegister(1, f'q_{agent_id}')
        self.creg = ClassicalRegister(1, f'c_{agent_id}')
        self.circuit = QuantumCircuit(self.qreg, self.creg)
        
        # Initialize qubit based on domain state
        self._initialize_quantum_state()
        
        # Strategy functions
        self.adversarial_reward_fn = adversarial_reward_fn or self._default_adversarial_reward
        self.entanglement_operator = entanglement_operator or self._default_entanglement_operator
        
        # Internal state tracking
        self.state = {
            'strategy_level': self.domain_state.value,
            'reward_history': [],
            'entanglement_trace': [],
            'collapse_history': [],
            'timestep': 0
        }
        
        # Entanglement tracking
        self.entangled_agents = set()
        self.entanglement_strength = 0.0
        
        logger.info(f"Agent {self.agent_id} initialized with domain state: {self.domain_state.name}")
    
    def _initialize_quantum_state(self):
        """Initialize qubit based on Scott domain state"""
        if self.domain_state == ScottDomainState.UNDEFINED:
            # |0⟩ state (⊥)
            pass  # Already in |0⟩
        elif self.domain_state == ScottDomainState.PARTIAL:
            # |+⟩ state (superposition)
            self.circuit.h(self.qreg[0])
        elif self.domain_state == ScottDomainState.RESOLVED:
            # |1⟩ state (resolved)
            self.circuit.x(self.qreg[0])
    
    def _default_adversarial_reward(self, context: Dict) -> float:
        """Default adversarial reward function f_i()"""
        # Adversarial logic: reward based on relative domain position
        own_level = context.get('own_level', 0)
        opponent_levels = context.get('opponent_levels', [])
        
        if not opponent_levels:
            return own_level * 0.5
        
        # Adversarial: reward for being ahead in domain hierarchy
        avg_opponent_level = sum(opponent_levels) / len(opponent_levels)
        return (own_level - avg_opponent_level) * 2.0 + random.uniform(-0.1, 0.1)
    
    def _default_entanglement_operator(self, theta: float) -> RYGate:
        """Default entanglement operator Φ() - RY rotation"""
        return RYGate(theta)
    
    def apply_internal_strategy(self, context: Dict = None):
        """
        Apply internal adversarial strategy to evolve through Scott domain
        ⊥ → + → resolved (monotonic transformation)
        """
        context = context or {}
        
        # Calculate adversarial reward
        reward_context = {
            'own_level': self.domain_state.value,
            'opponent_levels': context.get('opponent_levels', []),
            'timestep': self.state['timestep']
        }
        
        reward = self.adversarial_reward_fn(reward_context)
        self.state['reward_history'].append(reward)
        
        # Adversarial decision: whether to evolve domain state
        evolution_probability = min(0.8, max(0.1, (reward + 1.0) / 2.0))
        
        if random.random() < evolution_probability:
            # Monotonic progression through Scott domain
            if self.domain_state == ScottDomainState.UNDEFINED:
                self.domain_state = ScottDomainState.PARTIAL
                # Apply Hadamard to create superposition
                self.circuit.h(self.qreg[0])
                logger.info(f"Agent {self.agent_id}: ⊥ → + (partial strategy)")
                
            elif self.domain_state == ScottDomainState.PARTIAL:
                self.domain_state = ScottDomainState.RESOLVED
                # Bias toward |1⟩ with rotation
                self.circuit.ry(np.pi/4, self.qreg[0])
                logger.info(f"Agent {self.agent_id}: + → ⊤ (resolved strategy)")
        
        self.state['strategy_level'] = self.domain_state.value
        self.state['timestep'] += 1
    
    def entangle_with_agent(self, other_agent: 'SCGSAgent', 
                          entanglement_type: str = 'cx',
                          theta: float = np.pi/4):
        """
        Create quantum entanglement with another agent
        
        Args:
            other_agent: Target agent for entanglement
            entanglement_type: Type of entanglement ('cx', 'iswap', 'cu')
            theta: Rotation angle for entanglement operator
        """
        # Create combined circuit
        combined_qreg = QuantumRegister(2, f'entangled_{self.agent_id}_{other_agent.agent_id}')
        combined_circuit = QuantumCircuit(combined_qreg)
        
        # Apply current states to combined circuit
        if self.domain_state == ScottDomainState.PARTIAL:
            combined_circuit.h(combined_qreg[0])
        elif self.domain_state == ScottDomainState.RESOLVED:
            combined_circuit.x(combined_qreg[0])
            
        if other_agent.domain_state == ScottDomainState.PARTIAL:
            combined_circuit.h(combined_qreg[1])
        elif other_agent.domain_state == ScottDomainState.RESOLVED:
            combined_circuit.x(combined_qreg[1])
        
        # Apply entanglement operator
        if entanglement_type == 'cx':
            combined_circuit.cx(combined_qreg[0], combined_qreg[1])
        elif entanglement_type == 'iswap':
            combined_circuit.iswap(combined_qreg[0], combined_qreg[1])
        elif entanglement_type == 'cu':
            u_gate = self.entanglement_operator(theta)
            combined_circuit.cu(u_gate, combined_qreg[0], combined_qreg[1])
        
        # Apply entanglement bias operator Φ()
        phi_gate = self.entanglement_operator(theta)
        combined_circuit.append(phi_gate, [combined_qreg[0]])
        combined_circuit.append(phi_gate, [combined_qreg[1]])
        
        # Update entanglement tracking
        self.entangled_agents.add(other_agent.agent_id)
        other_agent.entangled_agents.add(self.agent_id)
        
        self.entanglement_strength += theta
        other_agent.entanglement_strength += theta
        
        entanglement_record = {
            'partner': other_agent.agent_id,
            'type': entanglement_type,
            'strength': theta,
            'timestep': self.state['timestep']
        }
        
        self.state['entanglement_trace'].append(entanglement_record)
        other_agent.state['entanglement_trace'].append({
            'partner': self.agent_id,
            'type': entanglement_type,
            'strength': theta,
            'timestep': other_agent.state['timestep']
        })
        
        logger.info(f"Entanglement created: {self.agent_id} ↔ {other_agent.agent_id} "
                   f"(type: {entanglement_type}, θ: {theta:.3f})")
        
        return combined_circuit
    
    def measure_and_collapse(self, simulator: AerSimulator = None, shots: int = 1024):
        """
        Measure quantum strategy register and update domain state
        
        Args:
            simulator: Qiskit simulator instance
            shots: Number of measurement shots
        """
        if simulator is None:
            simulator = AerSimulator()
        
        # Add measurement to circuit
        measure_circuit = self.circuit.copy()
        measure_circuit.measure(self.qreg[0], self.creg[0])
        
        # Execute measurement
        transpiled_circuit = transpile(measure_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=shots).result()
        counts = result.get_counts()
        
        # Determine collapse outcome
        if '1' in counts:
            prob_one = counts['1'] / shots
        else:
            prob_one = 0.0
        
        # Strategic collapse based on probability
        collapsed_to_one = prob_one > 0.5
        
        # Update domain state based on collapse
        if collapsed_to_one:
            if self.domain_state != ScottDomainState.RESOLVED:
                self.domain_state = ScottDomainState.RESOLVED
                logger.info(f"Agent {self.agent_id}: Quantum collapse → RESOLVED (|1⟩)")
        else:
            if self.domain_state == ScottDomainState.RESOLVED:
                self.domain_state = ScottDomainState.PARTIAL
                logger.info(f"Agent {self.agent_id}: Quantum collapse → PARTIAL (|0⟩)")
        
        # Record collapse
        collapse_record = {
            'prob_one': prob_one,
            'collapsed_to_one': collapsed_to_one,
            'resulting_state': self.domain_state.name,
            'timestep': self.state['timestep']
        }
        
        self.state['collapse_history'].append(collapse_record)
        self.state['strategy_level'] = self.domain_state.value
        
        return collapse_record
    
    def calculate_total_reward(self) -> float:
        """Calculate total reward f_i + Φ"""
        base_reward = sum(self.state['reward_history'])
        entanglement_bonus = self.entanglement_strength * 0.5
        return base_reward + entanglement_bonus
    
    def get_state_summary(self) -> Dict:
        """Get comprehensive state summary"""
        return {
            'agent_id': self.agent_id,
            'domain_state': self.domain_state.name,
            'strategy_level': self.state['strategy_level'],
            'total_reward': self.calculate_total_reward(),
            'entangled_with': list(self.entangled_agents),
            'entanglement_strength': self.entanglement_strength,
            'timestep': self.state['timestep'],
            'last_collapse': self.state['collapse_history'][-1] if self.state['collapse_history'] else None
        }
    
    def log_state(self):
        """Log current agent state"""
        summary = self.get_state_summary()
        logger.info(f"Agent {self.agent_id} State: {summary}")


class SCGSEnvironment:
    """SCGS Multi-Agent Environment Manager"""
    
    def __init__(self, agents: List[SCGSAgent]):
        self.agents = agents
        self.simulator = AerSimulator()
        self.round_number = 0
        
    def run_round(self):
        """Execute one round of SCGS interaction"""
        logger.info(f"\n=== SCGS Round {self.round_number} ===")
        
        # Phase 1: Internal strategy application
        opponent_levels = [agent.domain_state.value for agent in self.agents]
        
        for agent in self.agents:
            context = {
                'opponent_levels': [level for i, level in enumerate(opponent_levels) 
                                  if self.agents[i].agent_id != agent.agent_id]
            }
            agent.apply_internal_strategy(context)
        
        # Phase 2: Entanglement phase (random pairings)
        available_agents = self.agents.copy()
        while len(available_agents) >= 2:
            agent1 = available_agents.pop(random.randint(0, len(available_agents)-1))
            agent2 = available_agents.pop(random.randint(0, len(available_agents)-1))
            
            # Random entanglement parameters
            entanglement_type = random.choice(['cx', 'iswap', 'cu'])
            theta = random.uniform(np.pi/8, np.pi/2)
            
            agent1.entangle_with_agent(agent2, entanglement_type, theta)
        
        # Phase 3: Measurement and collapse
        for agent in self.agents:
            agent.measure_and_collapse(self.simulator)
        
        # Phase 4: Logging
        logger.info("\n--- Round Results ---")
        for agent in self.agents:
            agent.log_state()
        
        self.round_number += 1
    
    def get_system_state(self) -> Dict:
        """Get overall system state"""
        return {
            'round': self.round_number,
            'agents': [agent.get_state_summary() for agent in self.agents],
            'total_entanglements': sum(len(agent.entangled_agents) for agent in self.agents) // 2,
            'domain_distribution': {
                'UNDEFINED': sum(1 for a in self.agents if a.domain_state == ScottDomainState.UNDEFINED),
                'PARTIAL': sum(1 for a in self.agents if a.domain_state == ScottDomainState.PARTIAL),
                'RESOLVED': sum(1 for a in self.agents if a.domain_state == ScottDomainState.RESOLVED)
            }
        }


def create_adversarial_reward_fn(agent_id: str) -> Callable[[Dict], float]:
    """Create agent-specific adversarial reward function"""
    def adversarial_reward(context: Dict) -> float:
        own_level = context.get('own_level', 0)
        opponent_levels = context.get('opponent_levels', [])
        timestep = context.get('timestep', 0)
        
        if not opponent_levels:
            return own_level * 0.3
        
        # Adversarial logic with agent-specific bias
        avg_opponent = sum(opponent_levels) / len(opponent_levels)
        competitive_reward = (own_level - avg_opponent) * 1.5
        
        # Agent-specific strategy bias
        if agent_id.endswith('_catalyst'):
            # Catalyst prefers entanglement opportunities
            return competitive_reward + 0.2
        elif agent_id.endswith('_defender'):
            # Defender prefers resolved states
            return competitive_reward + (own_level == 2) * 0.3
        else:
            # Default adversarial agent
            return competitive_reward + random.uniform(-0.1, 0.1)
    
    return adversarial_reward


def create_entanglement_operator(bias: float = 0.0) -> Callable[[float], RYGate]:
    """Create entanglement operator with bias"""
    def entanglement_op(theta: float) -> RYGate:
        return RYGate(theta + bias)
    return entanglement_op


def main():
    """Example usage: 3 agents, 1 round of SCGS interaction"""
    logger.info("Initializing SCGS Multi-Agent System")
    
    # Create agents with different roles
    agents = [
        SCGSAgent(
            agent_id="alpha_catalyst",
            adversarial_reward_fn=create_adversarial_reward_fn("alpha_catalyst"),
            entanglement_operator=create_entanglement_operator(0.1),
            initial_domain_state=ScottDomainState.UNDEFINED
        ),
        SCGSAgent(
            agent_id="beta_defender", 
            adversarial_reward_fn=create_adversarial_reward_fn("beta_defender"),
            entanglement_operator=create_entanglement_operator(-0.05),
            initial_domain_state=ScottDomainState.PARTIAL
        ),
        SCGSAgent(
            agent_id="gamma_adversary",
            adversarial_reward_fn=create_adversarial_reward_fn("gamma_adversary"),
            entanglement_operator=create_entanglement_operator(0.0),
            initial_domain_state=ScottDomainState.UNDEFINED
        )
    ]
    
    # Create environment
    env = SCGSEnvironment(agents)
    
    # Run simulation
    logger.info("Starting SCGS simulation...")
    
    # Initial state
    logger.info("\n=== Initial System State ===")
    initial_state = env.get_system_state()
    logger.info(f"System state: {initial_state}")
    
    # Run one round
    env.run_round()
    
    # Final state
    logger.info("\n=== Final System State ===")
    final_state = env.get_system_state()
    logger.info(f"System state: {final_state}")
    
    # MAS Attractor Analysis
    logger.info("\n=== MAS Attractor Analysis ===")
    total_rewards = sum(agent.calculate_total_reward() for agent in agents)
    entanglement_density = final_state['total_entanglements'] / len(agents)
    
    logger.info(f"Total System Reward: {total_rewards:.3f}")
    logger.info(f"Entanglement Density: {entanglement_density:.3f}")
    
    # Check for MAS (Mutually Assured Success) attractor
    resolved_agents = final_state['domain_distribution']['RESOLVED']
    if resolved_agents >= 2 and total_rewards > 0:
        logger.info("✓ MAS Attractor detected: Cooperative emergence achieved")
    else:
        logger.info("✗ MAS Attractor not achieved: System in adversarial equilibrium")


if __name__ == "__main__":
    main()
