
## Overview

This project models a cache system with different configurations, including various cache sizes, block sizes, and associativities. Additionally, multiple replacement policies are applied to demonstrate the impact on cache performance. The simulation employs a Reinforcement Learning (RL) agent to learn optimal eviction strategies.

## Cache Simulation

The simulation focuses on a set-associative or fully-associative cache, with a variety of replacement policies, including:

- **LRU (Least Recently Used)**
- **RL (Reinforcement Learning-based)**
- **ReuseBased**
- **HybridLRUReuse**
- **HotCold**
- **MemorAI (Memory-Aware)**

### Key Components

1. **Cache Class**: Manages the cache's state, including cache blocks, eviction policies, and access history.
2. **RLAgent Class**: A reinforcement learning agent that updates its strategy based on reward signals (cache hits, misses, and evictions).
3. **Block Class**: Represents a cache block with a reuse counter and access history to simulate cache behavior.

## RL Agent

The RL agent uses a Q-table to learn from interactions with the cache. The state size is a fixed value (100), and the agent selects actions (cache eviction decisions) based on its exploration/exploitation strategy. The epsilon value determines the balance between random exploration and exploiting learned behavior.

### Key Methods:
- **get_action(state)**: Chooses an action based on the current state and epsilon-greedy strategy.
- **update(state, action, reward, next_state)**: Updates the Q-table after an action is performed.

## Replacement Policies

The cache supports multiple eviction policies:

- **LRU (Least Recently Used)**: Evicts the least recently accessed block.
- **RL (Reinforcement Learning)**: Uses the RL agent to decide which block to evict.
- **ReuseBased**: Evicts the block with the least reuse.
- **HybridLRUReuse**: Combines LRU and reuse-based strategies.
- **HotCold**: Differentiates hot and cold blocks, evicting cold blocks.
- **MemorAI**: Uses access history and probability to predict which block to evict.

## Running the Simulation

To run the simulation:

1. Set the desired **cache size**, **block size**, **associativity level**, and **replacement policy**.
2. Generate a sequence of memory access addresses (randomized, strided, and temporal locality).
3. The cache accesses the generated addresses, and the hit/miss statistics are recorded.
4. Results for different configurations are stored and can be plotted for analysis.
