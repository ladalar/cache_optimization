import random
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q_table = np.zeros((state_size, action_size))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.Q_table[state, :])

    def update(self, state, action, reward, next_state):
        self.Q_table[state, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.Q_table[next_state, :]) - self.Q_table[state, action])
        self.epsilon *= self.epsilon_decay

class Block:
    def __init__(self, block_index, reuse_counter=0, is_hot=False):
        self.block_index = block_index
        self.reuse_counter = reuse_counter
        self.is_hot = is_hot
        self.access_history = []

class Cache:
    def __init__(self, cache_size, block_size, associativity, replacement_policy='LRU'):
        self.cache_size = cache_size
        self.block_size = block_size
        self.associativity = associativity
        self.replacement_policy = replacement_policy
        self.rl_agent = RLAgent(state_size=100, action_size=associativity)  # Initialize RL agent with state_size and action_size
        
        # If associativity is the same as the number of blocks, it's fully associative
        if associativity == cache_size // block_size:
            self.num_sets = 1
        else:
            self.num_sets = cache_size // (block_size * associativity)

        if self.num_sets == 0:
            raise ValueError("Invalid cache configuration: num_sets is zero.")

        self.cache = {i: collections.deque(maxlen=associativity) for i in range(self.num_sets)}
        self.accesses = 0
        self.hits = 0
        self.misses = 0

    def get_state(self):
        # Convert the current state of the cache to an integer representation
        state = sum(len(self.cache[i]) for i in range(self.num_sets))
        return hash(state) % self.rl_agent.state_size

    def calculate_reward(self):
        # Placeholder for calculating the reward
        return 1

    def predict_access_probability(self, blocks):
        # Placeholder for predicting access probability
        return [random.random() for _ in blocks]

    def access(self, address):
        self.accesses += 1
        block_index = address // self.block_size
        set_index = block_index % self.num_sets

        if self.num_sets == 1:  # Fully associative cache
            found = False
            for i in range(len(self.cache[0])):
                if self.cache[0][i] is not None and self.cache[0][i].block_index == block_index:
                    self.hits += 1
                    if self.replacement_policy == 'LRU':
                        block = self.cache[0].remove(self.cache[0][i])
                        self.cache[0].append(block)
                    elif self.replacement_policy == 'HotCold':
                        block = next(block for block in self.cache[0] if block.block_index == block_index)
                        block.is_hot = True
                    found = True
                    break
            if not found:
                self.misses += 1
                if len(self.cache[0]) >= self.associativity:
                    if self.replacement_policy == 'LRU':
                        self.cache[0].popleft()
                    elif self.replacement_policy == 'FIFO':
                        self.cache[0].popleft()
                    elif self.replacement_policy == 'Random':
                        evict_index = random.randint(0, self.associativity - 1)
                        del self.cache[0][evict_index]
                    elif self.replacement_policy == 'RL':
                        state = self.get_state()
                        action = self.rl_agent.get_action(state)
                        evict_index = action
                        del self.cache[0][evict_index]
                        reward = self.calculate_reward()
                        next_state = self.get_state()
                        self.rl_agent.update(state, action, reward, next_state)
                    elif self.replacement_policy == 'ReuseBased':
                        min_reuse = float('inf')
                        evict_index = 0
                        for i, block in enumerate(self.cache[0]):
                            if block.reuse_counter < min_reuse:
                                min_reuse = block.reuse_counter
                                evict_index = i
                        del self.cache[0][evict_index]
                    elif self.replacement_policy == 'HybridLRUReuse':
                        min_reuse = float('inf')
                        min_lru_index = 0
                        for i, block in enumerate(self.cache[0]):
                            if block.reuse_counter < min_reuse:
                                min_reuse = block.reuse_counter
                                min_lru_index = i
                            elif block.reuse_counter == min_reuse and i < min_lru_index:
                                min_lru_index = i
                        del self.cache[0][min_lru_index]
                    elif self.replacement_policy == 'HotCold':
                        min_reuse = float('inf')
                        min_lru_index = 0
                        for i, block in enumerate(self.cache[0]):
                            if not block.is_hot and block.reuse_counter < min_reuse:
                                min_reuse = block.reuse_counter
                                min_lru_index = i
                        del self.cache[0][min_lru_index]
                    elif self.replacement_policy == 'MemorAI':
                        probabilities = self.predict_access_probability(self.cache[0])
                        evict_index = probabilities.index(min(probabilities))
                        del self.cache[0][evict_index]
                    else:
                        raise ValueError("Invalid replacement policy")
                self.cache[0].append(Block(block_index, is_hot=True))
        else:  # Set-associative cache
            if block_index in [block.block_index for block in self.cache[set_index] if block is not None]:
                self.hits += 1
                if self.replacement_policy == 'LRU':
                    block = next(block for block in self.cache[set_index] if block.block_index == block_index)
                    self.cache[set_index].remove(block)
                    self.cache[set_index].append(block)
                elif self.replacement_policy == 'HotCold':
                    block = next(block for block in self.cache[set_index] if block.block_index == block_index)
                    block.is_hot = True
            else:
                self.misses += 1
                if len(self.cache[set_index]) >= self.associativity:
                    if self.replacement_policy == 'LRU':
                        self.cache[set_index].popleft()
                    elif self.replacement_policy == 'FIFO':
                        self.cache[set_index].popleft()
                    elif self.replacement_policy == 'Random':
                        evict_index = random.randint(0, self.associativity - 1)
                        del self.cache[set_index][evict_index]
                    elif self.replacement_policy == 'RL':
                        state = self.get_state()
                        action = self.rl_agent.get_action(state)
                        evict_index = action
                        del self.cache[set_index][evict_index]
                        reward = self.calculate_reward()
                        next_state = self.get_state()
                        self.rl_agent.update(state, action, reward, next_state)
                    elif self.replacement_policy == 'ReuseBased':
                        min_reuse = float('inf')
                        evict_index = 0
                        for i, block in enumerate(self.cache[set_index]):
                            if block.reuse_counter < min_reuse:
                                min_reuse = block.reuse_counter
                                evict_index = i
                        del self.cache[set_index][evict_index]
                    elif self.replacement_policy == 'HybridLRUReuse':
                        min_reuse = float('inf')
                        min_lru_index = 0
                        for i, block in enumerate(self.cache[set_index]):
                            if block.reuse_counter < min_reuse:
                                min_reuse = block.reuse_counter
                                min_lru_index = i
                            elif block.reuse_counter == min_reuse and i < min_lru_index:
                                min_lru_index = i
                        del self.cache[set_index][min_lru_index]
                    elif self.replacement_policy == 'HotCold':
                        min_reuse = float('inf')
                        min_lru_index = 0
                        for i, block in enumerate(self.cache[set_index]):
                            if not block.is_hot and block.reuse_counter < min_reuse:
                                min_reuse = block.reuse_counter
                                min_lru_index = i
                        del self.cache[set_index][min_lru_index]
                    elif self.replacement_policy == 'MemorAI':
                        probabilities = self.predict_access_probability(self.cache[set_index])
                        evict_index = probabilities.index(min(probabilities))
                        del self.cache[set_index][evict_index]
                    else:
                        raise ValueError("Invalid replacement policy")
                self.cache[set_index].append(Block(block_index, is_hot=True))
    
    def get_hit_rate(self):
        return self.hits / self.accesses if self.accesses > 0 else 0
    
    def get_miss_rate(self):
        return self.misses / self.accesses if self.accesses > 0 else 0
    
    def clear_cache(self):
        self.cache = {i: collections.deque(maxlen=self.associativity) for i in range(self.num_sets)}
        self.accesses = 0
        self.hits = 0
        self.misses = 0

# Parameters for simulation
cache_sizes = [512, 1024, 2048, 4096]  # Increased cache sizes in number of blocks
block_sizes = [4, 8, 16, 32]  # Different block sizes
associativity_levels = [1, 2, 4, 8, 16]  # Increased associativity levels
replacement_policies = ['LRU', 'FIFO', 'Random', 'RL', 'ReuseBased', 'HybridLRUReuse', 'HotCold', 'MemorAI']  # Different replacement policies
num_accesses = 10000  # Increased number of memory accesses to simulate

# Simulate a mix of localized and random memory accesses
addresses = []
for _ in range(500):
    base = random.randint(0, 10000)
    # Localized accesses
    addresses.extend([base + i * 4 for i in range(10)])
    # Strided access pattern (every 4th word)
    addresses.extend([base + i * 4 for i in range(10, 100, 4)])  # Strided accesses
    # Temporal locality (repeated accesses to the same addresses)
    addresses.extend([base + i * 4 for i in range(5)])  # Repeated access
addresses.extend(random.randint(0, 10000) for _ in range(num_accesses - len(addresses)))

# Data for plotting
results = []

# Run the simulation
for cache_size in cache_sizes:
    for block_size in block_sizes:
        for associativity in associativity_levels:
            if associativity > cache_size // block_size:
                continue  # Skip invalid configurations
            for policy in replacement_policies:
                try:
                    cache = Cache(cache_size, block_size, associativity, policy)
                except ValueError as e:
                    print(f"Skipping invalid configuration: Cache Size: {cache_size} blocks, Block Size: {block_size} bytes, Associativity: {associativity}-way, Policy: {policy}")
                    continue
                
                for address in addresses:
                    cache.access(address)
                
                if associativity == 1:
                    config_type = "Direct-Mapped"
                elif associativity == cache_size // block_size:
                    config_type = "Fully Associative"
                else:
                    config_type = f"Set-Associative {associativity}-way"
                
                hit_rate = cache.get_hit_rate() * 100
                miss_rate = cache.get_miss_rate() * 100
                results.append((cache_size, block_size, associativity, policy, config_type, hit_rate, miss_rate))
                
                print(f"{config_type} Cache: Cache Size: {cache_size} blocks, Block Size: {block_size} bytes, Policy: {policy}")
                print(f"Hit Rate: {hit_rate:.2f}%, Miss Rate: {miss_rate:.2f}%")
                print("-" * 40)
                cache.clear_cache()

# Create a DataFrame from the results
# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['Cache Size', 'Block Size', 'Associativity', 'Policy', 'Config Type', 'Hit Rate (%)', 'Miss Rate (%)'])

# Save the DataFrame to a PDF file
df.to_csv('cache_simulation_results.csv', index=False)


# Plotting the impact of block size and replacement policies on cache hit rates
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

markers = ['o', 's', 'D', '^']

for ax, (block_size, marker) in zip(axs.flatten(), zip(block_sizes, markers)):
    for policy in replacement_policies:
        hit_rates = [result[5] for result in results if result[1] == block_size and result[3] == policy]
        cache_sizes_filtered = [result[0] for result in results if result[1] == block_size and result[3] == policy]
        ax.plot(cache_sizes_filtered, hit_rates, marker=marker, linestyle='-', label=f'Policy: {policy}', alpha=0.7)
    
    ax.set_xlabel('Cache Size (blocks)')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title(f'Cache Hit Rate for Block Size {block_size} bytes')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Plotting the comparison of different replacement policies using a bar plot
fig, ax = plt.subplots(figsize=(12, 8))

policy_hit_rates = {policy: [] for policy in replacement_policies}

for policy in replacement_policies:
    policy_hit_rates[policy] = [result[5] for result in results if result[3] == policy]

average_hit_rates = {policy: sum(policy_hit_rates[policy]) / len(policy_hit_rates[policy]) for policy in replacement_policies}

bars = ax.bar(average_hit_rates.keys(), average_hit_rates.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta'])

# Adding numeric values on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_xlabel('Replacement Policy')
ax.set_ylabel('Average Hit Rate (%)')
ax.set_title('Average Cache Hit Rate Comparison by Replacement Policy')
ax.grid(True)
plt.show()