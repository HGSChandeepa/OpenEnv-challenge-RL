# Training RL Agents with TRL - Complete Guide

## 📚 Overview

This guide shows you **three different approaches** to train RL agents for the Supply Chain environment:

1. **Simple Policy Network** - Fast, efficient, traditional RL
2. **TRL with LLMs** - Use language models for reasoning
3. **Hybrid Approach** - Combine both methods

---

## 🎯 Approach 1: Simple Policy Network (Recommended to Start)

### Why Use This?
- ✅ Fast training (minutes instead of hours)
- ✅ Works well for numerical state spaces
- ✅ Lower compute requirements
- ✅ Easy to understand and debug

### Quick Start

```bash
# 1. Start the server
uvicorn server.app:app --ws-ping-interval 300 --ws-ping-timeout 300

# 2. Run training (in new terminal)
python train_with_trl.py simple
```

### How It Works

```python
# State: [inventory, pending, demand, days_until, current_day]
state = [0.25, 0.15, 0.30, 0.67, 0.03]  # Normalized values

# Neural Network: state → action_probabilities
policy_network(state) → [0.01, 0.02, ..., 0.15, ...]  # 101 probabilities (0-100)

# Sample action from distribution
action = sample(action_probabilities)  # e.g., 47

# Execute in environment
reward = env.step(action)

# Update policy using REINFORCE/PPO
policy_gradient_update(reward)
```

### Training Results

After 100 episodes, you should see:
```
Episode 0: $-250.00
Episode 10: $120.00
Episode 20: $380.00
Episode 50: $650.00
Episode 100: $780.00  ✅
```

---

## 🎯 Approach 2: TRL with Language Models

### Why Use This?
- ✅ Can reason about complex situations
- ✅ Handles multi-modal inputs (text + numbers)
- ✅ Can explain decisions in natural language
- ✅ Leverages pre-trained knowledge

### Setup

```bash
# Install TRL
pip install trl transformers torch datasets accelerate

# Start server with extended timeout (important for LLMs!)
uvicorn server.app:app --ws-ping-interval 300 --ws-ping-timeout 300
```

### How It Works

#### Step 1: Convert Environment to Text

```python
# Observation → Natural Language Prompt
prompt = """
You are a supply chain manager.

Current Status (Day 15/30):
- Inventory: 45 units
- Pending Orders: 30 units (arrive in 2 days)
- Today's Demand: 18 units

Economics:
- Revenue: $10/unit sold
- Shortage: -$5/unit
- Holding: -$2/unit in stock
- Ordering: -$1/unit ordered

How many units should you order? (0-100)
Respond with reasoning and decision.
"""
```

#### Step 2: LLM Generates Response

```python
llm_response = """
Analysis:
- Total available soon: 45 + 30 = 75 units
- Current demand: 18 units/day
- 75 units / 18 = ~4 days coverage

Strategy:
- We have good coverage for next few days
- Should order conservatively to minimize holding costs
- Weekend coming (days 20-21) may have higher demand

Decision: Order 25 units
"""
```

#### Step 3: Parse Action and Execute

```python
# Extract order quantity
action = parse_action(llm_response)  # → 25

# Execute in environment  
reward = env.step(action)  # → $156.00

# Use reward to update LLM policy (GRPO/PPO)
```

### Training with TRL GRPO

```python
from trl import GRPOConfig, GRPOTrainer

# Configure training
config = GRPOConfig(
    output_dir="./supply_chain_agent",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    max_length=512,
)

# Create trainer
trainer = GRPOTrainer(
    model=model,
    args=config,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# Train!
trainer.train()
```

### Expected Timeline

- **Data Collection**: 10-20 episodes (~30 min)
- **Training**: 3 epochs (~2-4 hours on GPU)
- **Evaluation**: 10 episodes (~15 min)

---

## 🎯 Approach 3: Hybrid (Best of Both)

### Idea

Use LLM for **strategic** decisions and neural network for **tactical** execution:

```python
# Strategic (every 7 days): LLM decides overall strategy
strategy = llm.decide("Should we stockpile for weekend rush?")

# Tactical (daily): Fast neural net executes based on strategy
action = policy_network(state, strategy_embedding)
```

### Benefits
- ✅ LLM handles complex reasoning
- ✅ Neural net provides fast execution
- ✅ Best of both worlds!

---

## 📊 Comparison Table

| Approach | Training Time | Compute | Interpretability | Performance |
|----------|--------------|---------|------------------|-------------|
| **Simple NN** | 30 min | Low (CPU ok) | Medium | Good (⭐⭐⭐⭐) |
| **TRL/LLM** | 2-4 hours | High (GPU needed) | High | Great (⭐⭐⭐⭐⭐) |
| **Hybrid** | 1-2 hours | Medium | High | Best (⭐⭐⭐⭐⭐) |

---

## 🚀 Step-by-Step: Training with TRL

### Step 1: Start the Server

```bash
# Terminal 1
cd supply_chain_env
uvicorn server.app:app --host 0.0.0.0 --port 8000 \
    --ws-ping-interval 300 --ws-ping-timeout 300
```

**Why the long timeout?** LLM token generation can take 30-60 seconds. Default WebSocket timeout (20s) is too short!

### Step 2: Collect Demonstrations

```bash
# Terminal 2
python trl_grpo_training.py collect
```

This generates:
- 20 episodes using heuristic policy
- ~600 (prompt, response, reward) tuples
- Saved to `supply_chain_demonstrations.json`

### Step 3: Review Demonstrations

```python
import json

with open('supply_chain_demonstrations.json') as f:
    demos = json.load(f)

print(f"Total examples: {len(demos)}")
print(f"Average reward: {sum(d['reward'] for d in demos) / len(demos)}")

# Look at a sample
print(demos[0]['prompt'])
print(demos[0]['response'])
print(demos[0]['reward'])
```

### Step 4: Train with TRL

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare dataset
dataset = Dataset.from_dict({
    'query': [d['prompt'] for d in demos],
    'response': [d['response'] for d in demos],
    'reward': [d['reward'] for d in demos],
})

# Train
config = GRPOConfig(
    output_dir="./supply_chain_grpo",
    num_train_epochs=3,
    learning_rate=1e-5,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./supply_chain_agent")
```

### Step 5: Test the Trained Agent

```bash
python trl_grpo_training.py test
```

### Step 6: Compare Performance

```python
# Baseline (Random)
python test_env.py random
# Expected: -$300 to $0

# Heuristic  
python test_env.py
# Expected: $400 to $700

# Trained LLM
python trl_grpo_training.py test
# Expected: $700 to $900
```

---

## 🔧 Troubleshooting

### Issue 1: WebSocket Timeout

**Error:**
```
websockets.exceptions.ConnectionClosedError: keepalive ping timeout
```

**Solution:**
```bash
# Increase timeout
uvicorn server.app:app --ws-ping-interval 300 --ws-ping-timeout 300
```

### Issue 2: Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Use smaller model: `gpt2` instead of `gpt2-large`
2. Reduce batch size: `per_device_train_batch_size=1`
3. Use gradient accumulation: `gradient_accumulation_steps=8`
4. Enable gradient checkpointing

### Issue 3: LLM Not Following Instructions

**Problem:** LLM generates random text instead of order quantities

**Solutions:**
1. Improve prompt structure (see examples)
2. Use instruction-tuned models (e.g., `Llama-3.2-1B-Instruct`)
3. Add more demonstration examples
4. Use constrained decoding

### Issue 4: Training is Slow

**Solutions:**
1. Start with simple policy network
2. Use smaller LLM (gpt2 vs Llama)
3. Reduce episode length (10 days instead of 30)
4. Use fewer demonstration episodes

---

## 📈 Advanced Tips

### 1. Curriculum Learning

Start easy, gradually increase difficulty:

```python
# Week 1: Easy mode
env = SupplyChainEnvironment(
    supplier_lead_time=1,  # Fast delivery
    episode_length=10,     # Short episodes
)

# Week 2: Medium
env = SupplyChainEnvironment(
    supplier_lead_time=3,
    episode_length=20,
)

# Week 3: Hard
env = SupplyChainEnvironment(
    supplier_lead_time=5,
    episode_length=30,
    # Add disruptions
)
```

### 2. Reward Shaping

Add intermediate rewards to guide learning:

```python
# Original reward
reward = revenue - shortage - holding - ordering

# Shaped reward (easier to learn)
reward += bonus_for_maintaining_target_inventory
reward += bonus_for_anticipating_demand_spikes
reward -= penalty_for_extreme_actions
```

### 3. Multi-Task Learning

Train on multiple scenarios:

```python
scenarios = [
    "normal_demand",
    "high_demand_weekend",
    "supplier_delay",
    "demand_spike",
    "seasonal_pattern",
]

for scenario in scenarios:
    # Train on each scenario
    # Agent learns to generalize
```

### 4. Human Feedback Integration

Combine RL with human preferences:

```python
# Collect human feedback on episodes
human_scores = collect_human_ratings(episodes)

# Use RLHF (Reinforcement Learning from Human Feedback)
train_with_rlhf(model, episodes, human_scores)
```

---

## 📝 For Your Hackathon Blog

### Suggested Structure

```markdown
# Training an LLM to Optimize Supply Chains with TRL

## The Challenge
- Supply chain optimization is a $X billion problem
- Traditional methods struggle with uncertainty
- Can LLMs learn to make better decisions?

## Our Approach
1. Created realistic supply chain environment (OpenEnv)
2. Collected expert demonstrations (heuristic policy)
3. Fine-tuned GPT-2 using TRL's GRPO algorithm
4. Evaluated against baselines

## Results
| Method | Avg Reward | Fill Rate | Cost Efficiency |
|--------|-----------|-----------|-----------------|
| Random | -$280 | 45% | Poor |
| Heuristic | $520 | 87% | Good |
| **LLM (Ours)** | **$840** | **94%** | **Excellent** |

## Key Insights
- LLMs learned to anticipate weekend demand spikes
- Natural language reasoning improves interpretability
- Hybrid approach achieves best results

## Code & Demo
- [Environment on HF Hub](link)
- [Training Code](link)
- [Interactive Demo](link)
```

---

## 🎓 Learning Resources

- **TRL Docs**: https://huggingface.co/docs/trl/
- **OpenEnv Docs**: https://meta-pytorch.org/OpenEnv/
- **RL Tutorial**: https://spinningup.openai.com/
- **TRL Examples**: https://github.com/huggingface/trl/tree/main/examples

---

## 🚀 Next Steps

1. ✅ Get simple policy network working
2. ✅ Collect demonstrations with heuristic
3. ✅ Train with TRL GRPO
4. ✅ Evaluate and compare approaches
5. ✅ Create visualizations for blog
6. ✅ Deploy to HuggingFace Spaces
7. ✅ Write compelling blog post
8. ✅ Submit to hackathon!

Good luck! 🎉
