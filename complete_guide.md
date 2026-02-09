# Supply Chain OpenEnv Environment - Complete Implementation

## 📦 What You Got

A **production-ready OpenEnv environment** for reinforcement learning in supply chain management, complete with:

✅ Full OpenEnv compliance (models, server, client)
✅ Docker configuration for deployment
✅ Testing and visualization scripts
✅ Comprehensive documentation
✅ Ready for HuggingFace deployment

---

## 🎯 What This Environment Does

### The Challenge
Manage a warehouse by making daily ordering decisions to:
- **Fulfill customer demand** (maximize sales)
- **Minimize costs** (inventory, ordering, shortages)
- **Plan ahead** for 3-day supplier lead time

### The Agent's Task
Every day for 30 days:
1. **Observe**: Current inventory, pending orders, today's demand
2. **Decide**: How many units to order (0-100)
3. **Receive**: Reward based on sales, costs, and service level

### Why It's a Good RL Problem
- **Delayed consequences**: Orders arrive 3 days later
- **Uncertainty**: Random demand with patterns
- **Multi-objective**: Balance competing costs
- **Realistic**: Real supply chain dynamics

---

## 🏗️ Architecture Explained

### 1. **models.py** - Data Contracts
```python
SupplyChainAction:
  - order_quantity: int (0-100)

SupplyChainObservation:
  - current_inventory: int
  - pending_orders: int
  - current_demand: int
  - days_until_delivery: int
  - current_day: int
  - units_sold: int
  - units_shortage: int
  - reward: float
  - done: bool
```

**Purpose**: Define what the agent sees and does using Pydantic for type safety.

### 2. **server/supply_chain_environment.py** - Core Logic

**Key Methods**:
- `reset()`: Initialize new episode (50 starting inventory, day 1)
- `step(action)`: Execute one day simulation
  1. Process agent's order
  2. Receive deliveries (after 3-day lead time)
  3. Generate customer demand
  4. Fulfill demand from inventory
  5. Calculate multi-objective reward
  6. Return observation

**Reward Function**:
```python
reward = (units_sold × $10)           # Revenue
       - (units_shortage × $5)        # Lost sales penalty
       - (inventory_level × $2)       # Holding cost
       - (order_quantity × $1)        # Ordering cost
```

**Demand Generation**:
- Base: 10-20 units/day (random)
- Weekends: +10 units (days 6, 7, 13, 14, etc.)
- Variation: ±5 units daily noise

### 3. **server/app.py** - FastAPI Server
```python
app = create_app(
    SupplyChainEnvironment,      # Pass CLASS not instance
    SupplyChainAction,
    SupplyChainObservation,
    env_name="supply_chain"
)
```

**What it provides**:
- WebSocket endpoints for persistent connections
- Auto-routes: `/reset`, `/step`, `/state`, `/health`
- Session management (one environment per client)
- JSON serialization/deserialization

### 4. **client.py** - User Interface
```python
class SupplyChainEnv(EnvClient):
    def _step_payload(self, action):
        # Convert Action object → JSON
        
    def _parse_result(self, payload):
        # Convert JSON → Observation object
```

**Features**:
- Type-safe API (autocomplete in IDEs)
- Context manager support (`with env:`)
- Connection from Docker/HuggingFace/localhost
- Automatic reconnection handling

### 5. **server/Dockerfile** - Deployment
```dockerfile
FROM python:3.11-slim
# Install dependencies
# Copy environment code
# Run uvicorn with 5-minute WebSocket timeout
CMD ["uvicorn", "server.app:app", 
     "--ws-ping-interval", "300",
     "--ws-ping-timeout", "300"]
```

---

## 🚀 Quick Start

### Option 1: Local Development (No Docker)

```bash
# Terminal 1: Start server
cd supply_chain_env
pip install -e .
uvicorn server.app:app

# Terminal 2: Test
python test_env.py
python visualize.py
```

### Option 2: Docker Build

```bash
docker build -f server/Dockerfile -t supply-chain:latest .
docker run -p 8000:8000 supply-chain:latest
python test_env.py
```

### Option 3: OpenEnv CLI (Recommended for Hackathon)

```bash
cd supply_chain_env
openenv validate          # Check structure
openenv build            # Build Docker image
openenv push --repo-id username/supply-chain-env  # Deploy to HF
```

---

## 📊 Example Usage

```python
from supply_chain_env import SupplyChainEnv, SupplyChainAction

# Connect to environment
env = SupplyChainEnv(base_url="http://localhost:8000")

with env:
    # Reset
    obs = env.reset()
    print(f"Starting inventory: {obs.observation.current_inventory}")
    
    # Run episode
    total_reward = 0
    for day in range(30):
        # Simple policy: order when low
        if obs.observation.current_inventory < 30:
            action = SupplyChainAction(order_quantity=50)
        else:
            action = SupplyChainAction(order_quantity=0)
        
        result = env.step(action)
        total_reward += result.reward
        
        print(f"Day {result.observation.current_day}: "
              f"Inv={result.observation.current_inventory}, "
              f"Demand={result.observation.current_demand}, "
              f"Reward=${result.reward:.2f}")
        
        obs = result
        if result.done:
            break
    
    print(f"Total Reward: ${total_reward:.2f}")
```

---

## 🎓 Key Learning Points

### 1. **State Space Design**
- Include enough info for optimal decisions (inventory + pending)
- Add temporal features (days_until_delivery)
- Provide feedback signals (units_sold, units_shortage)

### 2. **Action Space**
- Start simple (single continuous value)
- Can extend to: order timing, supplier choice, pricing

### 3. **Reward Engineering**
- Multi-objective balancing is key
- Coefficients matter ($10 vs $5 vs $2)
- Tune for desired behavior (service level vs cost)

### 4. **Episode Design**
- 30 days = enough to see lead time effects
- Fixed length = easier learning
- Can extend with variable termination

### 5. **Stochasticity**
- Random demand = agent must generalize
- Patterns (weekends) = agent can learn structure
- Reproducible via seeding

---

## 🏆 For the Hackathon

### Required Deliverables ✅

1. **Environment on HF Hub**: Use `openenv push`
2. **Training notebooks/scripts**: Use test_env.py or create your own
3. **Blog on HuggingFace**: Structure provided below

### Making it Stand Out 🌟

**1. Technical Excellence**:
- Add difficulty levels (easy/medium/hard)
- Implement vectorized environments for parallel training
- Create comprehensive test suite
- Optimize for GPU if using neural networks

**2. Creative Use of OpenEnv**:
- Multi-agent variant (competing retailers)
- Hierarchical decisions (strategic + operational)
- Real-time visualization during training
- Integration with LLM for reasoning

**3. Compelling Story**:
- Show failure modes (random, greedy agents)
- Demonstrate learning curve
- Explain emergent behaviors (anticipating weekends)
- Compare strategies (JIT vs conservative)

**4. Green Agent Wrapper**:
```python
class SupplyChainAgent:
    def select_action(self, obs: SupplyChainObservation) -> SupplyChainAction:
        # Base class for agents
        pass

class HeuristicAgent(SupplyChainAgent):
    # Simple baseline
    
class RLAgent(SupplyChainAgent):
    # Learned policy
```

---

## 📝 Suggested Blog Structure

```markdown
# Supply Chain Optimization with Reinforcement Learning

## Introduction
- Real-world problem: $X billion lost to inventory inefficiency
- Our environment simulates realistic supply chain dynamics

## The Environment
- State/action/reward spaces
- Environment dynamics (lead times, demand patterns)
- Visualization of one episode

## The Challenge
- Show failure: random agent gets -$500 reward
- Show naive: greedy agent gets $200 reward
- Optimal strategy preview

## Our Approach
[Your RL algorithm here]
- Why we chose this
- Hyperparameters
- Training setup

## Results
- Learning curves
- Final performance: $800+ reward
- Comparison table: Random vs Heuristic vs Learned
- Emergent behaviors: Agent learns to anticipate weekends!

## Analysis
- What the agent learned
- Failure cases
- Ablation studies (what if no lead time?)

## Extensions & Future Work
- Multi-product inventory
- Network of warehouses
- Real-world deployment considerations

## Try It Yourself
[Link to HF Space]
[Link to GitHub]
[Link to Colab notebook]
```

---

## 🔧 Common Issues & Solutions

### Import Errors
```bash
# Solution 1: Install in editable mode
pip install -e .

# Solution 2: Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### WebSocket Timeouts
Already configured in Dockerfile (300s). For local testing:
```bash
uvicorn server.app:app --ws-ping-interval 300 --ws-ping-timeout 300
```

### Port Already in Use
```bash
# Find process
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
uvicorn server.app:app --port 8001
```

---

## 📚 Files Overview

```
supply_chain_env/
├── 📄 QUICKSTART.md              ← Start here!
├── 📄 README.md                  ← User documentation
├── 📄 IMPLEMENTATION_GUIDE.md    ← Deep dive explanation
├── 📄 openenv.yaml               ← Environment metadata
├── 📄 pyproject.toml             ← Python package config
├── 🐍 __init__.py                ← Package exports
├── 🐍 models.py                  ← Action/Observation definitions
├── 🐍 client.py                  ← Environment client
├── 🐍 test_env.py                ← Testing script
├── 🐍 visualize.py               ← Plotting utilities
└── 📁 server/
    ├── 🐍 __init__.py
    ├── 🐍 app.py                 ← FastAPI application
    ├── 🐍 supply_chain_environment.py  ← Core RL logic
    ├── 📄 requirements.txt       ← Server dependencies
    └── 🐳 Dockerfile             ← Container config
```

---

## 🎯 Next Steps

1. **Test locally**: Run `test_env.py` to verify everything works
2. **Build Docker**: `openenv build` 
3. **Deploy to HF**: `openenv push --repo-id your-username/supply-chain-env`
4. **Train an agent**: Use TRL, stable-baselines3, or your own RL code
5. **Visualize results**: Use `visualize.py` or create custom plots
6. **Write blog**: Document your journey and results
7. **Submit**: Add to HF Hub and publish blog

---

## 💡 Extension Ideas

**Easy**:
- Multiple difficulty levels (tune demand variance, lead times)
- Different demand distributions (seasonal, trending)
- Scoring system (gold/silver/bronze based on reward)

**Medium**:
- Multiple products sharing warehouse space
- Different suppliers with price/speed tradeoffs
- Disruption events (supplier delays, demand shocks)

**Hard**:
- Multi-warehouse network with transfer costs
- Competitive multi-agent (multiple retailers)
- Partial observability (noisy demand forecasts)
- Real-time pricing decisions

**Expert**:
- Integration with real supply chain data
- LLM-based reasoning (explain decisions)
- Multi-modal observations (text + numerical)
- Hierarchical RL (strategic vs operational)

---

## 📞 Support

- **OpenEnv Docs**: https://meta-pytorch.org/OpenEnv/
- **OpenEnv GitHub**: https://github.com/meta-pytorch/OpenEnv
- **This code**: Check IMPLEMENTATION_GUIDE.md for details

Good luck with your hackathon! 🚀