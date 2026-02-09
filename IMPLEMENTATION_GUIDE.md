# Supply Chain Environment - Implementation Guide

## Architecture Overview

```
supply_chain_env/
├── __init__.py              # Package exports
├── models.py                # Action & Observation definitions (Pydantic)
├── client.py                # Client for interacting with environment
├── openenv.yaml            # Environment metadata
├── pyproject.toml          # Python dependencies
├── README.md               # User documentation
├── test_env.py             # Testing scripts
├── visualize.py            # Visualization utilities
└── server/
    ├── __init__.py
    ├── app.py              # FastAPI application
    ├── supply_chain_environment.py  # Core RL logic
    ├── requirements.txt    # Server dependencies
    └── Dockerfile          # Container configuration
```

## Core Components Explained

### 1. models.py - Data Structures

**Purpose**: Define the contract between agent and environment using Pydantic.

**SupplyChainAction**:
- `order_quantity`: The agent's decision (0-100 units)
- Inherits from `Action` base class
- Pydantic validation ensures valid inputs (ge=0 means >= 0)

**SupplyChainObservation**:
- Contains all information the agent can see
- Includes both state info (inventory, demand) and feedback (units_sold, reward)
- Inherits from `Observation` which provides `done` and `reward` fields

**Why Pydantic?**
- Automatic validation (catches invalid actions)
- Type safety (IDE autocomplete)
- Easy serialization to/from JSON
- OpenEnv uses it for all environments

### 2. supply_chain_environment.py - RL Logic

This is the heart of your environment. Let's break down each method:

#### `__init__()`
- Sets up environment parameters (warehouse size, lead times, etc.)
- Initializes random number generator for reproducibility
- Creates the State object to track episodes

#### `reset()`
- Called at the start of each episode
- Resets all state variables to initial conditions
- Returns the first observation
- **Key**: Generates a new episode_id so you can track different runs

#### `step(action)`
This is where the magic happens. The execution order matters:

1. **Process Order**: Agent's action is validated and queued for future delivery
2. **Receive Deliveries**: Check if any shipments arrived today
3. **Generate Demand**: Stochastic customer demand appears
4. **Fulfill Demand**: Sell from inventory (can't oversell)
5. **Calculate Reward**: Multi-objective reward function
6. **Check Done**: Episode ends after 30 days

**Why this order?**
- Realistic: You place orders before seeing today's demand
- Lead time simulation: Orders take 3 days to arrive
- Inventory dynamics: Must balance incoming/outgoing flows

#### `_generate_demand()`
Creates realistic demand patterns:
- Base demand: Random but bounded (10-20 units)
- Seasonality: Weekend spikes (every 7 days)
- Randomness: Daily variation (±5 units)

**Why this matters**: Agents must learn patterns, not just memorize sequences

#### `_calculate_reward()`
Multi-objective optimization:
- **Revenue** (+$10/unit): Encourages fulfilling demand
- **Shortage penalty** (-$5/unit): Penalizes stockouts
- **Holding cost** (-$2/unit): Penalizes excess inventory
- **Ordering cost** (-$1/unit): Encourages efficient ordering

**Design choice**: These coefficients balance different objectives. Tune them to change difficulty!

### 3. app.py - FastAPI Server

```python
app = create_app(
    SupplyChainEnvironment,    # Environment CLASS (not instance)
    SupplyChainAction,         # Action type
    SupplyChainObservation,    # Observation type
    env_name="supply_chain"    # Name for API
)
```

**Why pass the class?**
- OpenEnv uses WebSockets for persistent connections
- Each client gets its own environment instance
- Enables parallel training without state conflicts

**What does create_app do?**
- Creates FastAPI routes: `/reset`, `/step`, `/state`, `/health`
- Handles WebSocket connections
- Manages environment lifecycle (creation/destruction)
- Serializes Pydantic models to JSON

### 4. client.py - User Interface

**SupplyChainEnv** extends `EnvClient` which provides:
- Connection management (WebSocket)
- Automatic serialization/deserialization
- Context manager support (`with` statement)
- Integration with Docker/HuggingFace

**Two key methods to implement**:

#### `_step_payload()`
Converts Python action object → JSON for server:
```python
SupplyChainAction(order_quantity=50) 
→ {"order_quantity": 50}
```

#### `_parse_result()`
Converts server JSON → Python observation object:
```python
{"observation": {...}, "reward": 123.4, "done": false}
→ StepResult(observation=SupplyChainObservation(...), ...)
```

**Why separate these?**
- Clean separation: network protocol vs business logic
- Type safety: Client code uses typed objects, not raw dicts
- Testability: Can mock server responses

### 5. Dockerfile - Deployment

Multi-stage build for efficiency:

**Stage 1 (builder)**:
- Install dependencies
- Build the environment package

**Stage 2 (runtime)**:
- Copy built artifacts
- Set up minimal runtime
- Configure health checks

**Key configurations**:
```dockerfile
CMD ["uvicorn", "server.app:app", 
     "--ws-ping-interval", "300",  # 5 min timeout for RL training
     "--ws-ping-timeout", "300"]
```

**Why these timeouts?**
- Default (20s) is too short for LLM-based agents
- Token generation can take minutes
- Prevents connection drops during training

## Data Flow Diagram

```
┌─────────────┐
│   Client    │
│  (Python)   │
└──────┬──────┘
       │ WebSocket
       │ 1. reset()
       ▼
┌─────────────────────────────────┐
│      FastAPI Server             │
│  ┌──────────────────────────┐  │
│  │  SupplyChainEnvironment  │  │
│  │                          │  │
│  │  1. Reset state          │  │
│  │  2. Generate demand      │  │
│  │  3. Return observation   │  │
│  └──────────────────────────┘  │
└──────┬──────────────────────────┘
       │ JSON: {observation: {...}}
       ▼
┌─────────────┐
│   Client    │
│  obs.current_inventory = 50   │
└──────┬──────┘
       │ 2. step(action)
       │ JSON: {order_quantity: 50}
       ▼
┌─────────────────────────────────┐
│      FastAPI Server             │
│  ┌──────────────────────────┐  │
│  │  SupplyChainEnvironment  │  │
│  │                          │  │
│  │  1. Process order        │  │
│  │  2. Receive deliveries   │  │
│  │  3. Generate demand      │  │
│  │  4. Fulfill demand       │  │
│  │  5. Calculate reward     │  │
│  └──────────────────────────┘  │
└──────┬──────────────────────────┘
       │ JSON: {observation: {...}, reward: 123.4}
       ▼
┌─────────────┐
│   Client    │
│  result.reward = 123.4        │
└─────────────┘
```

## Testing Workflow

### 1. Local Testing (No Docker)

```bash
# Terminal 1: Start server
cd supply_chain_env
uvicorn server.app:app --reload

# Terminal 2: Test client
python test_env.py
```

### 2. Docker Testing

```bash
# Build image
docker build -f server/Dockerfile -t supply-chain:latest .

# Run container
docker run -p 8000:8000 supply-chain:latest

# Test from host
python test_env.py
```

### 3. OpenEnv CLI

```bash
# Validate structure
openenv validate

# Build (auto-detects context)
openenv build

# Push to HuggingFace
openenv push --repo-id username/supply-chain-env
```

## Common Pitfalls & Solutions

### Pitfall 1: Importing Issues
**Problem**: `ModuleNotFoundError: No module named 'models'`

**Solution**: 
- Ensure `PYTHONPATH` includes the env directory
- Use relative imports in server: `from ..models import`
- Install in editable mode: `pip install -e .`

### Pitfall 2: WebSocket Timeouts
**Problem**: Connection drops during training

**Solution**:
```dockerfile
CMD ["uvicorn", "server.app:app", 
     "--ws-ping-interval", "300", 
     "--ws-ping-timeout", "300"]
```

### Pitfall 3: State Leaking Between Episodes
**Problem**: reset() doesn't fully reset state

**Solution**:
```python
def reset(self):
    # Create NEW State object
    self._state = State(episode_id=str(uuid4()), step_count=0)
    
    # Reset ALL environment variables
    self.current_inventory = 50
    self.pending_shipments = []  # Empty list, not clear()
    self.current_day = 1
```

### Pitfall 4: Reward Not Learning
**Problem**: Agent doesn't improve

**Possible causes**:
1. Reward scale too large/small → normalize to [-1, 1]
2. Reward too sparse → add dense intermediate rewards
3. Episode too long → start with 10 days, increase gradually
4. Action space too large → start with 3 discrete actions (low/med/high)

## Next Steps for Hackathon

### 1. Make it Impressive
- Add **visualizations** (dashboard, plots)
- Create **pre-trained baseline** agents
- Design **challenge scenarios** (disruptions, demand spikes)

### 2. Technical Excellence
- Implement **vectorized environments** for faster training
- Add **configuration options** (easy/medium/hard difficulty)
- Include **unit tests** for environment mechanics
- Optimize for **GPU training** if using neural networks

### 3. Storytelling (Blog)
- Show **failure case** (random agent)
- Demonstrate **learning curve**
- Explain **emergent behaviors** (anticipating weekends)
- Compare **different strategies** (JIT vs conservative)

### 4. Green Agent Wrapper
- Implement base agent class
- Provide example agents (random, heuristic, learned)
- Make it easy for others to test their agents

Good luck with your hackathon! 🚀
