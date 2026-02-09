# Quick Start Guide

## Setup in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install "openenv @ git+https://github.com/meta-pytorch/OpenEnv.git"
pip install fastapi uvicorn pydantic websockets matplotlib
```

### Step 2: Start the Server
```bash
cd supply_chain_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 3: Test the Environment
Open a new terminal and run:
```bash
python test_env.py
```

You should see output like:
```
============================================================
Testing Supply Chain Environment - Heuristic Policy
============================================================
Strategy: Order 50 units when inventory + pending < 40

Episode started!
Initial inventory: 50

Day 1:
  Ordered: 0 units
  Inventory: 50 | Pending: 0
  Demand: 15 | Sold: 15 | Shortage: 0
  Step Reward: $80.00 | Cumulative: $80.00
...
```

### Step 4: Visualize Performance
```bash
python visualize.py
```

This creates plots showing inventory, demand, and rewards over time.

## What Just Happened?

1. **Server** (`uvicorn server.app:app`) started a FastAPI server with WebSocket support
2. **Client** (`test_env.py`) connected via WebSocket and ran an episode
3. **Environment** simulated 30 days of supply chain operations
4. **Agent** (simple heuristic) made ordering decisions
5. **Visualization** plotted the results

## Next Steps

### For the Hackathon:

1. **Enhance the environment**:
   - Add more complexity (multiple suppliers, products)
   - Implement disruption events
   - Add realistic scenarios

2. **Train an RL agent**:
   ```python
   from supply_chain_env import SupplyChainEnv, SupplyChainAction
   import torch
   
   env = SupplyChainEnv(base_url="http://localhost:8000")
   # Train with your favorite RL library (TRL, stable-baselines3, etc.)
   ```

3. **Build with OpenEnv CLI**:
   ```bash
   openenv build
   openenv validate
   openenv push --repo-id your-username/supply-chain-env
   ```

4. **Write your blog**:
   - Explain the problem
   - Show baseline performance
   - Demonstrate learning
   - Highlight emergent behaviors

## Debugging Tips

**Server won't start?**
- Check port 8000 is free: `lsof -i :8000`
- Check imports: `python -c "from server.app import app"`

**Client can't connect?**
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall settings

**WebSocket timeout?**
- Already configured in Dockerfile (300s timeout)
- For local testing, add: `--ws-ping-interval 300 --ws-ping-timeout 300`

**Import errors?**
- Install in editable mode: `pip install -e .`
- Or set PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)`

## File Overview

```
supply_chain_env/
├── models.py                    # ✅ Action & Observation definitions
├── client.py                    # ✅ Client for connecting to environment
├── server/
│   ├── app.py                   # ✅ FastAPI application
│   ├── supply_chain_environment.py  # ✅ Core RL logic
│   ├── Dockerfile               # ✅ Container configuration
│   └── requirements.txt         # ✅ Server dependencies
├── openenv.yaml                 # ✅ Environment metadata
├── pyproject.toml               # ✅ Package configuration
├── test_env.py                  # 🧪 Testing script
├── visualize.py                 # 📊 Visualization script
├── README.md                    # 📖 User documentation
└── IMPLEMENTATION_GUIDE.md      # 📚 Developer guide
```

Happy hacking! 🎉
