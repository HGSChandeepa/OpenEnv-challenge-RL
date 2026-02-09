# Supply Chain Environment

A simple reinforcement learning environment for supply chain management built with OpenEnv.

## Overview

Agents learn to optimize inventory ordering decisions by managing a single warehouse with:
- Limited storage capacity (200 units)
- Variable customer demand (10-30 units/day with weekend spikes)
- Supplier with 3-day lead time
- 30-day episodes

## State Space

The agent observes:
- `current_inventory`: Units in warehouse
- `pending_orders`: Units ordered but not delivered
- `current_demand`: Today's customer demand
- `days_until_delivery`: Days until next shipment
- `current_day`: Current simulation day (1-30)
- `units_sold`: Units fulfilled today
- `units_shortage`: Unmet demand today

## Action Space

- `order_quantity`: Integer 0-100 (units to order from supplier)

## Rewards

- **+$10** per unit sold (revenue)
- **-$5** per unit shortage (lost sales penalty)
- **-$2** per unit in inventory (holding cost)
- **-$1** per unit ordered (ordering cost)

## Installation

```bash
# Install dependencies
pip install -e .

# Or manually
pip install openenv @ git+https://github.com/meta-pytorch/OpenEnv.git
```

## Usage

### Running the Server Locally

```bash
# From the environment directory
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Using the Client

```python
from supply_chain_env import SupplyChainEnv, SupplyChainAction

# Connect to local server
env = SupplyChainEnv(base_url="http://localhost:8000")

with env:
    # Reset environment
    obs = env.reset()
    print(f"Starting inventory: {obs.observation.current_inventory}")
    
    # Take steps
    for day in range(30):
        # Simple policy: order when inventory is low
        if obs.observation.current_inventory < 30:
            order_qty = 50
        else:
            order_qty = 0
        
        result = env.step(SupplyChainAction(order_quantity=order_qty))
        
        print(f"Day {result.observation.current_day}: "
              f"Inventory={result.observation.current_inventory}, "
              f"Demand={result.observation.current_demand}, "
              f"Reward={result.reward:.2f}")
        
        if result.done:
            break
```

### Building Docker Image

```bash
# Build the image
docker build -f server/Dockerfile -t supply-chain-env:latest .

# Run the container
docker run -p 8000:8000 supply-chain-env:latest
```

### Using with OpenEnv CLI

```bash
# Validate environment
openenv validate

# Build Docker image
openenv build

# Push to Hugging Face
openenv push --repo-id your-username/supply-chain-env
```

## Environment Dynamics

### Demand Generation
- Base demand: 10-20 units/day
- Weekend spike: +10 units on Saturdays and Sundays
- Random variation: ±5 units

### Optimal Strategy
The agent must balance:
1. **Ordering enough** to meet demand (avoid shortages)
2. **Not over-ordering** to minimize holding costs
3. **Planning ahead** for 3-day delivery lead time
4. **Anticipating spikes** on weekends

## Example Scenarios

**Scenario 1: Under-ordering**
- Agent orders 10 units/day
- Weekend demand: 30+ units
- Result: Frequent stockouts, high shortage penalties

**Scenario 2: Over-ordering**
- Agent orders 100 units/day
- Average demand: 15 units/day
- Result: Warehouse fills up, high holding costs

**Scenario 3: Balanced (Target)**
- Agent monitors inventory and pending orders
- Orders 40-50 units when inventory < 30
- Skips ordering when inventory + pending > 70
- Result: ~90% service level, minimized costs

## Extending the Environment

Easy extensions for more complexity:
1. **Multiple suppliers** with different prices/lead times
2. **Multiple products** with shared warehouse space
3. **Perishable goods** with expiration dates
4. **Price elasticity** where demand responds to pricing
5. **Disruptions** (supplier delays, demand shocks)
6. **Multi-warehouse network** with transfer costs

## Training Tips

1. **Normalize observations** for better learning
2. **Curriculum learning**: Start with easier demand patterns
3. **Reward shaping**: Add intermediate rewards for good inventory levels
4. **Exploration**: Use epsilon-greedy or entropy bonuses
5. **Episode length**: Start with 10 days, increase to 30

## License

MIT License
