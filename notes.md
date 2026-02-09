## 1. models.py

- This file contains all the Action and the Observations.
- Here for actions use the Action
- Here for the observations we can use Observation

Here the action and its observation is bind togeather.
the action we use is ,

order_quantity for that action we can use the observations from the environment below.

1. current_inventory
2. pending_orders
3. current_demand
4. days_until_deliery
5. current_day
6. units_sold
7. units_shortage

## 2. server/supply_chain_environment.py

Here we design our environment fro the egent to act:The agent manages inventory by ordering from a supplier and fulfilling customer demand.

Here we need to design the following,

1. **init** (this will initialize the env)
2. reset() - this will reset the state
3. step() - this will execute one step of the environment and retun the reward performing the action.

### 3.server/app.py

The app.py file is the entry point for your FastAPI web server. It creates the web API that allows clients to interact with your RL environment over the network.
