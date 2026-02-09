# test_env.py
"""
Simple test script to demonstrate the Supply Chain environment.
Run this after starting the server with: uvicorn server.app:app
"""

from client import SupplyChainEnv
from models import SupplyChainAction


def run_random_policy():
    """Run a simple random policy."""
    import random
    
    print("=" * 60)
    print("Testing Supply Chain Environment - Random Policy")
    print("=" * 60)
    
    env = SupplyChainEnv(base_url="http://localhost:8000")
    
    try:
        reset_result = env.reset()
        obs = reset_result.observation
        print(f"\nEpisode started!")
        print(f"Initial inventory: {obs.current_inventory}")
        
        total_reward = 0
        
        for step in range(30):
            # Random order between 0 and 50
            order_qty = random.randint(0, 50)
            
            result = env.step(SupplyChainAction(order_quantity=order_qty))
            total_reward += result.reward
            
            obs = result.observation
            
            print(f"\nDay {obs.current_day}:")
            print(f"  Ordered: {order_qty} units")
            print(f"  Demand: {obs.current_demand} units")
            print(f"  Sold: {obs.units_sold} units")
            print(f"  Shortage: {obs.units_shortage} units")
            print(f"  Inventory: {obs.current_inventory} units")
            print(f"  Pending: {obs.pending_orders} units")
            print(f"  Reward: ${result.reward:.2f}")
            print(f"  Total Reward: ${total_reward:.2f}")
            
            if result.done:
                print(f"\nEpisode finished after {obs.current_day} days")
                break
        
        print(f"\n{'=' * 60}")
        print(f"Final Total Reward: ${total_reward:.2f}")
        print(f"{'=' * 60}")
        
    finally:
        env.close()


def run_simple_heuristic():
    """Run a simple heuristic policy: order when inventory is low."""
    
    print("\n" + "=" * 60)
    print("Testing Supply Chain Environment - Heuristic Policy")
    print("=" * 60)
    print("Strategy: Order 50 units when inventory + pending < 40")
    
    env = SupplyChainEnv(base_url="http://localhost:8000")
    
    try:
        reset_result = env.reset()
        obs = reset_result.observation
        print(f"\nEpisode started!")
        print(f"Initial inventory: {obs.current_inventory}")
        
        total_reward = 0
        total_sold = 0
        total_shortage = 0
        
        for step in range(30):
            # Heuristic: order when total available inventory is low
            total_available = obs.current_inventory + obs.pending_orders
            
            if total_available < 40:
                order_qty = 50
            else:
                order_qty = 0
            
            result = env.step(SupplyChainAction(order_quantity=order_qty))
            total_reward += result.reward
            total_sold += result.observation.units_sold
            total_shortage += result.observation.units_shortage
            
            obs = result.observation
            
            print(f"\nDay {obs.current_day}:")
            print(f"  Ordered: {order_qty} units")
            print(f"  Inventory: {obs.current_inventory} | Pending: {obs.pending_orders}")
            print(f"  Demand: {obs.current_demand} | Sold: {obs.units_sold} | Shortage: {obs.units_shortage}")
            print(f"  Step Reward: ${result.reward:.2f} | Cumulative: ${total_reward:.2f}")
            
            if result.done:
                print(f"\nEpisode finished after {obs.current_day} days")
                break
        
        print(f"\n{'=' * 60}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Reward: ${total_reward:.2f}")
        print(f"Total Units Sold: {total_sold}")
        print(f"Total Shortage: {total_shortage}")
        fill_rate = (total_sold / (total_sold + total_shortage) * 100) if (total_sold + total_shortage) > 0 else 0
        print(f"Fill Rate: {fill_rate:.1f}%")
        print(f"{'=' * 60}")
        
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "random":
        run_random_policy()
    else:
        run_simple_heuristic()
    
    print("\nTo test random policy, run: python test_env.py random")
