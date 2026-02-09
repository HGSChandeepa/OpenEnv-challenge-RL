# visualize.py
"""
Visualization script for Supply Chain environment.
Creates plots to analyze agent performance.
"""

import matplotlib.pyplot as plt
from client import SupplyChainEnv
from models import SupplyChainAction


def visualize_episode(policy_fn, policy_name="Policy"):
    """
    Run an episode and create visualizations.
    
    Args:
        policy_fn: Function that takes observation and returns order_quantity
        policy_name: Name for the policy (for plot title)
    """
    env = SupplyChainEnv(base_url="http://localhost:8000")
    
    # Data collection
    days = []
    inventory_levels = []
    pending_orders = []
    demands = []
    orders = []
    rewards = []
    sold = []
    shortages = []
    
    try:
        obs = env.reset()
        
        for step in range(30):
            # Collect current state
            days.append(obs.observation.current_day)
            inventory_levels.append(obs.observation.current_inventory)
            pending_orders.append(obs.observation.pending_orders)
            demands.append(obs.observation.current_demand)
            
            # Execute policy
            order_qty = policy_fn(obs.observation)
            orders.append(order_qty)
            
            # Step environment
            result = env.step(SupplyChainAction(order_quantity=order_qty))
            
            rewards.append(result.reward)
            sold.append(result.observation.units_sold)
            shortages.append(result.observation.units_shortage)
            
            obs = result
            
            if result.done:
                break
        
    finally:
        env.close()
    
    # Create visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Supply Chain Performance - {policy_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Inventory over time
    ax = axes[0, 0]
    ax.plot(days, inventory_levels, label='Inventory', linewidth=2, color='blue')
    ax.axhline(y=50, color='green', linestyle='--', label='Target Level')
    ax.fill_between(days, 0, inventory_levels, alpha=0.3, color='blue')
    ax.set_xlabel('Day')
    ax.set_ylabel('Units')
    ax.set_title('Inventory Level Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Demand vs Sold
    ax = axes[0, 1]
    ax.plot(days, demands, label='Demand', linewidth=2, color='orange', marker='o')
    ax.plot(days, sold, label='Units Sold', linewidth=2, color='green', marker='s')
    ax.fill_between(days, sold, demands, where=[s < d for s, d in zip(sold, demands)], 
                     alpha=0.3, color='red', label='Shortage')
    ax.set_xlabel('Day')
    ax.set_ylabel('Units')
    ax.set_title('Demand vs Sales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Orders placed
    ax = axes[1, 0]
    ax.bar(days, orders, color='purple', alpha=0.7)
    ax.set_xlabel('Day')
    ax.set_ylabel('Units Ordered')
    ax.set_title('Order Quantities')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Pending orders
    ax = axes[1, 1]
    ax.plot(days, pending_orders, linewidth=2, color='brown', marker='d')
    ax.fill_between(days, 0, pending_orders, alpha=0.3, color='brown')
    ax.set_xlabel('Day')
    ax.set_ylabel('Units')
    ax.set_title('Pending Orders (In Transit)')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative reward
    ax = axes[2, 0]
    cumulative_rewards = [sum(rewards[:i+1]) for i in range(len(rewards))]
    ax.plot(days, cumulative_rewards, linewidth=2, color='darkgreen', marker='o')
    ax.fill_between(days, 0, cumulative_rewards, alpha=0.3, 
                     color='green' if cumulative_rewards[-1] > 0 else 'red')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Reward ($)')
    ax.set_title(f'Cumulative Reward (Total: ${cumulative_rewards[-1]:.2f})')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Daily metrics
    ax = axes[2, 1]
    ax.bar(days, rewards, color='teal', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Daily Reward ($)')
    ax.set_title('Daily Rewards')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Print summary statistics
    total_reward = sum(rewards)
    total_sold = sum(sold)
    total_shortage = sum(shortages)
    total_demand = sum(demands)
    fill_rate = (total_sold / total_demand * 100) if total_demand > 0 else 0
    avg_inventory = sum(inventory_levels) / len(inventory_levels)
    
    print(f"\n{'=' * 60}")
    print(f"EPISODE SUMMARY - {policy_name}")
    print(f"{'=' * 60}")
    print(f"Total Reward: ${total_reward:.2f}")
    print(f"Total Demand: {total_demand} units")
    print(f"Total Sold: {total_sold} units")
    print(f"Total Shortage: {total_shortage} units")
    print(f"Fill Rate: {fill_rate:.1f}%")
    print(f"Average Inventory: {avg_inventory:.1f} units")
    print(f"Total Orders Placed: {sum(orders)} units")
    print(f"{'=' * 60}\n")
    
    return fig


# Example policies
def heuristic_policy(obs):
    """Simple heuristic: order when total available is low."""
    total_available = obs.current_inventory + obs.pending_orders
    return 50 if total_available < 40 else 0


def conservative_policy(obs):
    """Conservative: maintain high inventory."""
    return 60 if obs.current_inventory < 80 else 0


def aggressive_policy(obs):
    """Aggressive: minimize inventory, just-in-time."""
    return 30 if obs.current_inventory < 20 else 0


if __name__ == "__main__":
    print("Starting visualization...")
    print("Make sure the server is running: uvicorn server.app:app")
    print()
    
    # Test different policies
    policies = [
        (heuristic_policy, "Heuristic Policy (Order when low)"),
        (conservative_policy, "Conservative Policy (High inventory)"),
        (aggressive_policy, "Aggressive Policy (JIT)"),
    ]
    
    for policy_fn, policy_name in policies:
        try:
            fig = visualize_episode(policy_fn, policy_name)
            # Save figure
            filename = f"supply_chain_{policy_name.lower().replace(' ', '_')}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"Error running {policy_name}: {e}")
    
    print("\nVisualization complete!")
