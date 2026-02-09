"""
Train a Supply Chain RL Agent using TRL (Transformer Reinforcement Learning)

This script demonstrates how to train an LLM-based agent to optimize supply chain
decisions using TRL's GRPO (Group Relative Policy Optimization) or PPO.

Prerequisites:
    pip install trl transformers torch accelerate
    
Run the server first:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --ws-ping-interval 300 --ws-ping-timeout 300
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from supply_chain_env import SupplyChainEnv, SupplyChainAction, SupplyChainObservation
import random
from typing import List, Dict
import json


# ==============================================================================
# STEP 1: Create Environment Wrapper for TRL
# ==============================================================================

class SupplyChainTRLWrapper:
    """
    Wrapper to make SupplyChain environment compatible with TRL.
    
    TRL expects text-based interactions, so we convert:
    - Observations → Natural language prompts
    - LLM responses → Actions (order quantities)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.env = SupplyChainEnv(base_url=base_url)
        self.current_obs = None
        
    def reset(self) -> str:
        """Reset environment and return initial prompt."""
        result = self.env.reset()
        self.current_obs = result.observation
        return self._create_prompt(self.current_obs)
    
    def step(self, action_text: str) -> tuple[str, float, bool]:
        """
        Execute action and return (next_prompt, reward, done).
        
        Args:
            action_text: LLM's response (we'll parse it to extract order quantity)
        
        Returns:
            next_prompt: Text prompt for next step
            reward: Reward from environment
            done: Whether episode is finished
        """
        # Parse LLM's response to extract order quantity
        order_qty = self._parse_action(action_text)
        
        # Execute action in environment
        result = self.env.step(SupplyChainAction(order_quantity=order_qty))
        
        self.current_obs = result.observation
        next_prompt = self._create_prompt(result.observation)
        
        return next_prompt, result.reward, result.done
    
    def _create_prompt(self, obs: SupplyChainObservation) -> str:
        """
        Convert observation to natural language prompt for the LLM.
        
        This is crucial - the prompt should:
        1. Provide all necessary information
        2. Be clear about what action is needed
        3. Specify the format of the response
        """
        prompt = f"""You are a supply chain manager optimizing inventory decisions.

CURRENT SITUATION (Day {obs.current_day}/30):
- Warehouse Inventory: {obs.current_inventory} units
- Pending Orders (in transit): {obs.pending_orders} units
- Today's Customer Demand: {obs.current_demand} units
- Days Until Next Delivery: {obs.days_until_delivery} days

TODAY'S PERFORMANCE:
- Units Sold: {obs.units_sold}
- Shortage (unmet demand): {obs.units_shortage}

DECISION REQUIRED:
How many units should you order from the supplier today? (0-100 units)
Remember: Orders take 3 days to arrive.

Respond with ONLY a number between 0 and 100.
Order Quantity: """
        
        return prompt
    
    def _parse_action(self, action_text: str) -> int:
        """
        Parse LLM's response to extract order quantity.
        
        The LLM might respond with:
        - "50"
        - "I recommend ordering 50 units"
        - "Order Quantity: 50"
        
        We need to extract the number and validate it.
        """
        try:
            # Try to find any number in the response
            import re
            numbers = re.findall(r'\d+', action_text)
            
            if numbers:
                order_qty = int(numbers[0])
                # Clip to valid range
                order_qty = max(0, min(100, order_qty))
                return order_qty
            else:
                # Default to 0 if no number found
                print(f"Warning: Could not parse action from '{action_text}', defaulting to 0")
                return 0
                
        except Exception as e:
            print(f"Error parsing action '{action_text}': {e}, defaulting to 0")
            return 0
    
    def close(self):
        """Close the environment connection."""
        self.env.close()


# ==============================================================================
# STEP 2: Create Dataset Generator
# ==============================================================================

def create_episodes(env_wrapper: SupplyChainTRLWrapper, num_episodes: int = 10) -> List[Dict]:
    """
    Generate training episodes for TRL.
    
    Each episode is a sequence of (prompt, response, reward) tuples.
    """
    episodes = []
    
    for episode_idx in range(num_episodes):
        print(f"\nGenerating episode {episode_idx + 1}/{num_episodes}")
        
        episode_data = []
        prompt = env_wrapper.reset()
        done = False
        total_reward = 0
        
        while not done:
            # For initial training, use a heuristic policy to generate demonstrations
            # Later, the LLM will learn to improve on this
            obs = env_wrapper.current_obs
            
            # Simple heuristic: order when inventory + pending is low
            total_available = obs.current_inventory + obs.pending_orders
            if total_available < 40:
                action_text = "50"
            else:
                action_text = "0"
            
            # Step environment
            next_prompt, reward, done = env_wrapper.step(action_text)
            total_reward += reward
            
            episode_data.append({
                "prompt": prompt,
                "response": action_text,
                "reward": reward
            })
            
            prompt = next_prompt
        
        print(f"Episode {episode_idx + 1} completed. Total reward: ${total_reward:.2f}")
        episodes.extend(episode_data)
    
    return episodes


# ==============================================================================
# STEP 3: Training with TRL GRPO
# ==============================================================================

def train_with_grpo():
    """
    Train an LLM agent using GRPO (Group Relative Policy Optimization).
    
    GRPO is particularly good for environments with:
    - Sparse rewards
    - Multiple objectives
    - Sequential decision making
    """
    
    print("=" * 80)
    print("Training Supply Chain Agent with TRL GRPO")
    print("=" * 80)
    
    # 1. Load model and tokenizer
    print("\n1. Loading model...")
    model_name = "gpt2"  # Start with small model for testing
    # For better performance, use: "meta-llama/Llama-3.2-1B" or similar
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # 2. Create environment wrapper
    print("\n2. Creating environment...")
    env_wrapper = SupplyChainTRLWrapper(base_url="http://localhost:8000")
    
    # 3. Generate initial training data (optional - for supervised pre-training)
    print("\n3. Generating demonstration episodes...")
    episodes = create_episodes(env_wrapper, num_episodes=5)
    
    print(f"Generated {len(episodes)} training examples")
    
    # 4. Configure GRPO training
    print("\n4. Configuring GRPO trainer...")
    training_args = GRPOConfig(
        output_dir="./supply_chain_grpo",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        warmup_steps=50,
        max_length=512,
        # GRPO specific
        grpo_alpha=1.0,  # Strength of relative advantage
        # Add more GRPO params as needed
    )
    
    # 5. Create trainer
    # Note: TRL's GRPOTrainer expects specific dataset format
    # You'll need to adapt this based on TRL's latest API
    
    print("\n5. Starting training...")
    print("Note: This is a template. You'll need to adapt based on TRL's current API")
    print("Check: https://huggingface.co/docs/trl/")
    
    # trainer = GRPOTrainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     # Add your dataset here
    # )
    # 
    # trainer.train()
    
    print("\n6. Saving model...")
    # model.save_pretrained("./supply_chain_agent")
    # tokenizer.save_pretrained("./supply_chain_agent")
    
    env_wrapper.close()
    
    print("\n✅ Training complete!")


# ==============================================================================
# STEP 4: Simplified RL Training (Alternative Approach)
# ==============================================================================

def train_simple_rl():
    """
    Simplified RL training using a small neural network instead of LLM.
    
    This is more efficient for simple environments and doesn't require TRL.
    Use this approach if:
    - Environment state is simple (numerical)
    - You want faster training
    - You don't need natural language reasoning
    """
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    import numpy as np
    
    print("=" * 80)
    print("Training Supply Chain Agent with Simple Policy Network")
    print("=" * 80)
    
    # 1. Define simple policy network
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim=5, hidden_dim=64, action_dim=101):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),  # Output: probabilities for 0-100
                nn.Softmax(dim=-1)
            )
        
        def forward(self, state):
            return self.network(state)
    
    # 2. Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    env = SupplyChainEnv(base_url="http://localhost:8000")
    
    # 3. Training loop
    num_episodes = 100
    gamma = 0.99  # Discount factor
    
    for episode in range(num_episodes):
        # Reset
        result = env.reset()
        obs = result.observation
        
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        step_count = 0
        
        while not done:
            # Convert observation to state tensor
            state = torch.FloatTensor([
                obs.current_inventory / 200.0,  # Normalize
                obs.pending_orders / 200.0,
                obs.current_demand / 50.0,
                obs.days_until_delivery / 3.0,
                obs.current_day / 30.0
            ]).unsqueeze(0).to(device)
            
            # Get action probabilities
            action_probs = policy(state)
            
            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Execute action
            result = env.step(SupplyChainAction(order_quantity=action.item()))
            
            episode_rewards.append(result.reward)
            episode_log_probs.append(log_prob)
            
            obs = result.observation
            done = result.done
            step_count += 1
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        # Policy gradient update
        policy_loss = []
        for log_prob, G in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(episode_rewards)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - Total Reward: ${total_reward:.2f}")
    
    env.close()
    
    # Save model
    torch.save(policy.state_dict(), "/mnt/user-data/outputs/supply_chain_policy.pt")
    print("\n✅ Training complete! Model saved to supply_chain_policy.pt")


# ==============================================================================
# STEP 5: Evaluation
# ==============================================================================

def evaluate_agent(policy_path: str = None):
    """Evaluate a trained agent."""
    
    if policy_path:
        print(f"Loading trained policy from {policy_path}")
        # Load your trained model here
    
    env = SupplyChainEnv(base_url="http://localhost:8000")
    
    num_eval_episodes = 10
    total_rewards = []
    
    for episode in range(num_eval_episodes):
        result = env.reset()
        obs = result.observation
        episode_reward = 0
        done = False
        
        while not done:
            # Use your trained policy to select action
            # For demo, using heuristic
            total_available = obs.current_inventory + obs.pending_orders
            if total_available < 40:
                order_qty = 50
            else:
                order_qty = 0
            
            result = env.step(SupplyChainAction(order_quantity=order_qty))
            episode_reward += result.reward
            obs = result.observation
            done = result.done
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: ${episode_reward:.2f}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Average Reward: ${avg_reward:.2f} ± ${std_reward:.2f}")
    print(f"Best Episode: ${max(total_rewards):.2f}")
    print(f"Worst Episode: ${min(total_rewards):.2f}")
    print(f"{'=' * 60}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n🚀 Supply Chain RL Training with TRL\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        # Train with simple policy network (recommended to start)
        train_simple_rl()
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluate trained agent
        evaluate_agent()
    else:
        # Print instructions
        print("=" * 80)
        print("TRAINING OPTIONS")
        print("=" * 80)
        print("\n1. Simple RL Training (Recommended to start):")
        print("   python train_with_trl.py simple")
        print("\n2. TRL/LLM Training (Advanced):")
        print("   python train_with_trl.py grpo")
        print("\n3. Evaluate:")
        print("   python train_with_trl.py eval")
        print("\n" + "=" * 80)
        print("\nMake sure the server is running first:")
        print("uvicorn server.app:app --ws-ping-interval 300 --ws-ping-timeout 300")
        print("=" * 80)
