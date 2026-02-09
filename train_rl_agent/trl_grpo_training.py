"""
Complete TRL GRPO Training Example for Supply Chain Environment

This notebook shows how to train an LLM to make supply chain decisions using TRL.

Setup:
1. Install dependencies:
   pip install trl transformers torch datasets accelerate
   
2. Start the supply chain server:
   uvicorn server.app:app --host 0.0.0.0 --port 8000 --ws-ping-interval 300 --ws-ping-timeout 300

3. Run this script:
   python trl_grpo_training.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from supply_chain_env import SupplyChainEnv, SupplyChainAction
import json
from typing import List, Dict
import re



# PART 1: Environment Integration with TRL


class SupplyChainRLEnvironment:
    """
    RL Environment wrapper that works with TRL.
    
    TRL expects:
    - generate_prompt(state) -> str
    - parse_response(text) -> action
    - step(action) -> (next_state, reward, done, info)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.env = SupplyChainEnv(base_url=base_url)
        self.current_obs = None
        
    def reset(self):
        """Reset and return initial state."""
        result = self.env.reset()
        self.current_obs = result.observation
        return self.current_obs
    
    def get_prompt(self) -> str:
        """Generate prompt for LLM based on current observation."""
        obs = self.current_obs
        
        # Create a structured prompt
        prompt = f"""<|system|>
You are an expert supply chain manager optimizing inventory decisions.
Your goal is to maximize profit by balancing sales revenue against inventory costs.

<|user|>
# Supply Chain Status (Day {obs.current_day}/30)

## Inventory
- Current Stock: {obs.current_inventory} units
- Pending Orders: {obs.pending_orders} units (arrive in {obs.days_until_delivery} days)
- Total Available: {obs.current_inventory + obs.pending_orders} units

## Today's Demand
- Customer Orders: {obs.current_demand} units

## Recent Performance
- Units Sold: {obs.units_sold} units
- Shortage: {obs.units_shortage} units

## Economics
- Revenue: $10 per unit sold
- Shortage Penalty: $5 per unit unmet
- Holding Cost: $2 per unit in inventory
- Ordering Cost: $1 per unit ordered

## Your Task
Decide how many units to order from the supplier (0-100).
Remember: Orders take 3 days to arrive!

Think step by step:
1. Analyze current inventory levels
2. Consider upcoming demand patterns
3. Account for pending orders
4. Make an optimal decision

<|assistant|>
Analysis and Decision:
"""
        return prompt
    
    def parse_action(self, response: str) -> int:
        """
        Extract order quantity from LLM response.
        
        The LLM might say:
        "Based on analysis, I recommend ordering 50 units..."
        "Order: 45"
        "I will order 60 units to meet demand"
        """
        # Look for patterns like "order 50", "ordering 50", "50 units"
        patterns = [
            r'order(?:ing)?\s*:?\s*(\d+)',
            r'(\d+)\s*units',
            r'recommend(?:ing)?\s*(\d+)',
            r'decision\s*:?\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                quantity = int(match.group(1))
                return max(0, min(100, quantity))  # Clip to valid range
        
        # Fallback: find any number
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            quantity = int(numbers[-1])  # Take last number
            return max(0, min(100, quantity))
        
        # Default: order nothing
        return 0
    
    def step(self, response_text: str):
        """Execute action and return transition."""
        action = self.parse_action(response_text)
        
        result = self.env.step(SupplyChainAction(order_quantity=action))
        
        self.current_obs = result.observation
        
        return {
            'observation': result.observation,
            'reward': result.reward,
            'done': result.done,
            'action': action,
        }
    
    def close(self):
        self.env.close()



# PART 2: Generate Training Data


def collect_demonstrations(num_episodes: int = 20) -> List[Dict]:
    """
    Collect demonstration episodes using a heuristic policy.
    
    This creates a dataset of (prompt, response, reward) tuples that
    TRL can use for training.
    """
    print(f"Collecting {num_episodes} demonstration episodes...")
    
    env = SupplyChainRLEnvironment()
    all_experiences = []
    
    for episode_idx in range(num_episodes):
        env.reset()
        episode_experiences = []
        episode_reward = 0
        done = False
        
        while not done:
            # Get prompt
            prompt = env.get_prompt()
            obs = env.current_obs
            
            # Heuristic policy (this is what we'll try to improve with RL)
            total_available = obs.current_inventory + obs.pending_orders
            
            if total_available < 30:
                order_qty = 60
                reasoning = "Low inventory detected. Ordering 60 units to prevent stockouts."
            elif total_available < 50:
                order_qty = 40
                reasoning = "Moderate inventory. Ordering 40 units to maintain buffer."
            elif total_available > 80:
                order_qty = 0
                reasoning = "High inventory levels. Not ordering to reduce holding costs."
            else:
                order_qty = 20
                reasoning = "Stable inventory. Ordering 20 units for gradual replenishment."
            
            # Format response as LLM would
            response = f"""Let me analyze the situation:

Current total inventory: {total_available} units
Today's demand: {obs.current_demand} units
Days until next delivery: {obs.days_until_delivery} days

{reasoning}

Decision: Order {order_qty} units."""
            
            # Execute and record
            transition = env.step(response)
            
            episode_experiences.append({
                'prompt': prompt,
                'response': response,
                'reward': transition['reward'],
                'action': transition['action'],
            })
            
            episode_reward += transition['reward']
            done = transition['done']
        
        print(f"Episode {episode_idx + 1}/{num_episodes} - Total Reward: ${episode_reward:.2f}")
        all_experiences.extend(episode_experiences)
    
    env.close()
    
    print(f"\nCollected {len(all_experiences)} experiences")
    print(f"Average reward per step: ${sum(e['reward'] for e in all_experiences) / len(all_experiences):.2f}")
    
    return all_experiences


def create_trl_dataset(experiences: List[Dict]) -> Dataset:
    """Convert experiences to HuggingFace Dataset format for TRL."""
    
    # TRL typically expects: query, response, reward
    dataset_dict = {
        'query': [exp['prompt'] for exp in experiences],
        'response': [exp['response'] for exp in experiences],
        'reward': [exp['reward'] for exp in experiences],
    }
    
    return Dataset.from_dict(dataset_dict)



# PART 3: Train with TRL


def train_supply_chain_agent():
    """
    Train an LLM agent using TRL's online RL approach.
    
    Note: TRL's API evolves frequently. This is based on their general approach.
    Check https://huggingface.co/docs/trl/ for the latest API.
    """
    
    print("=" * 80)
    print("Training Supply Chain Agent with TRL")
    print("=" * 80)
    
    # 1. Load model
    print("\n📦 Loading model and tokenizer...")
    model_name = "gpt2"  # Use "meta-llama/Llama-3.2-1B-Instruct" for better results
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # 2. Collect demonstrations
    print("\n🎯 Collecting demonstration data...")
    experiences = collect_demonstrations(num_episodes=10)
    
    # Save demonstrations
    with open('/mnt/user-data/outputs/supply_chain_demonstrations.json', 'w') as f:
        json.dump(experiences, f, indent=2, default=str)
    print("Saved demonstrations to supply_chain_demonstrations.json")
    
    # 3. Create dataset
    print("\n📊 Creating training dataset...")
    dataset = create_trl_dataset(experiences)
    
    print(f"Dataset size: {len(dataset)} examples")
    print("\nExample entry:")
    print(f"Prompt: {dataset[0]['query'][:200]}...")
    print(f"Response: {dataset[0]['response'][:200]}...")
    print(f"Reward: {dataset[0]['reward']}")
    
    # 4. Training with TRL
    print("\n🚀 Starting RL training...")
    print("Note: TRL's exact API depends on version. Here's the conceptual approach:")
    print("""
    from trl import GRPOConfig, GRPOTrainer
    
    # Configure training
    training_config = GRPOConfig(
        output_dir="./supply_chain_agent",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        max_length=512,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model("./supply_chain_agent_final")
    """)
    
    print("\n✅ Training template complete!")
    print("\nFor actual training, install TRL and adapt to their latest API:")
    print("pip install trl")
    print("See: https://huggingface.co/docs/trl/")



# PART 4: Inference with Trained Agent


def test_trained_agent(model_path: str = "gpt2"):
    """
    Test a trained agent on the environment.
    """
    print("=" * 80)
    print("Testing Trained Agent")
    print("=" * 80)
    
    # Load model
    print(f"\n📦 Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Create environment
    env = SupplyChainRLEnvironment()
    
    # Run episode
    print("\n🎮 Running test episode...")
    env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    
    while not done:
        # Get prompt
        prompt = env.get_prompt()
        
        # Generate response from LLM
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Execute action
        transition = env.step(response)
        
        print(f"\n--- Step {step_count + 1} (Day {env.current_obs.current_day}) ---")
        print(f"Action: Order {transition['action']} units")
        print(f"Reward: ${transition['reward']:.2f}")
        print(f"LLM Response: {response[:100]}...")
        
        episode_reward += transition['reward']
        done = transition['done']
        step_count += 1
    
    env.close()
    
    print(f"\n{'=' * 80}")
    print(f"Episode Complete!")
    print(f"Total Reward: ${episode_reward:.2f}")
    print(f"Steps: {step_count}")
    print(f"{'=' * 80}")



# MAIN


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        # Just collect demonstrations
        experiences = collect_demonstrations(num_episodes=20)
        with open('/mnt/user-data/outputs/supply_chain_demonstrations.json', 'w') as f:
            json.dump(experiences, f, indent=2, default=str)
        print("✅ Demonstrations saved!")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test with base model (no training)
        test_trained_agent()
        
    else:
        # Full training pipeline
        train_supply_chain_agent()
