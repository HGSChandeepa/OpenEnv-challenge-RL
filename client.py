# client.py
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import SupplyChainAction, SupplyChainObservation


class SupplyChainEnv(EnvClient[SupplyChainAction, SupplyChainObservation, State]):
    """
    Client for interacting with the Supply Chain environment.
    
    This maintains a persistent WebSocket connection to the server,
    enabling efficient multi-step interactions.
    
    Usage:
        # From Docker
        env = SupplyChainEnv.from_docker_image("supply-chain-env:latest")
        
        # From Hugging Face Hub
        env = SupplyChainEnv.from_hub("username/supply-chain-env")
        
        # From local server
        env = SupplyChainEnv(base_url="http://localhost:8000")
        
        # Use with context manager
        with env:
            obs = env.reset()
            result = env.step(SupplyChainAction(order_quantity=50))
            print(f"Reward: {result.reward}")
    """
    
    def _step_payload(self, action: SupplyChainAction) -> dict:
        """Convert action to JSON payload for the server."""
        return {
            "order_quantity": action.order_quantity
        }
    
    def _parse_result(self, payload: dict) -> StepResult[SupplyChainObservation]:
        """Parse server response into StepResult with typed observation."""
        obs_data = payload.get("observation", {})
        
        # Create observation from server data
        obs = SupplyChainObservation(
            current_inventory=obs_data.get("current_inventory", 0),
            pending_orders=obs_data.get("pending_orders", 0),
            current_demand=obs_data.get("current_demand", 0),
            days_until_delivery=obs_data.get("days_until_delivery", 0),
            current_day=obs_data.get("current_day", 1),
            units_sold=obs_data.get("units_sold", 0),
            units_shortage=obs_data.get("units_shortage", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: dict) -> State:
        """Parse server state information."""
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
