# server/app.py
import sys
from pathlib import Path

# Add parent directory to path to allow importing models
sys.path.insert(0, str(Path(__file__).parent.parent))

from openenv.core.env_server import create_app
from models import SupplyChainAction, SupplyChainObservation
from server.supply_chain_environment import SupplyChainEnvironment


# Create the FastAPI app
# Pass the class (not an instance) so each WebSocket session gets its own environment
app = create_app(
    SupplyChainEnvironment,
    SupplyChainAction,
    SupplyChainObservation,
    env_name="supply_chain"
)


# If you need to configure the environment with parameters, use a factory function:
# def create_supply_chain_env():
#     """Factory function to create environment with custom config."""
#     return SupplyChainEnvironment(
#         warehouse_capacity=300,
#         max_order_size=150,
#         supplier_lead_time=5
#     )
#
# app = create_app(
#     create_supply_chain_env,
#     SupplyChainAction,
#     SupplyChainObservation,
#     env_name="supply_chain"
# )
