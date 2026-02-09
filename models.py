# models.py
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class SupplyChainAction(Action):
    """Action representing an order decision in the supply chain."""
    
    order_quantity: int = Field(
        ..., 
        ge=0,  # Greater than or equal to 0
        description="Number of units to order from supplier (0-100)"
    )


class SupplyChainObservation(Observation):
    """Observation of the current supply chain state."""
    
    # Inventory information
    current_inventory: int = Field(..., description="Units currently in warehouse")
    pending_orders: int = Field(..., description="Units ordered but not yet delivered")
    
    # Demand information
    current_demand: int = Field(..., description="Customer demand for today")
    
    # Time information
    days_until_delivery: int = Field(..., description="Days until next shipment arrives")
    current_day: int = Field(..., description="Current simulation day (1-30)")
    
    # Performance metrics (for agent to learn from)
    units_sold: int = Field(default=0, description="Units successfully sold today")
    units_shortage: int = Field(default=0, description="Unmet demand today")
    
    # Standard RL fields (inherited from Observation)
    # done: bool - whether episode is finished
    # reward: float - reward for the current step
