# server/supply_chain_environment.py
from uuid import uuid4
import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import SupplyChainAction, SupplyChainObservation


class SupplyChainEnvironment(Environment):
    """
    A simple supply chain management environment.
    
    The agent manages inventory by ordering from a supplier and fulfilling customer demand.
    """
    
    def __init__(
        self,
        warehouse_capacity: int = 200,
        max_order_size: int = 100,
        supplier_lead_time: int = 3,  # Days for delivery
        episode_length: int = 30,
        seed: int = None
    ):
        """
        Initialize the supply chain environment.
        
        Args:
            warehouse_capacity: Maximum units that can be stored
            max_order_size: Maximum units that can be ordered at once
            supplier_lead_time: Days between ordering and delivery
            episode_length: Number of days per episode
            seed: Random seed for reproducibility
        """
        self.warehouse_capacity = warehouse_capacity
        self.max_order_size = max_order_size
        self.supplier_lead_time = supplier_lead_time
        self.episode_length = episode_length
        
        # Initialize random number generator
        self.rng = random.Random(seed)
        
        # Environment state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Supply chain state
        self.current_inventory = 0
        self.pending_shipments = []  # List of (delivery_day, quantity) tuples
        self.current_day = 0
        
    def reset(self) -> SupplyChainObservation:
        """Reset the environment to initial state."""
        # Reset episode tracking
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Reset supply chain state
        self.current_inventory = 50  # Start with some initial inventory
        self.pending_shipments = []
        self.current_day = 1
        
        # Generate first day's demand
        demand = self._generate_demand()
        
        return SupplyChainObservation(
            current_inventory=self.current_inventory,
            pending_orders=sum(qty for _, qty in self.pending_shipments),
            current_demand=demand,
            days_until_delivery=self.supplier_lead_time if not self.pending_shipments else 
                               (self.pending_shipments[0][0] - self.current_day),
            current_day=self.current_day,
            units_sold=0,
            units_shortage=0,
            done=False,
            reward=0.0
        )
    
    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """
        Execute one step of the environment.
        
        1. Process the agent's order
        2. Receive any deliveries
        3. Generate customer demand
        4. Fulfill demand from inventory
        5. Calculate rewards
        """
        self._state.step_count += 1
        self.current_day += 1
        
        # Validate action
        order_qty = max(0, min(action.order_quantity, self.max_order_size))
        
        # 1. Place order (will arrive after lead time)
        if order_qty > 0:
            delivery_day = self.current_day + self.supplier_lead_time
            self.pending_shipments.append((delivery_day, order_qty))
        
        # 2. Receive deliveries that arrived today
        arrived_shipments = [
            (day, qty) for day, qty in self.pending_shipments 
            if day <= self.current_day
        ]
        
        for _, qty in arrived_shipments:
            # Add to inventory (respect capacity)
            space_available = self.warehouse_capacity - self.current_inventory
            units_received = min(qty, space_available)
            self.current_inventory += units_received
            
            # If we couldn't receive all units, that's a problem (penalized implicitly)
        
        # Remove delivered shipments
        self.pending_shipments = [
            (day, qty) for day, qty in self.pending_shipments 
            if day > self.current_day
        ]
        
        # 3. Generate customer demand
        demand = self._generate_demand()
        
        # 4. Fulfill demand
        units_sold = min(demand, self.current_inventory)
        units_shortage = demand - units_sold
        
        self.current_inventory -= units_sold
        
        # 5. Calculate reward
        reward = self._calculate_reward(
            order_qty=order_qty,
            units_sold=units_sold,
            units_shortage=units_shortage,
            inventory_level=self.current_inventory
        )
        
        # Check if episode is done
        done = self.current_day >= self.episode_length
        
        # Calculate days until next delivery
        if self.pending_shipments:
            days_until_delivery = self.pending_shipments[0][0] - self.current_day
        else:
            days_until_delivery = self.supplier_lead_time
        
        return SupplyChainObservation(
            current_inventory=self.current_inventory,
            pending_orders=sum(qty for _, qty in self.pending_shipments),
            current_demand=demand,
            days_until_delivery=days_until_delivery,
            current_day=self.current_day,
            units_sold=units_sold,
            units_shortage=units_shortage,
            done=done,
            reward=reward
        )
    
    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state
    
    def _generate_demand(self) -> int:
        """
        Generate customer demand for the current day.
        
        Uses a simple pattern:
        - Base demand: 10-20 units
        - Weekend spike: +10 units on days 6, 7, 13, 14, etc.
        - Random variation: ±5 units
        """
        base_demand = self.rng.randint(10, 20)
        
        # Weekend effect (every 7 days)
        day_of_week = self.current_day % 7
        weekend_bonus = 10 if day_of_week in [6, 0] else 0
        
        # Random variation
        variation = self.rng.randint(-5, 5)
        
        demand = max(0, base_demand + weekend_bonus + variation)
        return demand
    
    def _calculate_reward(
        self,
        order_qty: int,
        units_sold: int,
        units_shortage: int,
        inventory_level: int
    ) -> float:
        """
        Calculate reward for the current step.
        
        Reward components:
        - Sales revenue: +$10 per unit sold
        - Shortage penalty: -$5 per unit of unmet demand
        - Holding cost: -$2 per unit in inventory at end of day
        - Ordering cost: -$1 per unit ordered
        """
        revenue = units_sold * 10.0
        shortage_penalty = units_shortage * -5.0
        holding_cost = inventory_level * -2.0
        ordering_cost = order_qty * -1.0
        
        total_reward = revenue + shortage_penalty + holding_cost + ordering_cost
        
        return total_reward
