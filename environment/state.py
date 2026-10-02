"""Representation helpers for the delivery world state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeliveryState:
    rider_position: tuple[int, int]
    restaurant_position: tuple[int, int]
    customer_position: tuple[int, int]
    picked_up: bool = False
    delivered: bool = False

    def as_observation(self, grid_size: int) -> np.ndarray:
        rider_x, rider_y = self.rider_position
        restaurant_x, restaurant_y = self.restaurant_position
        customer_x, customer_y = self.customer_position

        return np.array(
            [
                rider_x,
                rider_y,
                restaurant_x,
                restaurant_y,
                customer_x,
                customer_y,
                int(self.picked_up),
            ],
            dtype=np.int32,
        )


def clamp_position(position: tuple[int, int], grid_size: int) -> tuple[int, int]:
    x, y = position
    x = max(0, min(grid_size - 1, x))
    y = max(0, min(grid_size - 1, y))
    return x, y
