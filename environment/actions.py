"""Action constants and helpers for the delivery environment."""

ACTION_NAMES = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "PICKUP",
    "DELIVER",
    "SELECT_ORDER_0",
    "SELECT_ORDER_1",
    "ASSIGN_ORDER_0_TO_RIDER_0",
    "ASSIGN_ORDER_0_TO_RIDER_1",
    "ASSIGN_ORDER_1_TO_RIDER_0",
    "ASSIGN_ORDER_1_TO_RIDER_1",
)
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
INDEX_TO_ACTION = {index: name for name, index in ACTION_TO_INDEX.items()}

MOVE_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
