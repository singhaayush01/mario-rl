def compute_reward(info, action, done, prev_info=None):
    """
    Compute a shaped reward for the Mario agent.

    info: dict with game state (e.g., x_pos, coins, enemies_defeated)
    action: int representing what action the agent took
    done: bool indicating if the episode (level) ended
    prev_info: previous step's info (to calculate progress)
    """

    reward = 0

    # Reward for moving forward (progress)
    if prev_info is not None:
        progress = info.get('x_pos', 0) - prev_info.get('x_pos', 0)
        reward += max(0, progress * 0.1)  # small reward for forward motion

    # Reward for collecting coins
    if info.get('coins', 0) > (prev_info.get('coins', 0) if prev_info else 0):
        reward += 100

    # Reward for defeating enemies
    if info.get('enemies_defeated', 0) > (prev_info.get('enemies_defeated', 0) if prev_info else 0):
        reward += 500

    # Reward for finishing the level
    if info.get('flag_get', False):
        reward += 1000

    # Penalty for dying before finishing
    if done and not info.get('flag_get', False):
        reward -= 50

    # Small penalty for doing nothing (if action 0 = no-op)
    if action == 0:
        reward -= 0.1

    return reward
