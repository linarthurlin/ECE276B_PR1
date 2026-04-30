from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
import os

from minigrid.core.world_object import Wall, Key, Door, Goal

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

KEY_POSITIONS = [(2, 2), (2, 3), (1, 6)]
GOAL_POSITIONS = [(6, 1), (7, 3), (6, 6)]
DOOR_POSITIONS = [(5, 3), (5, 7)]
DIR_LABELS = ["R", "D", "L", "U"]

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)


# Dynamic Programming Planner
class KnownMapDPPlanner:
    def __init__(self, env, info):
        self.env = env
        self.info = info
        self.width = env.width
        self.height = env.height
        self.goal_pos = tuple(info["goal_pos"])

        # Value
        self.V = np.full((self.width, self.height, 4, 2, 2), np.inf)
        
        # Policy
        self.Policy = np.full((self.width, self.height, 4, 2, 2), -1, dtype=int)

        # Terminal Cost
        goal_x, goal_y = self.goal_pos
        self.V[goal_x, goal_y, :, :, :] = 0.0

    def get_front_pos(self, x, y, d):
        if d == 0: return x + 1, y  # Right
        if d == 1: return x, y + 1  # Down
        if d == 2: return x - 1, y  # Left
        if d == 3: return x, y - 1  # up
        return x, y

    def get_next_state_and_cost(self, state, action):
        x, y, direction, has_key, door_open = state
        next_state = list(state)
        cost = step_cost(action)

        fx, fy = self.get_front_pos(x, y, direction)

        if action == TL:
            next_state[2] = (direction - 1) % 4
        elif action == TR:
            next_state[2] = (direction + 1) % 4
        elif action == MF:
            if not (0 <= fx < self.width and 0 <= fy < self.height):
                return tuple(state), np.inf
            
            cell = self.env.grid.get(fx, fy)
            # Check if the next cell is a Wall
            if isinstance(cell, Wall):
                return tuple(state), np.inf
            # Check if the next cell is a Closed Door
            if isinstance(cell, Door) and door_open == 0 and not cell.is_open:
                return tuple(state), np.inf

            next_state[0], next_state[1] = fx, fy                

        elif action == PK:
            if not (0 <= fx < self.width and 0 <= fy < self.height):
                return tuple(state), np.inf
            cell = self.env.grid.get(fx, fy)
            # Check if the next cell is a Key while currently having no key
            if isinstance(cell, Key) and has_key == 0:
                next_state[3] = 1
            else:
                return tuple(state), np.inf
            
        elif action == UD:
            if not (0 <= fx < self.width and 0 <= fy < self.height):
                return tuple(state), np.inf
            cell = self.env.grid.get(fx, fy)
            # Check if the next cell is a Closed Door while currently having a key
            if isinstance(cell, Door) and has_key == 1 and door_open == 0:
                next_state[4] = 1
            else:
                return tuple(state), np.inf

        return tuple(next_state), cost

    def get_action(self, state):
        x, y, direction, has_key, door_open = state
        return int(self.Policy[x, y, direction, has_key, door_open])
    
    def value_iteration(self, max_iters=1000, tol=1e-6):
        gx, gy = self.info["goal_pos"]

        for iteration in range(max_iters):
            delta = 0.0
            V_old = np.copy(self.V)

            # State Space
            for x in range(self.width):
                for y in range(self.height):
                    if isinstance(self.env.grid.get(x, y), Wall):
                        continue
                    if (x, y) == (gx, gy):
                        continue
            
                    for d in range(4):
                        for k in range(2):
                            for o in range(2):
                                state = (x, y, d, k , o)
                                best_val = np.inf
                                best_act = -1

                                for a in [MF, TL, TR, PK, UD]:
                                    next_state, cost = self.get_next_state_and_cost(state, a)
                                    if cost == np.inf:
                                        continue

                                    nx, ny, nd, nk, no = next_state
                                    # Value
                                    val = cost + V_old[nx, ny, nd, nk, no]

                                    if val < best_val:
                                        best_val = val
                                        best_act = a

                                self.V[x, y, d, k, o] = best_val
                                self.Policy[x, y, d, k, o] = best_act
                                old_val = V_old[x, y, d, k, o]
                                if np.isfinite(old_val) and np.isfinite(best_val):
                                    change = abs(old_val - best_val)
                                elif np.isfinite(old_val) != np.isfinite(best_val):
                                    change = np.inf
                                else:
                                    change = 0.0
                                delta = max(delta, change)

            if delta < tol:
                print(f"Converged in {iteration} iterations.")
                break


class RandomMapDPPlanner:
    def __init__(self):
        self.width = 10
        self.height = 10
        self.actions = [MF, TL, TR, PK, UD]

        # Dimension: [x, y, dir, has_key, d1_open, d2_open, k_idx, g_idx]
        self.V = np.full((10, 10, 4, 2, 2, 2, 3, 3), np.inf)
        self.Policy = np.full((10, 10, 4, 2, 2, 2, 3, 3), -1, dtype=int)

        # Terminal Cost: g_idx and V = 0
        for g_idx, (gx, gy) in enumerate(GOAL_POSITIONS):
            self.V[gx, gy, :, :, :, :, :, g_idx] = 0.0

    def is_wall(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True

        if x == 5 and (x, y) not in DOOR_POSITIONS:
            return True

        return False
    
    def get_front_pos(self, x, y, d):
        if d == 0: return x + 1, y  # Right
        if d == 1: return x, y + 1  # Down
        if d == 2: return x - 1, y  # Left
        if d == 3: return x, y - 1  # up
        return x, y

    def get_next_state_and_cost(self, state, action):
        x, y, direction, has_key, d1_open, d2_open, k_idx, g_idx = state
        next_state = list(state)
        cost = step_cost(action)

        fx, fy = self.get_front_pos(x, y, direction)
        front_pos = (fx, fy)

        if action == TL:
            next_state[2] = (direction - 1) % 4
        elif action == TR:
            next_state[2] = (direction + 1) % 4
        elif action == MF:
            if self.is_wall(fx, fy):
                return tuple(state), np.inf

            if front_pos == DOOR_POSITIONS[0] and d1_open == 0:
                return tuple(state), np.inf
            if front_pos == DOOR_POSITIONS[1] and d2_open == 0:
                return tuple(state), np.inf

            next_state[0], next_state[1] = fx, fy
        elif action == PK:
            if has_key == 1:
                return tuple(state), np.inf

            if front_pos == KEY_POSITIONS[k_idx]:
                next_state[3] = 1
            else:
                return tuple(state), np.inf
        elif action == UD:
            if has_key == 0:
                return tuple(state), np.inf

            if front_pos == DOOR_POSITIONS[0] and d1_open == 0:
                next_state[4] = 1
            elif front_pos == DOOR_POSITIONS[1] and d2_open == 0:
                next_state[5] = 1
            else:
                return tuple(state), np.inf

        return tuple(next_state), cost

    def get_action(self, state):
        x, y, d, k, d1, d2, k_idx, g_idx = state
        return int(self.Policy[x, y, d, k, d1, d2, k_idx, g_idx])

    def value_iteration(self, max_iters=2000, tol=1e-6):
        for iteration in range(max_iters):
            delta = 0.0
            V_old = np.copy(self.V)

            for x in range(self.width):
                for y in range(self.height):
                    if self.is_wall(x, y):
                        continue

                    for d in range(4):
                        for has_key in range(2):
                            for d1_open in range(2):
                                for d2_open in range(2):
                                    for k_idx in range(3):
                                        for g_idx, goal_pos in enumerate(GOAL_POSITIONS):
                                            if (x, y) == goal_pos:
                                                continue

                                            state = (x, y, d, has_key, d1_open, d2_open, k_idx, g_idx)
                                            best_val = np.inf
                                            best_act = -1

                                            for a in self.actions:
                                                next_state, stage_cost = self.get_next_state_and_cost(state, a)
                                                if stage_cost == np.inf:
                                                    continue

                                                nx, ny, nd, nk, nd1, nd2, nk_idx, ng_idx = next_state
                                                val = stage_cost + V_old[nx, ny, nd, nk, nd1, nd2, nk_idx, ng_idx]

                                                if val < best_val:
                                                    best_val = val
                                                    best_act = a

                                            self.V[x, y, d, has_key, d1_open, d2_open, k_idx, g_idx] = best_val
                                            self.Policy[x, y, d, has_key, d1_open, d2_open, k_idx, g_idx] = best_act

                                            old_val = V_old[x, y, d, has_key, d1_open, d2_open, k_idx, g_idx]
                                            if np.isfinite(old_val) and np.isfinite(best_val):
                                                change = abs(old_val - best_val)
                                            elif np.isfinite(old_val) != np.isfinite(best_val):
                                                change = np.inf
                                            else:
                                                change = 0.0

                                            delta = max(delta, change)

            if delta < tol:
                print(f"Converged in {iteration} iterations.")
                break


def doorkey_problem_random(env, planner, start_dir=None):
    key_pos = None
    k_idx = -1
    for idx, pos in enumerate(KEY_POSITIONS):
        if isinstance(env.grid.get(*pos), Key):
            key_pos = pos
            k_idx = idx
            break

    goal_pos = None
    g_idx = -1
    for idx, pos in enumerate(GOAL_POSITIONS):
        if isinstance(env.grid.get(*pos), Goal):
            goal_pos = pos
            g_idx = idx
            break

    d1_open = 1 if isinstance(env.grid.get(*DOOR_POSITIONS[0]), Door) and env.grid.get(*DOOR_POSITIONS[0]).is_open else 0
    d2_open = 1 if isinstance(env.grid.get(*DOOR_POSITIONS[1]), Door) and env.grid.get(*DOOR_POSITIONS[1]).is_open else 0

    ax, ay = env.agent_pos
    ad = env.agent_dir if start_dir is None else int(start_dir)
    has_key = 1 if env.carrying is not None else 0

    curr_state = (ax, ay, ad, has_key, d1_open, d2_open, k_idx, g_idx)
    optim_act_seq = []

    max_steps = planner.width * planner.height * 30
    for _ in range(max_steps):
        if (curr_state[0], curr_state[1]) == goal_pos:
            break

        act = planner.get_action(curr_state)
        if act == -1:
            print("Warning: Goal unreachable in random map rollout.")
            break

        optim_act_seq.append(act)
        curr_state, _ = planner.get_next_state_and_cost(curr_state, act)

    if (curr_state[0], curr_state[1]) != goal_pos:
        print("Warning: Cannot hit the goal.")

    return optim_act_seq

def doorkey_problem(env, info, start_dir=None):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Template:
        Replace the placeholder list with the action sequence returned by your
        planner. Minimize the same total stage cost as in utils.step_cost (and
        as defined in your report's MDP). You may branch on env / loaded map if
        needed for Part (A); Part (B) should respect the single-policy requirement.
    """
    # STUDENT: placeholder sequence for wiring; not a solution for all maps.
    # optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    # return optim_act_seq

    # DP Planner
    planner = KnownMapDPPlanner(env, info)
    planner.value_iteration()

    ax, ay = env.agent_pos
    ad = env.agent_dir if start_dir is None else int(start_dir)
    has_key = 1 if env.carrying is not None else 0

    door_open = 0
    for x in range(env.width):
        for y in range(env.height):
            cell = env.grid.get(x, y)
            if isinstance(cell, Door) and cell.is_open:
                door_open = 1
                break

    curr_state = (ax, ay, ad, has_key, door_open)
    optim_act_seq = []
    
    max_steps = planner.width * planner.height * 20
    for _ in range(max_steps):
        if (curr_state[0], curr_state[1]) == planner.goal_pos:
            break
            
        act = planner.get_action(curr_state)
        
        if act == -1:
            print("Warning: Goal unreachable from initial state.")
            break
            
        optim_act_seq.append(act)
        curr_state, _ = planner.get_next_state_and_cost(curr_state, act)

    if (curr_state[0], curr_state[1]) != planner.goal_pos:
        print("Warning: Cannot hit the goal.")

    return optim_act_seq


def sequence_cost(seq):
    return sum(step_cost(a) for a in seq)


def partA():
    # env_path = "./envs/known_envs/example-8x8.env"
    # env, info = load_env(env_path)  # load an environment
    # seq = doorkey_problem(env)  # find the optimal action sequence
    # draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save
    env_list = [
        "doorkey-5x5-normal.env",
        "doorkey-6x6-normal.env",
        "doorkey-8x8-normal.env",
        "doorkey-6x6-direct.env",
        "doorkey-8x8-direct.env",
        "doorkey-6x6-shortcut.env",
        "doorkey-8x8-shortcut.env"
    ]

    output_dir = "./results/partA"
    os.makedirs(output_dir, exist_ok=True)

    for map_name in env_list:
        env_path = f"./envs/known_envs/{map_name}"
        map_base = map_name.replace('.env', '')

        for start_dir in range(4):
            print(f"\nEvaluating: {map_name} (start dir={DIR_LABELS[start_dir]}) ...")

            env, info = load_env(env_path)
            env.unwrapped.agent_dir = start_dir

            seq = doorkey_problem(env, info, start_dir=start_dir)
            total_cost = sequence_cost(seq)

            out_name = f"{map_base}-dir{DIR_LABELS[start_dir]}"
            frame_dir = f"{output_dir}/{out_name}"
            draw_gif_from_seq(seq, env, path=f"{output_dir}/{out_name}.gif", frame_dir=frame_dir)
            print(f"-> Sequence length: {len(seq)}, total cost: {total_cost}. ")

def partB():
    env_folder = "./envs/random_envs"
    output_dir = "./results/partB"
    os.makedirs(output_dir, exist_ok=True)

    planner = RandomMapDPPlanner()
    planner.value_iteration()

    env_list = sorted(
        [f for f in os.listdir(env_folder) if f.endswith(".env")],
        key=lambda name: int(name.split("-")[-1].split(".")[0]),
    )

    for map_name in env_list:
        env_path = os.path.join(env_folder, map_name)
        map_base = map_name.replace(".env", "")

        for start_dir in range(4):
            print(f"\nEvaluating: {map_name} (start dir={DIR_LABELS[start_dir]})")

            env, _ = load_env(env_path)
            env.unwrapped.agent_dir = start_dir

            seq = doorkey_problem_random(env, planner, start_dir=start_dir)
            total_cost = sequence_cost(seq)

            out_name = f"{map_base}-dir{DIR_LABELS[start_dir]}"
            frame_dir = f"{output_dir}/{out_name}"
            draw_gif_from_seq(seq, env, path=f"{output_dir}/{out_name}.gif", frame_dir=frame_dir)
            print(f"-> Sequence length: {len(seq)}, total cost: {total_cost}. ")


if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    partB()

