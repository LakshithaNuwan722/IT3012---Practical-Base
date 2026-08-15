# agent.py
import random
from collections import deque
import heapq


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    """
    Practical 03 - A Goal-Based / Planning Agent.

    Instead of reacting to immediate percepts, this agent builds an abstract model
    of the world from the percept and runs an uninformed search (BFS, DFS or UCS)
    to compute a complete plan (sequence of actions) to the nearest food pellet,
    then executes that plan one step at a time.
    """

    # Map a movement action to its (dx, dy) offset. Matches execute_action() in
    # visual_grid_game.py: Up = +y, Down = -y, Left = -x, Right = +x.
    MOVES = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0),
    }

    def __init__(self):
        # Step 1.3: the offline plan and the active search strategy.
        self.plan = []
        self.active_algo = 'BFS'   # Change to 'DFS' or 'UCS' to compare behaviour.

    # ------------------------------------------------------------------ helpers
    def _get_successors(self, state, grid_size, walls):
        """Yield (action, next_state) pairs that are on-grid and not walls."""
        width, height = grid_size
        x, y = state
        for action, (dx, dy) in self.MOVES.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                yield action, (nx, ny)

    def _closest_food(self, start, all_food):
        """Pick the food pellet with the smallest Manhattan distance from start."""
        return min(
            all_food,
            key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1])
        )

    # ------------------------------------------------------------------ searches
    def bfs_search(self, start, goal, grid_size, walls):
        """Breadth-First Search: FIFO queue -> shallowest node first (optimal here)."""
        frontier = deque([(start, [])])          # (state, path_of_actions)
        reached = {start}                        # graph search: track explored states
        while frontier:
            state, path = frontier.popleft()     # FIFO
            if state == goal:
                return path
            for action, nxt in self._get_successors(state, grid_size, walls):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return []                                # no path found

    def dfs_search(self, start, goal, grid_size, walls):
        """Depth-First Search: LIFO stack -> deepest node first (winding paths)."""
        frontier = [(start, [])]                 # use a list as a stack
        reached = {start}
        while frontier:
            state, path = frontier.pop()         # LIFO
            if state == goal:
                return path
            for action, nxt in self._get_successors(state, grid_size, walls):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return []

    def ucs_search(self, start, goal, grid_size, walls):
        """Uniform-Cost Search: priority queue ordered by total path cost g(n)."""
        counter = 0                              # tie-breaker so tuples never compare paths
        frontier = [(0, counter, start, [])]     # (cost, tiebreak, state, path)
        reached = {start: 0}                     # best known cost to each state
        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            if state == goal:
                return path
            for action, nxt in self._get_successors(state, grid_size, walls):
                new_cost = cost + 1              # uniform step cost of 1 per move
                if nxt not in reached or new_cost < reached[nxt]:
                    reached[nxt] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, nxt, path + [action]))
        return []

    # ------------------------------------------------------------------ act
    def sense_and_act(self, percept: dict) -> str:
        # Step 1.3: only plan when we have no plan left to execute.
        if not self.plan:
            all_food = percept.get('all_food', [])
            if not all_food:
                return 'Up'                      # nothing to chase; harmless default

            start = tuple(percept['agent_pos'])
            grid_size = percept['grid_size']
            walls = set(tuple(w) for w in percept['walls'])
            goal = self._closest_food(start, all_food)

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, goal, grid_size, walls)

            # If search failed (e.g. food walled off), fall back to a single step.
            if not self.plan:
                return random.choice(list(self.MOVES.keys()))

        # Step 1.3: return and consume the first action of the plan.
        return self.plan.pop(0)
