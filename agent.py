import random
from collections import deque
import heapq
import math

# agent.py

class SimpleReflexAgent:
    """A simple reflex agent that acts only on the current percept."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Stay'
        if percept.get('wall_ahead'):
            return 'Left'
        return 'Right'


class ModelBasedAgent:
    """A model-based agent that uses internal state to avoid repeating failed actions."""

    def __init__(self):
        self.last_percept = None
        self.last_action = None
        self.percept_history = {}

    def sense_and_act(self, percept: dict) -> str:
        percept_key = tuple(sorted(percept.items()))

        if self.last_percept is not None and self.last_action is not None:
            last_key = tuple(sorted(self.last_percept.items()))
            self.percept_history.setdefault(last_key, set()).add(self.last_action)

        if percept.get('food_here'):
            action = 'Stay'
        elif percept.get('wall_ahead'):
            tried_actions = self.percept_history.get(percept_key, set())
            if 'Left' not in tried_actions:
                action = 'Left'
            elif 'Right' not in tried_actions:
                action = 'Right'
            elif 'Up' not in tried_actions:
                action = 'Up'
            else:
                action = 'Down'
        else:
            action = 'Right'

        self.last_percept = dict(percept)
        self.last_action = action
        return action


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
    A search agent that implements three distinct uninformed search strategies.
    Core node expansion structure is identical; only the Frontier data structure changes.
    """

    def __init__(self, initial_state, goal_test, actions_fn, result_fn):
        """
        Initialize the search agent.
        
        Args:
            initial_state: The starting state
            goal_test: Function that returns True if a state is a goal
            actions_fn: Function that returns available actions for a state
            result_fn: Function that returns the resulting state after an action
        """
        self.initial_state = initial_state
        self.goal_test = goal_test
        self.actions_fn = actions_fn
        self.result_fn = result_fn
        self.plan = []  # Store the sequence of actions to execute
        self.active_algo = 'BFS'  # Can be 'BFS', 'DFS', or 'UCS'

    def manhattan_distance(self, pos, goal):
        """Return the Manhattan distance between two grid positions."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        """Return the Euclidean distance between two grid positions."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """Find a path from start_pos to goal_pos using A* search."""
        frontier = []
        reached_states = set()
        walls = set(walls)
        heuristic = (self.euclidean_distance if heuristic_type == 'euclidean'
                     else self.manhattan_distance)
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)

        start_g_cost = 0
        start_f_cost = start_g_cost + heuristic(start_pos, goal_pos)
        heapq.heappush(frontier, (start_f_cost, start_g_cost, start_pos, []))

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            x, y = current_pos
            width, height = grid_size
            neighbors = [
                ('Up', (x, y + 1)),
                ('Down', (x, y - 1)),
                ('Left', (x - 1, y)),
                ('Right', (x + 1, y)),
            ]

            for action, neighbor in neighbors:
                neighbor_x, neighbor_y = neighbor
                in_bounds = (0 <= neighbor_x < width and
                             0 <= neighbor_y < height)
                if not in_bounds or neighbor in walls or neighbor in reached_states:
                    continue

                new_g_cost = g_cost + 1
                new_f_cost = new_g_cost + heuristic(neighbor, goal_pos)
                new_path = path_taken + [action]
                heapq.heappush(frontier, (new_f_cost, new_g_cost, neighbor, new_path))

        return None

    def bfs_search(self):
        """
        Breadth-First Search using a FIFO queue (deque).
        Explores the shallowest nodes first.
        Maintains a 'reached' set for Graph Search (prevents infinite loops).
        
        Returns:
            A list of actions representing the path to the goal, or None if not found.
        """
        frontier = deque()
        reached = {self.initial_state}
        parent = {self.initial_state: None}
        action_to_state = {self.initial_state: None}
        
        frontier.append(self.initial_state)
        
        while frontier:
            state = frontier.popleft()  # FIFO: remove from front
            
            if self.goal_test(state):
                return self._reconstruct_path(state, parent, action_to_state)
            
            for action in self.actions_fn(state):
                next_state = self.result_fn(state, action)
                
                if next_state not in reached:
                    reached.add(next_state)
                    parent[next_state] = state
                    action_to_state[next_state] = action
                    frontier.append(next_state)
        
        return None  # No solution found

    def dfs_search(self):
        """
        Depth-First Search using a LIFO stack (list).
        Explores the deepest nodes first.
        Maintains a 'reached' set for Graph Search (prevents infinite loops).
        
        Returns:
            A list of actions representing the path to the goal, or None if not found.
        """
        frontier = [self.initial_state]  # LIFO stack
        reached = {self.initial_state}
        parent = {self.initial_state: None}
        action_to_state = {self.initial_state: None}
        
        while frontier:
            state = frontier.pop()  # LIFO: remove from end
            
            if self.goal_test(state):
                return self._reconstruct_path(state, parent, action_to_state)
            
            for action in self.actions_fn(state):
                next_state = self.result_fn(state, action)
                
                if next_state not in reached:
                    reached.add(next_state)
                    parent[next_state] = state
                    action_to_state[next_state] = action
                    frontier.append(next_state)
        
        return None  # No solution found

    def ucs_search(self):
        """
        Uniform Cost Search using a Priority Queue (heapq).
        Ordered by the total path cost g(n).
        Maintains a 'reached' set for Graph Search (prevents infinite loops).
        
        Returns:
            A list of actions representing the path to the goal, or None if not found.
        """
        frontier = []  # Min-heap: (cost, state_id, state)
        reached = {}   # state -> min_cost_to_reach
        parent = {self.initial_state: None}
        action_to_state = {self.initial_state: None}
        
        state_counter = 0  # For unique ordering when costs are equal
        heapq.heappush(frontier, (0, state_counter, self.initial_state))
        reached[self.initial_state] = 0
        state_counter += 1
        
        while frontier:
            cost, _, state = heapq.heappop(frontier)  # Pop lowest cost
            
            if self.goal_test(state):
                return self._reconstruct_path(state, parent, action_to_state)
            
            # Skip if we've already found a cheaper path to this state
            if cost > reached.get(state, float('inf')):
                continue
            
            for action in self.actions_fn(state):
                next_state = self.result_fn(state, action)
                next_cost = cost + 1  # Assume unit cost; modify if needed
                
                if next_state not in reached or next_cost < reached[next_state]:
                    reached[next_state] = next_cost
                    parent[next_state] = state
                    action_to_state[next_state] = action
                    heapq.heappush(frontier, (next_cost, state_counter, next_state))
                    state_counter += 1
        
        return None  # No solution found

    def _reconstruct_path(self, state, parent, action_to_state):
        """
        Reconstruct the path from initial state to goal state.
        
        Args:
            state: The goal state
            parent: Dictionary mapping states to their parent states
            action_to_state: Dictionary mapping states to the action that led to them
            
        Returns:
            A list of actions from initial state to goal state.
        """
        path = []
        current = state
        
        while parent[current] is not None:
            path.append(action_to_state[current])
            current = parent[current]
        
        path.reverse()
        return path

    def sense_and_act(self, percept: dict) -> str:
        """
        Sense the environment and act based on the current plan.
        If no plan exists, find the closest food and create a plan to reach it.
        
        Args:
            percept: Dictionary containing agent percepts including all_food, grid_size, walls, agent_pos
            
        Returns:
            The next action to take based on the plan.
        """
        # If plan is empty, create a new plan
        if not self.plan:
            agent_pos = tuple(percept.get('agent_pos', (0, 0)))
            all_food = percept.get('all_food', [])
            grid_size = percept.get('grid_size', (10, 10))
            walls = set(percept.get('walls', []))
            
            if not all_food:
                return 'Stay'  # No food to find
            
            # Find the closest food pellet
            closest_food = self._find_closest_food(agent_pos, all_food)
            
            # Set up the search problem
            goal_test = lambda state: state == closest_food
            actions_fn = lambda state: self._get_actions(state, grid_size, walls)
            result_fn = lambda state, action: self._get_result(state, action, grid_size, walls)
            
            # Create a new search agent for this problem
            search_agent = SearchAgent(agent_pos, goal_test, actions_fn, result_fn)
            
            # Execute the appropriate search algorithm
            if self.active_algo == 'BFS':
                self.plan = search_agent.bfs_search() or []
            elif self.active_algo == 'DFS':
                self.plan = search_agent.dfs_search() or []
            elif self.active_algo == 'UCS':
                self.plan = search_agent.ucs_search() or []
            elif self.active_algo == 'AStar':
                self.plan = search_agent.astar_search(
                    agent_pos, closest_food, walls, grid_size
                ) or []
            else:
                self.plan = []
        
        # Execute the first action from the plan
        if self.plan:
            return self.plan.pop(0)
        else:
            return 'Stay'

    def _find_closest_food(self, agent_pos, all_food):
        """
        Find the closest food pellet to the agent using Manhattan distance.
        
        Args:
            agent_pos: Tuple (x, y) of agent's current position
            all_food: List of tuples representing food positions
            
        Returns:
            Tuple representing the closest food position.
        """
        if not all_food:
            return agent_pos
        
        def manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        closest = min(all_food, key=lambda food: manhattan_distance(agent_pos, food))
        return closest

    def _get_actions(self, state, grid_size, walls):
        """
        Get valid actions from a state.
        
        Args:
            state: Tuple (x, y) representing current position
            grid_size: Tuple (width, height) of the grid
            walls: Set of wall positions
            
        Returns:
            List of valid action strings.
        """
        x, y = state
        width, height = grid_size
        actions = []
        
        # Up
        if y < height - 1 and (x, y + 1) not in walls:
            actions.append('Up')
        # Down
        if y > 0 and (x, y - 1) not in walls:
            actions.append('Down')
        # Left
        if x > 0 and (x - 1, y) not in walls:
            actions.append('Left')
        # Right
        if x < width - 1 and (x + 1, y) not in walls:
            actions.append('Right')
        
        return actions

    def _get_result(self, state, action, grid_size, walls):
        """
        Get the resulting state after taking an action.
        
        Args:
            state: Tuple (x, y) representing current position
            action: String representing the action ('Up', 'Down', 'Left', 'Right')
            grid_size: Tuple (width, height) of the grid
            walls: Set of wall positions
            
        Returns:
            Tuple (x, y) representing the new position.
        """
        x, y = state
        width, height = grid_size
        
        if action == 'Up':
            new_pos = (x, min(height - 1, y + 1))
        elif action == 'Down':
            new_pos = (x, max(0, y - 1))
        elif action == 'Left':
            new_pos = (max(0, x - 1), y)
        elif action == 'Right':
            new_pos = (min(width - 1, x + 1), y)
        else:
            new_pos = state
        
        # Don't move into walls
        if new_pos in walls:
            return state
        
        return new_pos