# agent.py
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    def sense_and_act(self, percept):
        if percept.get('food_here'):
            return 'suck'
        if percept.get('wall_ahead'):
            return 'turn_left'
        return 'move_forward'

class ModelBasedAgent:
    def __init__(self):
        self.visited_cells = set()
        self.pos = (0, 0)
        self.facing = 'Up'
        self.last_action = None

    def sense_and_act(self, percept):
        if self.last_action == 'turn_left':
            dirs = ['Up', 'Left', 'Down', 'Right']
            self.facing = dirs[(dirs.index(self.facing) + 1) % 4]
        elif self.last_action == 'turn_right':
            dirs = ['Up', 'Right', 'Down', 'Left']
            self.facing = dirs[(dirs.index(self.facing) + 1) % 4]
        elif self.last_action == 'move_forward':
            if self.facing == 'Up': self.pos = (self.pos[0], self.pos[1] + 1)
            elif self.facing == 'Down': self.pos = (self.pos[0], self.pos[1] - 1)
            elif self.facing == 'Left': self.pos = (self.pos[0] - 1, self.pos[1])
            elif self.facing == 'Right': self.pos = (self.pos[0] + 1, self.pos[1])

        self.visited_cells.add(self.pos)

        left_pos = self.pos
        if self.facing == 'Up': left_pos = (self.pos[0] - 1, self.pos[1])
        elif self.facing == 'Down': left_pos = (self.pos[0] + 1, self.pos[1])
        elif self.facing == 'Left': left_pos = (self.pos[0], self.pos[1] - 1)
        elif self.facing == 'Right': left_pos = (self.pos[0], self.pos[1] + 1)

        left_is_visited = left_pos in self.visited_cells

        if percept.get('food_here'):
            action = 'suck'
        elif percept.get('wall_ahead'):
            if left_is_visited:
                action = 'turn_right'
            else:
                action = 'turn_left'
        else:
            action = 'move_forward'

        self.last_action = action
        return action