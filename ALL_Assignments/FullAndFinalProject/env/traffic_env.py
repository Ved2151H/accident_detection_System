import numpy as np

class TrafficEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.random.randint(0, 10, size=4)
        self.wait_time = 0
        return self._get_state()

    def _get_state(self):
        return np.append(self.state, self.wait_time)

    def step(self, action):
        # 0 = NS green, 1 = EW green

        if action == 0:
            self.state[0] = max(0, self.state[0] - 3)
            self.state[1] = max(0, self.state[1] - 3)
        else:
            self.state[2] = max(0, self.state[2] - 3)
            self.state[3] = max(0, self.state[3] - 3)

        # New cars arrive
        self.state += np.random.randint(0, 3, size=4)

        # Update waiting time
        self.wait_time += np.sum(self.state)

        reward = -np.sum(self.state)

        return self._get_state(), reward, False