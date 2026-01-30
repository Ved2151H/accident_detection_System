import random

# -------------------- ENVIRONMENT --------------------

GRID = 4
ACTIONS = ['u', 'd', 'l', 'r']

START = (0, 0)
TREASURE = (3, 3)
TRAP = (2, 2)

TREASURE_REWARD = 10
TRAP_PENALTY = -10
STEP_PENALTY = -1

# -------------------- PARAMETERS --------------------

alpha = 0.8
alpha_decay = 0.995
gamma = 0.9
eps = 1.0
eps_decay = 0.995
min_eps = 0.01

EPISODES = 1000


# -------------------- MOVEMENT --------------------

def move(state, action):
    x, y = state

    if action == 'u':
        x = max(x - 1, 0)
    elif action == 'd':
        x = min(x + 1, GRID - 1)
    elif action == 'l':
        y = max(y - 1, 0)
    elif action == 'r':
        y = min(y + 1, GRID - 1)

    return (x, y)


# -------------------- Q-TABLE INITIALIZATION --------------------

def initialize_q_table():
    Q = {}
    for i in range(GRID):
        for j in range(GRID):
            Q[(i, j)] = {}
            for a in ACTIONS:
                Q[(i, j)][a] = 0.0
    return Q


# -------------------- ACTION SELECTION --------------------

def take_action(Q, state):
    if random.uniform(0, 1) < eps:
        return random.choice(ACTIONS)
    else:
        return max(Q[state], key=Q[state].get)


# -------------------- REWARD & TERMINAL --------------------

def get_reward(state):
    if state == TREASURE:
        return TREASURE_REWARD
    elif state == TRAP:
        return TRAP_PENALTY
    else:
        return STEP_PENALTY


def is_terminal(state):
    return state == TREASURE or state == TRAP


# -------------------- Q UPDATE --------------------

def update_q_value(Q, state, action, reward, next_state):
    # Q(s,a) <- Q(s,a) + alpha [ r + gamma * max(Q(s',a')) - Q(s,a) ]

    old_value = Q[state][action]
    future_max = max(Q[next_state].values())

    Q[state][action] = old_value + alpha * (
        reward + gamma * future_max - old_value
    )


# -------------------- TRAINING --------------------

def train_agent():
    global alpha, eps

    Q = initialize_q_table()

    for episode in range(EPISODES):
        state = START

        while not is_terminal(state):
            action = take_action(Q, state)
            next_state = move(state, action)
            reward = get_reward(next_state)

            update_q_value(Q, state, action, reward, next_state)

            state = next_state

        # Decay alpha and epsilon
        alpha *= alpha_decay
        eps = max(min_eps, eps * eps_decay)

    return Q


# -------------------- TESTING --------------------

def test_agent(Q):
    state = START
    path = [state]

    while not is_terminal(state):
        action = max(Q[state], key=Q[state].get)
        state = move(state, action)
        path.append(state)

    print("Final Path Taken by Agent:")
    for p in path:
        print(p, end=" -> ")

    if state == TREASURE:
        print("TREASURE")
    else:
        print("TRAP")


# -------------------- RUN --------------------

if __name__ == "__main__":
    Q = train_agent()
    test_agent(Q)
