import random

# 🎯 Environment settings
GRID = 4
ACTIONS = ['u', 'd', 'l', 'r']

# Rewards
TREASURE_REWARD = 10
TRAP_PENALTY = -10
STEP_PENALTY = -1

# Learning parameters
alpha = 0.8
alpha_decay = 0.995
gamma = 0.9
eps = 1.0
eps_decay = 0.995
min_eps = 0.01

# Positions
start = (0, 0)
treasure = (3, 3)
trap = (2, 2)


# 🚶 Movement function (same style as your code)
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


# 📊 Initialize Q-table using dictionary
def initialize_q_table():
    Q = {}
    for i in range(GRID):
        for j in range(GRID):
            Q[(i, j)] = {}
            for a in ACTIONS:
                Q[(i, j)][a] = 0.0
    return Q


# 🎲 ε-greedy action selection
def take_action(Q, state):
    if random.uniform(0, 1) < eps:
        return random.choice(ACTIONS)     # 🔍 Exploration
    else:
        return max(Q[state], key=Q[state].get)  # 🧠 Exploitation


# 🎁 Reward function
def get_reward(state):
    if state == treasure:
        return TREASURE_REWARD
    elif state == trap:
        return TRAP_PENALTY
    else:
        return STEP_PENALTY


# ⛔ Terminal check
def is_terminal(state):
    return state == treasure or state == trap


# 🧮 Q-value update (exact formula implementation)
def update_q_value(Q, state, action, reward, next_state):
    global alpha
    max_next_q = max(Q[next_state].values())

    Q[state][action] = Q[state][action] + alpha * (
        reward + gamma * max_next_q - Q[state][action]
    )


# 🧭 Print learned policy
def print_policy(Q):
    print("\n🧭 Final Learned Policy:")
    for i in range(GRID):
        for j in range(GRID):
            state = (i, j)
            if state == treasure:
                print("💎", end="  ")
            elif state == trap:
                print("💣", end="  ")
            else:
                best_action = max(Q[state], key=Q[state].get)
                print(best_action.upper(), end="  ")
        print()


# 🚀 Training loop
def train():
    global alpha, eps

    Q = initialize_q_table()
    episodes = 1000

    for ep in range(1, episodes + 1):
        state = start

        while not is_terminal(state):
            action = take_action(Q, state)
            next_state = move(state, action)
            reward = get_reward(next_state)

            # 🔁 Update Q(s,a)
            update_q_value(Q, state, action, reward, next_state)

            state = next_state

        # 📉 Decay learning rate and exploration rate
        alpha *= alpha_decay
        eps = max(min_eps, eps * eps_decay)

        if ep % 100 == 0:
            print(f"Episode {ep} 🏁 | Alpha = {alpha:.3f} | Epsilon = {eps:.3f}")

    print("\n🎉 Training completed successfully!")
    print_policy(Q)


# ▶️ Run
if __name__ == "__main__":
    train()
