from env.traffic_env import TrafficEnv
from agent.dqn_agent import Agent
import os





env = TrafficEnv()
agent = Agent(state_dim=5, action_dim=2)

episodes = 300

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, _ = env.step(action)

        agent.store(state, action, reward, next_state)
        agent.train()

        state = next_state
        total_reward += reward

    agent.update_target()

    print(f"Episode {ep+1}, Reward: {total_reward}")

# create folder if not exists
os.makedirs("models", exist_ok=True)

agent.save("models/dqn.pth")
print("Model saved!")