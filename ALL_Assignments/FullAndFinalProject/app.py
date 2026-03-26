import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from env.traffic_env import TrafficEnv
from agent.dqn_agent import Agent
import os

st.set_page_config(page_title="Traffic RL", layout="wide")

st.title("🚦 AI Traffic Signal Control using RL")

env = TrafficEnv()
agent = Agent(state_dim=5, action_dim=2)

# Load model if exists
if os.path.exists("models/dqn.pth"):
    agent.load("models/dqn.pth")
    agent.epsilon = 0  # disable randomness

st.sidebar.header("Controls")
episodes = st.sidebar.slider("Episodes", 10, 200, 50)
steps = st.sidebar.slider("Steps per Episode", 10, 200, 100)

run = st.sidebar.button("Run Simulation")

def random_policy(env, steps):
    state = env.reset()
    total = 0
    for _ in range(steps):
        action = np.random.randint(0, 2)
        state, reward, _ = env.step(action)
        total += reward
    return total

if run:
    rewards = []

    progress = st.progress(0)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(steps):
            action = agent.select_action(state)
            next_state, reward, _ = env.step(action)

            agent.store(state, action, reward, next_state)
            agent.train()

            state = next_state
            total_reward += reward

        agent.update_target()
        rewards.append(total_reward)

        progress.progress((ep + 1) / episodes)

    st.success("Simulation Done!")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title("Reward vs Episodes")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")

    st.pyplot(fig)

    # Compare
    random_score = random_policy(env, steps)

    st.subheader("Comparison")
    st.write(f"🤖 RL Reward: {total_reward}")
    st.write(f"🎲 Random Reward: {random_score}")

    # Show final state
    st.subheader("Final Traffic State")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("North", int(state[0]))
    col2.metric("South", int(state[1]))
    col3.metric("East", int(state[2]))
    col4.metric("West", int(state[3]))

    signal = "NS GREEN" if action == 0 else "EW GREEN"
    st.write(f"🚦 Final Signal: {signal}")