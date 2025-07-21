import ale_py  # type: ignore
import shimmy
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
from PIL import Image
from gymnasium.utils.save_video import save_video  # Để lưu video
from plot_utils import save_plots_to_folder

# Hyperparameters
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
BUFFER_SIZE = 100000
LEARNING_RATE = 0.00025
TARGET_UPDATE = 1000  # Update target network every X steps
NUM_EPISODES = 50
FRAME_STACK = 4  # Stack 4 frames
IMG_SIZE = 84  # Resize to 84x84
RENDER_EVERY = 5  # Render và lưu video mỗi X episodes
EVAL_EVERY = 1  # Đánh giá average max Q mỗi X episodes


# Preprocess frame: grayscale, resize, normalize
def preprocess_frame(frame):
    frame = np.array(Image.fromarray(frame).convert("L").resize((IMG_SIZE, IMG_SIZE)))
    return frame / 255.0


# Q-Network (CNN)
class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Main DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.policy_net = (
            QNetwork(action_size).cuda()
            if torch.cuda.is_available()
            else QNetwork(action_size)
        )
        self.target_net = (
            QNetwork(action_size).cuda()
            if torch.cuda.is_available()
            else QNetwork(action_size)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps = 0
        self.epsilon = EPSILON_START
        self.action_size = action_size

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = (
            torch.FloatTensor(state).unsqueeze(0).cuda()
            if torch.cuda.is_available()
            else torch.FloatTensor(state).unsqueeze(0)
        )
        with torch.no_grad():
            return self.policy_net(state).argmax(1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = (
            torch.FloatTensor(np.array(states)).cuda()
            if torch.cuda.is_available()
            else torch.FloatTensor(np.array(states))
        )
        next_states = (
            torch.FloatTensor(np.array(next_states)).cuda()
            if torch.cuda.is_available()
            else torch.FloatTensor(np.array(next_states))
        )
        actions = (
            torch.LongTensor(actions).unsqueeze(1).cuda()
            if torch.cuda.is_available()
            else torch.LongTensor(actions).unsqueeze(1)
        )
        rewards = (
            torch.FloatTensor(rewards).cuda()
            if torch.cuda.is_available()
            else torch.FloatTensor(rewards)
        )
        dones = (
            torch.FloatTensor(dones).cuda()
            if torch.cuda.is_available()
            else torch.FloatTensor(dones)
        )

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# Training Loop cho DQN
env = gym.make("ALE/Galaxian-v5", render_mode="rgb_array")
action_size = env.action_space.n
agent = DQNAgent((FRAME_STACK, IMG_SIZE, IMG_SIZE), action_size)

rewards = []
avg_max_q = []  # List để lưu average max Q-value

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = preprocess_frame(state)
    state_deque = collections.deque([state] * FRAME_STACK, maxlen=FRAME_STACK)
    state_stack = np.stack(state_deque, axis=0)

    total_reward = 0
    done = False
    frames = []  # List để lưu frames cho video
    if episode % RENDER_EVERY == 0:
        print("Rendering frame")
        frames.append(env.render())  # Render frame đầu tiên

    while not done:
        action = agent.act(state_stack)
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_frame(next_state)
        state_deque.append(next_state)
        next_state_stack = np.stack(state_deque, axis=0)

        # Clip reward for stability
        reward = np.clip(reward, -1, 1)
        agent.remember(state_stack, action, reward, next_state_stack, done)
        agent.replay()

        state_stack = next_state_stack
        total_reward += reward

        if episode % RENDER_EVERY == 0:
            frames.append(env.render())  # Lưu frame sau mỗi step nếu cần render

    rewards.append(total_reward)
    print(
        f"Episode {episode + 1}/{NUM_EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}"
    )

    # Lưu video nếu episode cần render
    if episode % RENDER_EVERY == 0 and len(frames) > 0:
        print("Saving video")
        try:
            # print(frames)
            save_video(
                frames,
                video_folder="videos",  # Thư mục lưu video (tạo nếu chưa có)
                fps=30,  # FPS video
                # episode_index=episode,
                name_prefix="galaxian_dqn",
            )
            print(
                f"Saved video for episode {episode + 1} to videos/galaxian_dqn-episode-{episode}.mp4"
            )
        except Exception as e:
            print(f"Error saving video for episode {episode + 1}: {e}")

    # Đánh giá average max Q-value mỗi EVAL_EVERY episodes
    if episode % EVAL_EVERY == 0:
        eval_states = []
        # Tạo một environment riêng cho việc đánh giá để không ảnh hưởng đến training
        eval_env = gym.make("ALE/Galaxian-v5", render_mode="rgb_array")
        for _ in range(100):  # Sample 100 states ngẫu nhiên
            eval_state, _ = eval_env.reset()
            eval_state = preprocess_frame(eval_state)
            eval_stack = np.stack([eval_state] * FRAME_STACK, axis=0)
            eval_states.append(eval_stack)
        eval_env.close()

        eval_states = (
            torch.FloatTensor(np.array(eval_states)).cuda()
            if torch.cuda.is_available()
            else torch.FloatTensor(np.array(eval_states))
        )
        with torch.no_grad():
            max_q = agent.policy_net(eval_states).max(1)[0].mean().item()
        avg_max_q.append(max_q)

env.close()

# Plot 1: Reward Curve (biểu đồ cơ bản cho reward)
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Reward Curve on Galaxian")
plt.grid(True)
save_plots_to_folder("plots")
plt.show()

# Plot 2: Average Max Q-Value Over Time (biểu đồ cơ bản cho Q estimates)
eval_points = list(range(0, NUM_EPISODES, EVAL_EVERY))
plt.figure(figsize=(10, 5))
plt.plot(eval_points[: len(avg_max_q)], avg_max_q)
plt.xlabel("Episode")
plt.ylabel("Average Max Q-Value")
plt.title("DQN Average Max Q-Value Estimates Over Training")
plt.grid(True)
save_plots_to_folder("plots")
plt.show()

# Save model
torch.save(agent.policy_net.state_dict(), "dqn_galaxian.pth")

# Phần riêng: Plot Toy Overestimation Bias (biểu đồ minh họa bias cơ bản cho DQN-like)
num_actions_list = [2, 5, 10, 20, 50]
num_samples = 10000
bias_dqn = []  # Plot cho DQN-like để minh họa bias cao

for num_actions in num_actions_list:
    true_q = np.zeros(num_actions)  # True Q=0
    errors = np.random.normal(0, 1, (num_samples, num_actions))  # Noise

    # DQN-like: max(Q + error)
    max_q = np.max(true_q + errors, axis=1)
    bias_dqn.append(np.mean(max_q))

plt.figure(figsize=(10, 5))
plt.plot(num_actions_list, bias_dqn, label="DQN-like Bias", marker="o")
plt.xlabel("Number of Actions")
plt.ylabel("Average Bias (Overestimation)")
plt.title("Overestimation Bias in Toy Environment (DQN-like)")
plt.legend()
plt.grid(True)
save_plots_to_folder("plots")
plt.show()
