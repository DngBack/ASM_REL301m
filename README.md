# ASM_REL301m - Deep Q-Network (DQN) and Double DQN (DDQN) for Atari Games

A comprehensive implementation of Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) algorithms for training AI agents to play Atari games using PyTorch and Gymnasium.

## 🎮 Features

- **DQN Implementation**: Classic Deep Q-Network with experience replay and target networks
- **DDQN Implementation**: Double Deep Q-Network to reduce overestimation bias
- **Atari Game Support**: Currently configured for Galaxian, easily extensible to other Atari games
- **CNN Architecture**: Convolutional neural network for processing game frames
- **Frame Stacking**: Stacks 4 consecutive frames for temporal information
- **Video Recording**: Automatically saves gameplay videos during training
- **Real-time Display**: Watch the agent play in real-time with game information overlay
- **GPU Support**: CUDA acceleration when available
- **Training Visualization**: Real-time reward plotting and progress tracking

## 🏗️ Architecture

### Neural Network Structure

- **Input**: 4 stacked grayscale frames (84x84 pixels)
- **Convolutional Layers**:
  - Conv1: 32 filters, 8x8 kernel, stride 4
  - Conv2: 64 filters, 4x4 kernel, stride 2
  - Conv3: 64 filters, 3x3 kernel, stride 1
- **Fully Connected Layers**:
  - FC1: 512 neurons
  - FC2: Action space size (output layer)

### Key Components

- **Experience Replay Buffer**: Stores and samples past experiences
- **Target Network**: Separate network for stable Q-value estimation
- **Epsilon-Greedy Exploration**: Gradually reduces exploration rate
- **Frame Preprocessing**: Grayscale conversion, resizing, and normalization

## 📋 Requirements

- Python 3.11+
- PyTorch 2.0+
- Gymnasium with Atari support
- CUDA-compatible GPU (optional, for acceleration)

## 🚀 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/ASM_REL301m.git
   cd ASM_REL301m
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import gymnasium; import torch; print('Installation successful!')"
   ```

## 🎯 Usage

### Training DQN Agent

```bash
python dqn_atarin.py
```

### Training DDQN Agent

```bash
python ddqn_atarin.py
```

### Testing Real-time Display

```bash
python test_realtime_display.py
```

### Real-time Display Controls

When running with real-time display enabled:

- **Press 'q'**: Quit training and close display
- **Press 'p'**: Pause/unpause the display
- **Automatic**: Displays game frame with episode info, step count, reward, epsilon, and current action

### Key Parameters (Configurable in the scripts)

- `NUM_EPISODES`: Number of training episodes (default: 500)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Adam optimizer learning rate (default: 0.00025)
- `EPSILON_DECAY`: Exploration rate decay (default: 0.995)
- `TARGET_UPDATE`: Target network update frequency (default: 1000 steps)
- `RENDER_EVERY`: Video recording frequency (default: every 50 episodes)
- `SHOW_REALTIME`: Enable real-time display (default: True)
- `REALTIME_DELAY`: Frame delay for real-time display (default: 0.05s = 20 FPS)

## 📊 Training Process

1. **Environment Setup**: Creates Galaxian Atari environment
2. **Agent Initialization**: Sets up DQN/DDQN agent with CNN
3. **Training Loop**:
   - Preprocess game frames
   - Select actions using epsilon-greedy policy
   - Store experiences in replay buffer
   - Train network on mini-batches
   - Update target network periodically
4. **Monitoring**: Tracks rewards, epsilon decay, and saves videos
5. **Model Saving**: Saves trained model weights

## 📁 Output Files

- **Model Weights**: `dqn_galaxian.pth` or `ddqn_galaxian.pth`
- **Training Videos**: `videos/` directory with gameplay recordings
- **Training Plot**: Real-time reward visualization

## 🔧 Customization

### Changing the Game

Replace `"ALE/Galaxian-v5"` with other Atari games:

```python
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
```

### Adjusting Hyperparameters

Modify the constants at the top of each script:

```python
NUM_EPISODES = 1000  # More episodes for better performance
BATCH_SIZE = 64      # Larger batch size
LEARNING_RATE = 0.0001  # Different learning rate
```

### Network Architecture

Modify the `QNetwork` class to change the CNN structure.

## 🧠 Algorithm Details

### DQN vs DDQN

- **DQN**: Uses target network for both action selection and evaluation
- **DDQN**: Uses policy network for action selection, target network for evaluation

### DDQN Advantage

Reduces overestimation bias by decoupling action selection from action evaluation, leading to more stable training and often better performance.

## 📈 Performance Tips

1. **Training Time**: Expect 2-4 hours for 500 episodes on CPU, faster with GPU
2. **Convergence**: Monitor reward plots for learning progress
3. **Hyperparameter Tuning**: Adjust learning rate and epsilon decay based on performance
4. **Memory Usage**: Large replay buffer (100k experiences) requires sufficient RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the original DQN paper by DeepMind
- Uses Gymnasium for environment management
- PyTorch for deep learning implementation

## 📚 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Note**: This implementation is for educational and research purposes. Training times and performance may vary depending on your hardware configuration.
