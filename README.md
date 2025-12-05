 Super Mario Bros. D3QN Agent 🍄

  

This project implements a Double Deep Q-Network (D3QN) agent trained to master Super Mario Bros using OpenAI Gym (`gym-super-mario-bros`) and PyTorch. The agent utilizes Convolutional Neural Networks (CNNs) and specialized environment wrappers to perceive and navigate the game world effectively.

 🚀 Getting Started

 1\. Prerequisites

Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

 2\. Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

> Note on Dependencies:
> This project relies on `nes-py` and `gym-super-mario-bros`. If you encounter version compatibility issues, try installing these specific versions:
>
> ```bash
> pip install gym-super-mario-bros==7.3.0
> pip install nes-py==8.2.1
> ```

-----

 🎮 Usage

 Run the Pre-Trained Agent

To watch the agent play using the best-performing weights:

Run the Smart/Best Agent:

```bash
python3 play_mario_smart.py
```

Run the Standard Agent:

```bash
python3 play_mario.py
```

Both scripts typically load the `mario_d3qn_best.pth` or `mario_d3qn_interrupted.pth` checkpoints.

 Train From Scratch (Optional)

If you wish to retrain the model from the beginning:

```bash
python3 train.py
```

(Note: Training Deep RL models on CPU can take a significant amount of time. A GPU is recommended.)

-----

 📁 Project Structure

Here is an overview of the core files in this repository:

| File | Description |
| :--- | :--- |
| `play_mario_smart.py` | Main Entry Point. Loads the optimized D3QN logic and best weights to demonstrate high-level gameplay. |
| `play_mario.py` | Runs the agent using standard parameters. |
| `model.py` | Defines the CNN Architecture. This neural network processes raw pixel data to estimate Q-values. |
| `policy.py` | Implements the D3QN Agent. Handles action selection ($\epsilon$-greedy), network updates, and weight saving/loading. |
| `memory.py` | Implements Experience Replay. Stores past transitions (state, action, reward) to stabilize training via batch sampling. |
| `environment.py` | Custom Wrappers. Handles frame stacking, gray-scaling, and action-space simplification for the gym environment. |
| `mario_d3qn_best.pth` | The saved model weights for the best-performing agent. |
| `runs/` | Contains TensorBoard logs (rewards, loss, training metrics). |

-----

 📊 Monitoring

To visualize training progress, loss, and rewards, you can launch TensorBoard pointing to the logs directory:

```bash
tensorboard --logdir runs
```

-----

 📝 Acknowledgments

   Frameworks: [PyTorch](https://pytorch.org/), [OpenAI Gym](https://gym.openai.com/)
   Environment: [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/)
