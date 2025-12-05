🍄 Super Mario Bros. D3QN Agent

This project implements a Dueling Double Deep Q-Network (D3QN) agent to learn how to play Super Mario Bros. The agent uses a custom environment and a PyTorch-based implementation of the D3QN algorithm.


 🛠️ Prerequisites

   Python (3.7+)
   The Super Mario Bros. ROM is required to run the environment.

 ⚙️ Setup and Installation

1.  Clone the repository:

    bash
    git clone https://github.com/singhaayush01/mario-rl 
    cd super-mario-d3qn-agent
    

    (Note: Replace https://github.com/singhaayush01/mario-rl  with the actual link to your GitHub repository.)

2.  Install dependencies:

    bash
    pip install -r requirements.txt
    

    This typically includes libraries like torch, gym-super-mario-bros, nes-py, etc.

 🏃 How to Run the Program (Play)

To watch the pre-trained agent (mario_d3qn_best.pth) play the game, run the evaluation script:

bash
python play_mario_smart.py


This script will load the saved weights and render the game window, showing the agent's performance.

 🧠 How to Train the Agent

To start or resume training the D3QN agent, use the main training script. By default, this script will save model checkpoints to the current directory and logs to the runs/ folder.

1.  Start New Training:

    bash
    python play_mario.py --new-session
    

    (The --new-session flag ensures that a fresh training run is started, ignoring any interrupted checkpoints.)

2.  Resume Training:

    bash
    python play_mario.py
    

    (If a file like mario_d3qn_interrupted.pth exists, the script will automatically load it and continue training from that point.)



 🔗 GitHub Repository

The source code for this project is hosted on GitHub:

https://github.com/singhaayush01/mario-rl 

