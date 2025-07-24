# Hangman Solver using Deep Reinforcement Learning

This repository contains the code and resources for a sophisticated Hangman-solving agent. The agent is built using a **Dueling Double Deep Q-Network (Dueling Double DQN)** and leverages a pre-trained **TinyBERT** model to understand the game's state and make intelligent guesses.

The project is designed to be a comprehensive implementation of modern reinforcement learning techniques applied to a classic word game.

## Key Features

* **Advanced RL Architecture**: Implements a Dueling Double DQN, which stabilizes training and improves performance by separating state value and action advantage streams.
* **Natural Language Understanding**: Uses a pre-trained TinyBERT model from Hugging Face (`prajjwal1/bert-tiny`) as the backbone for state representation, allowing the agent to process the hangman puzzle as a linguistic challenge.
* **Efficient Training**: Employs an experience replay buffer to store and sample game transitions, breaking temporal correlations and improving learning stability.
* **Optimized Action Selection**: Uses an epsilon-greedy policy for a balance between exploration (random guesses) and exploitation (using the learned policy). Illegal moves (guessing the same letter twice) are masked out.
* **Robust Environment**: The `HangmanEnv` class provides a clean and efficient simulation of the game, complete with a well-defined state, action space, and reward structure.

## How It Works

1.  **State Representation**: The current state of the hangman puzzle (e.g., `_ p p _ _`) is converted into a string and tokenized using the BERT tokenizer.
2.  **Q-Value Estimation**: The tokenized input is fed into the TinyBERT model. The output from BERT is then passed to two separate neural network heads:
    * A **Value Stream** that estimates the value of the current state ($V(s)$).
    * An **Advantage Stream** that calculates the advantage for each possible action ($A(s,a)$).
3.  **Action Selection**: The Q-values for all possible actions are combined from the value and advantage streams. The agent then selects the best legal action based on these Q-values or explores a random action.
4.  **Learning**: The agent plays thousands of games, storing the results of each move in its replay memory. It periodically samples batches from this memory to update its neural network weights, gradually improving its guessing strategy. The target network is updated less frequently to provide a stable learning target.

## Getting Started

### Prerequisites

* Python 3.7+
* PyTorch
* Transformers (by Hugging Face)
* NumPy
* scikit-learn
* tqdm

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/abhiraj-ranjan>
    cd <PretrainedRLHangman>
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Training the Agent

To train the agent, run the `pretrained_rl_hangman.py` script with the required arguments. You will need to provide a path to a word list and a path to save the trained model.

```bash
python hangman_rl_agent.py \
    --words /path/to/words_250000_train.txt \
    --save best_model.pth \
    --episodes 50000 \
    --batch-size 128 \
    --lr 5e-5 \
    --gamma 0.99 \
    --memory-size 100000 \
    --target-update 500 \
    --eval-interval 1000
