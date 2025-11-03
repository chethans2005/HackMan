# 🎮 Hangman Solver: HMM + Deep Q-Learning

A sophisticated AI agent that combines **Hidden Markov Model (HMM) pattern analysis** with **Deep Q-Learning** to solve Hangman games with high accuracy.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Training Details](#training-details)
- [Results](#results)
- [Improving Performance](#improving-performance)
- [File Descriptions](#file-descriptions)

---

## 🎯 Overview

This project trains an intelligent Hangman solver that:

1. **Analyzes word patterns** using HMM to predict letter probabilities
2. **Makes strategic decisions** using a Deep Q-Network (DQN)
3. **Learns from experience** through reinforcement learning with experience replay
4. **Evaluates performance** on both training and test sets
5. **Provides interactive gameplay** through a Gradio web interface

### Key Features

✅ **HMM Pattern Matching** - Uses trigrams, bigrams, and unigrams for letter probability estimation  
✅ **Deep Q-Learning** - Neural network learns optimal guessing strategies  
✅ **Double DQN** - Reduces overestimation bias in Q-value predictions  
✅ **Experience Replay** - Efficient learning from mini-batches of past experiences  
✅ **Interactive Interface** - Play against the AI and visualize decision-making  
✅ **Comprehensive Logging** - Track training progress with detailed statistics  
✅ **GPU Support** - Automatic CUDA detection for faster training  

---

## 🏗️ Architecture

### State Vector (58 dimensions)

The agent observes the game state as a 58-element vector:

```
[26 HMM Probabilities] + [26 Guessed Mask] + [6 Context Features]
```

**HMM Probabilities (26):**
- Probability of each letter (a-z) based on the current word pattern
- Computed using n-gram language models trained on the corpus

**Guessed Mask (26):**
- Binary vector: 1 if letter was guessed, 0 otherwise
- Prevents the agent from selecting invalid actions

**Game Context (6):**
- Lives remaining (normalized)
- Progress ratio (revealed letters / word length)
- Remaining blanks ratio
- Guess exhaustion (guesses made / 26)
- Normalized word length
- Urgency flag (1 if lives ≤ 2, else 0)

### Deep Q-Network (DQN)

```
Input (58) → FC Layer (256) → ReLU + Dropout
         → FC Layer (256) → ReLU + Dropout
         → FC Layer (128) → ReLU
         → Output (26)  [Q-values for all letters]
```

**Key Features:**
- **Dropout (0.2)** - Prevents overfitting
- **Gradient Clipping** - Stabilizes training
- **Double DQN** - Uses two networks (Q-network and Target network)
- **Target Network Update** - Periodic synchronization reduces instability

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch (CPU or GPU)
- NumPy, Matplotlib, Gradio
- Jupyter Notebook

### Setup

1. **Clone or download** the project files
2. **Install dependencies:**

```bash
pip install torch numpy matplotlib gradio tqdm
```

3. **Prepare corpus file:**
   - Ensure `corpus.txt` contains one word per line (lowercase, alphabetic only)
   - Recommended: 10,000+ unique words for better training

4. **Prepare test file:**
   - Create `test.txt` with test words (one per line)
   - Use words NOT in the training corpus for accurate evaluation

---

## 🚀 Usage

### Step 1: Run the Notebook

Open `hangman_hmm_rl_train.ipynb` in Jupyter and run cells sequentially:

```python
# Cell 1-2: Import libraries and load corpus
# Cells 3-6: Build n-gram language models
# Cell 7: Initialize HMM Pattern Matcher
# Cell 8-13: Define game environment and neural network
# Cell 14-15: Initialize and train agent
# Cell 16-20: Evaluate on test set and visualize results
# Cell 21: Launch interactive Gradio interface
```

### Step 2: Train the Agent

```python
stats = train_agent(
    agent=agent,
    corpus=corpus,
    num_episodes=10000,        # Increase for better performance
    batch_size=64,
    target_update_freq=100,
    save_freq=2000            # Saves checkpoint every 2000 episodes
)
```

**Training takes ~30-60 minutes** for 10,000 episodes (GPU recommended)

### Step 3: Evaluate Performance

```python
test_results = evaluate_agent(agent, test_words, verbose=True)
print(f"Success rate: {test_results['success_rate']*100:.2f}%")
print(f"Final score: {test_results['final_score']:.2f}")
```

### Step 4: Interactive Play

```python
demo = create_gradio_interface()
demo.launch(share=True)
```

Then open the provided link in your browser and play!

---

## 📁 Project Structure

```
Hackathon/
├── hangman_hmm_rl_train.ipynb    # Main training notebook
├── corpus.txt                     # Dictionary of training words
├── test.txt                       # Test set words
├── README.md                      # This file
│
├── checkpoint_ep2000.pth         # Saved model at 2000 episodes
├── checkpoint_ep4000.pth         # Saved model at 4000 episodes
├── checkpoint_ep6000.pth         # Saved model at 6000 episodes
├── checkpoint_ep8000.pth         # Saved model at 8000 episodes
├── checkpoint_ep10000.pth        # Saved model at 10000 episodes
│
├── hangman_hmm_rl_final.pth      # Final trained model
├── hmm_components.pkl           # Saved HMM (trigrams, bigrams, corpus)
├── training_progress.png        # Training visualization
├── test_results.json            # Test evaluation results
└── hangman_dqn_model.pt         # Alternative model format
```

---

## 🔑 Key Components

### 1. HMMPatternMatcher

Analyzes the current game pattern and estimates letter probabilities using:
- **Exact pattern matching** - Finds words matching the current pattern
- **N-gram language model** - Falls back to trigram scoring when pattern has no matches
- **Character frequency analysis** - Computes probability for each letter

```python
hmm_probs = hmm_matcher.get_letter_probabilities(pattern, guessed_letters)
# Returns: 26-element array with probability for each letter
```

### 2. HangmanGame

Simulates the Hangman game environment:
- Tracks word, lives, pattern, and guessed letters
- Rewards: +10 for winning, -10 for losing, +1 per correct guess, -1 for wrong
- Penalty: -2 for repeated guesses

```python
game = HangmanGame("python", max_lives=6)
reward, done = game.guess("e")  # Guess letter 'e'
```

### 3. RLAgent

Manages the learning process:
- Builds state vectors from game state
- Selects actions using ε-greedy policy
- Trains DQN via mini-batch gradient descent
- Manages experience replay buffer

```python
agent = RLAgent(hmm_matcher, state_size=58, action_size=26)
action = agent.select_action(state_vector, valid_actions)
loss = agent.train_step(batch_size=64)
```

### 4. ReplayBuffer

Stores and samples experiences for training:
- Capacity: 100,000 transitions
- Enables efficient mini-batch learning

---

## 📊 Training Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 0.0005 | Adam optimizer learning rate |
| `gamma` | 0.95 | Discount factor (how much future rewards matter) |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.9995 | Decay per episode (slower = more exploration) |
| `batch_size` | 64 | Mini-batch size for training |
| `target_update_freq` | 100 | Update target network every N episodes |
| `num_episodes` | 10000 | Total training episodes |

### Training Loop Workflow

For each episode:
1. Pick random word from corpus
2. Initialize game (6 lives, all blanks)
3. **Loop until game ends:**
   - Build state vector (HMM + context)
   - Select action (ε-greedy)
   - Execute guess and get reward
   - Store in replay buffer
   - Train neural network on mini-batch
4. Decay epsilon
5. Update target network (every 100 episodes)
6. Save checkpoint (every 2000 episodes)

---

## 📈 Results

### What to Expect

| Metric | After 2000 ep | After 5000 ep | After 10000 ep |
|--------|--------------|--------------|----------------|
| Training Win Rate | 40-50% | 70-80% | 85-95% |
| Avg Wrong Guesses | 3.5 | 2.5 | 1.5 |
| Training Loss | 0.5-1.0 | 0.1-0.2 | 0.05-0.1 |

### Test Set Performance

Expected performance on unseen test words:
- **Success Rate:** 75-85%
- **Avg Wrong Guesses:** 2.0-2.5
- **Avg Repeated Guesses:** 0.1-0.3

### Key Plots

Run the visualization cell to generate `training_progress.png` with 6 plots:

1. **Win Rate** - Should increase and plateau
2. **Episode Reward** - Should increase over time
3. **Epsilon Decay** - Should decrease exponentially
4. **Wrong Guesses** - Should decrease (better strategy)
5. **Repeated Guesses** - Should stay near 0
6. **Training Loss** - Should decrease and stabilize

---

## 🚀 Improving Performance (1-Week Timeline)

### Phase 1: Enhance Foundation (Days 1-2)

**Expand Corpus:**
- Use a larger word list (50,000+ words)
- Download from SCOWL or Google N-grams
- Include diverse word lengths and patterns

**Refine HMM:**
- Build 4-grams (more context)
- Add positional analysis (letter likelihood at position i)

### Phase 2: Feature Engineering (Days 2-3)

**Add State Features:**
```python
# Add vowel/consonant ratios
guessed_vowels = sum(1 for l in guessed if l in "aeiou")
state_vector.append(guessed_vowels / 5.0)

# Add HMM confidence
state_vector.append(hmm_probs.max())
```

**Update state_size to 61** (was 58)

### Phase 3: Hyperparameter Tuning (Days 3-5)

**Longer Training:**
- Train for 50,000-100,000 episodes
- Slower epsilon decay: `epsilon_decay=0.9999`

**Network Architecture:**
- Experiment: `512 → 256 → 128` (wider)
- Or: `256 → 256 → 256 → 128` (deeper)

**Learning Rate Scheduling:**
- Start with 0.001, decay after 10,000 episodes
- Or use learning rate scheduler

### Phase 4: Validation Split (Days 5-6)

**Prevent Overfitting:**
1. Split corpus: 90% train, 10% validation
2. Evaluate every 1000 episodes on validation set
3. Save best model (by validation win rate, not final)
4. Plot training vs validation curves

### Phase 5: Advanced Techniques (Days 6-7)

**Dueling DQN:**
- Separate value and advantage streams
- Better feature learning

**Prioritized Experience Replay:**
- Sample important experiences more often
- Faster convergence

**Noisy Networks:**
- Parametric exploration instead of ε-greedy
- Better exploration efficiency

---

## 🎮 Interactive Mode

### Play Against the AI

The Gradio interface shows:

```
Pattern: _ _ _ _ _ _
Lives: 4/6
Guessed: a, e, t, r

🧠 HMM Top 5: s(0.18), n(0.15), i(0.12), o(0.11), l(0.09)
🎯 Q-Top 3: s(2.5), n(2.1), o(1.8)
✅ Agent chooses: 's' → CORRECT! (+1)
```

Each step shows:
- **Current pattern** and game state
- **HMM predictions** - What letter pattern matching suggests
- **Q-values** - What the neural network learned to do
- **Agent choice** and result

---

## 📝 File Descriptions

### `hangman_hmm_rl_train.ipynb`
Main Jupyter notebook with 21 cells:
- Data loading and preprocessing
- HMM and game environment setup
- Neural network definition
- Training loop
- Evaluation and visualization
- Interactive Gradio interface

### `corpus.txt`
Dictionary of training words (one per line, lowercase)

### `test.txt`
Test set words for evaluation (should be different from corpus)

### `checkpoint_ep*.pth`
PyTorch checkpoints saved during training containing:
- Q-network weights
- Target network weights
- Optimizer state
- Training statistics

### `hangman_hmm_rl_final.pth`
Final trained model (can be loaded separately)

### `hmm_components.pkl`
Pickled HMM components (trigrams, bigrams, corpus) for later use

### `training_progress.png`
6-subplot visualization of training metrics

### `test_results.json`
JSON file with test evaluation metrics

---

## 🔍 Diagnosing Performance Issues

### Model is Underfitting
- **Signs:** Training loss stays high and flat; win rate stays low
- **Solution:** 
  - Train longer (more episodes)
  - Increase network capacity
  - Improve corpus quality
  - Adjust learning rate upward

### Model is Overfitting
- **Signs:** Training win rate → 95%+, but test win rate → 60-70%
- **Solution:**
  - Use validation split to monitor generalization
  - Increase dropout
  - Expand corpus (more diversity)
  - Add more regularization
  - Save best validation model (not final)

### Training is Unstable
- **Signs:** Loss oscillates wildly; win rate fluctuates
- **Solution:**
  - Reduce learning rate (try 0.0001)
  - Increase batch size (try 128)
  - Enable gradient clipping (already implemented)
  - Use Double DQN (already implemented)

---

## 💾 Saving and Loading Models

### Save After Training

```python
torch.save({
    'q_network_state_dict': agent.q_network.state_dict(),
    'target_network_state_dict': agent.target_network.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
}, 'my_model.pth')
```

### Load Pretrained Model

```python
checkpoint = torch.load('hangman_hmm_rl_final.pth')
agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
```

---

## 📚 References

- **Reinforcement Learning:** Sutton & Barto, "Reinforcement Learning: An Introduction"
- **Deep Q-Learning:** Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
- **Double DQN:** Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning"
- **HMM:** Rabiner & Juang, "Fundamentals of Speech Recognition"

---

## 📧 Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `batch_size=32`
- Reduce network size (fewer neurons)
- Use CPU: `torch.device('cpu')`

### Corpus Loading Fails
- Check file encoding: `encoding='utf-8'`
- Ensure file exists in working directory
- Check for null bytes or unusual characters

### Gradio Won't Launch
- Ensure Gradio is installed: `pip install gradio`
- Try without `share=True`: `demo.launch()`
- Check firewall settings

### Model Training Takes Too Long
- Use GPU (check `torch.cuda.is_available()`)
- Reduce num_episodes
- Increase batch_size (faster per update)
- Reduce save_freq

---

## 🎓 Learning Outcomes

By studying this project, you'll understand:

✅ Hidden Markov Models and pattern matching  
✅ Deep Q-Learning and value-based RL  
✅ Neural network design and optimization  
✅ Experience replay and off-policy learning  
✅ Exploration vs exploitation trade-off  
✅ State representation engineering  
✅ Model evaluation and validation  
✅ Hyperparameter tuning strategies  

---

## 📄 License

This project is created for educational purposes.

---

## 👤 Author

**Chethan S**  
ML Hackathon Project - PES University

---

**Happy learning! 🚀**

For questions or improvements, refer to the inline comments in `hangman_hmm_rl_train.ipynb`.
