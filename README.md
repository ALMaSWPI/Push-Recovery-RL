# RL and Domain Randomization Techniques for Learning Push Recovery for Humanoids across different environments 

## Overview

The project focuses on developing control policies that allow humanoid robots to maintain balance when subjected to external forces (pushes) in varied environments. By using domain randomization during training, the policies become more robust and can generalize to unseen conditions.

## Installation

### Prerequisites

- Python 3.8+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/humanoid-push-recovery.git
   cd humanoid-push-recovery
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv pendulum_env
   ```

3. **Activate the environment**
   - Windows:
     ```bash
     pendulum_env\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source pendulum_env/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install gymnasium==0.29.1
   pip install mujoco==3.1.4
   pip install glfw imageio
   pip install stable-baselines3[extra]
   pip install PyOpenGL PyOpenGL_accelerate  # if rendering fails
   ```

5. **Verify installation**
   ```bash
   python test_mujoco_install.py
   ```
   This should open a MuJoCo window, confirming successful installation.

## Usage

### Training

To train a policy with default parameters:

```bash
python train_ip.py
```
The parameters can be changed inside the code.

### Evaluation

Visualize the trained policy:

```bash
python visualize_ip_video.py 
```
