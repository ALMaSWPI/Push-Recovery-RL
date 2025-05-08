# RL and Domain Randomization Techniques for Learning Push Recovery for Humanoids across different environments 

## Setup 
### 1️⃣ Create a Virtual Environment

```bash python -m venv pendulum_env```
```pendulum_env\Scripts\activate```

### Install dependencies

pip install gymnasium==0.29.1
pip install mujoco==3.1.4
pip install glfw imageio
pip install stable-baselines3[extra]
pip install PyOpenGL PyOpenGL_accelerate  # if rendering fails 

Check installation by running test_mujoco_install.py. It should open the mujoco window

Run train_ip.py to train a single policy (by manually defining the parameters)
Run visualize_ip_video.py to visualize the video output
