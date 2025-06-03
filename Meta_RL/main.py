import torch
from meta_ppo import MetaPPo
from ac_network import ActorCritic
from meta_inverted_pendulum_env import MetaInvertedPendulumEnv

meta_env = MetaInvertedPendulumEnv()
meta_policy = ActorCritic(in_dim=4, out_dim=1).to("cpu")


agent = MetaPPo(meta_policy, env=None, inner_lr=0.1, outer_lr=0.001, gamma=0.99, tau=0.95, continuous=True)

num_meta_iterations = 10000
num_tasks_per_batch = 5
num_inner_updates = 1

for iteration in range(num_meta_iterations):
    meta_loss = 0
    grads_list = []

    for _ in range(num_tasks_per_batch):
        task = meta_env.sample_task()
        env = meta_env.make_env(task)

        # Clone policy and adapt to the task
        inner_policy = agent.clone_policy()
        agent.inner_update(env, inner_policy, num_steps=1000)

        # Compute loss after adaptation
        loss, grads = agent.compute_meta_loss(env, inner_policy)
        meta_loss += loss
        grads_list.append(grads)

        env.close()

    agent.outer_update(grads_list)
    # Save meta-policy every N iterations
    if (iteration + 1) % 500 == 0 or iteration == num_meta_iterations - 1:
        torch.save(agent.meta_policy.state_dict(), f"meta_policy_iter_{iteration+1}.pth")
        print(f"✅ Saved checkpoint: meta_policy_iter_{iteration+1}.pth")


    print(f"[{iteration}] Meta Loss: {meta_loss / num_tasks_per_batch:.4f}")
