import os.path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from matplotlib import pyplot as plt
import utils
from RL_Algorithm.DDPG.network.Actor import Actor
from RL_Algorithm.DDPG.network.Critic import Critic
from RL_Algorithm.DDPG.utils.OUNoise import OUNoise
from RL_Algorithm.DDPG.utils.ReplayBuffer import ReplayBuffer
from SB3.environment.withBandwidth import CustomEnv
from config import ROOT_DIR

class DDPG:
    """DDPGAgent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
            self,
            env: CustomEnv,
            memory_size: int,
            batch_size: int,
            ou_noise_theta: float,
            ou_noise_sigma: float,
            gamma: float = 0.99,
            tau: float = 5e-3,
            initial_random_steps: int = 1e4,
    ):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )

        # device: cpu / gpu
        self.device = torch.device("mps")
        print(self.device)

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()

        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray):
        """Take an action and return the response of the env."""
        next_state, reward, done, _, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks

        # train critic
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target update
        self._target_soft_update()

        return actor_loss.data, critic_loss.data

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state, _ = self.env.reset()
        actor_losses = []
        critic_losses = []
        scores = []
        score = 0

        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if (
                    len(self.memory) >= self.batch_size
                    and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            # plotting
            if self.total_step % plotting_interval == 0:
                self._plot(
                    self.total_step,
                    scores,
                    actor_losses,
                    critic_losses,
                )

        self.env.close()

    def test(self):
        """Test the agent."""
        self.is_test = True
        self.env.unwrapped.isEvaluation = True

        frames = []
        for i in range(10):
            state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                # frames.append(self.env.render(mode="rgb_array"))
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
            print(f"score of episode {i}: ", score)

        remainingEnergy = self.env.unwrapped.episode_remainingEnergy[1:]
        remainingEnergyVariance = self.env.unwrapped.episode_remainingEnergyVariance[1:]
        iotBW = self.env.unwrapped.episode_effectiveBW[1:]
        import random
        for i in range(len(remainingEnergy)):
            plt.figure(figsize=(int(25), int(5)))
            for k in range(self.env.unwrapped.iotDeviceNum):
                iotDevice_K = []
                for j in range(1, len(remainingEnergy[i])):
                    iotDevice_K.append(remainingEnergy[i][j][k])
                r = random.random()
                b = random.random()
                g = random.random()
                color = (r, g, b)
                plt.subplot(2, 1, 1)
                plt.title(f"Remaining Energy of iot device - episode {i}")
                plt.xlabel("timestep")
                plt.ylabel("remaining energy")
                plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
                plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(remainingEnergyVariance[i][:-1], color='black', linewidth='3', label=f"Variance")
            plt.xlabel("timestep")
            plt.ylabel("remaining energy variance")
            plt.savefig(os.path.join(f"{ROOT_DIR}/DDPG/Graphs", f"Remaining Energy - episode {i}.png"))
            plt.close()

        consumedEnergy = self.env.unwrapped.episode_consumedEnergy[1:]
        for i in range(len(consumedEnergy)):
            color = []
            plt.figure(figsize=(int(40), int(10)))
            x = [i for i in range(len(consumedEnergy[i]) - 1)]
            for k in range(self.env.unwrapped.iotDeviceNum):
                actionIndex = k * 2
                iotDevice_K = []
                for j in range(1, len(consumedEnergy[i])):
                    iotDevice_K.append(consumedEnergy[i][j][k])
                r = random.random()
                b = random.random()
                g = random.random()
                color.append((r, g, b))
                plt.subplot(3, 1, 1)
                plt.plot(x, iotDevice_K, color=color[k], marker='o', label=f"Device {k}")
                plt.legend()
                plt.ylabel("consumed energy")

                iotDevice_K_BW = []
                for j in range(1, len(iotBW[i])):
                    iotDevice_K_BW.append(iotBW[i][j][k])
                plt.subplot(3, 1, 2)
                plt.plot(x, iotDevice_K_BW, color=color[k], linewidth='2', marker='o', label=f"BW of Device {k}")
                plt.legend()
                plt.ylabel("Bandwidth")
            plt.title(f"Consumed Energy of iot device - episode {i}")
            plt.xlabel("timestep")
            plt.savefig(os.path.join(f"{ROOT_DIR}/DDPG/Graphs", f"Consumed Energy - episode {i}.png"))
            plt.close()

        self.env.close()

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            mps_tensor = torch.tensor(values).to(torch.float32).to("mps")
            cpu_Tensor = mps_tensor.cpu()
            numpy_array = cpu_Tensor.numpy()
            plt.plot(numpy_array)

        subplot_params = [
            (311, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (312, "actor_loss", actor_losses),
            (313, "critic_loss", critic_losses),
        ]

        clear_output(True)
        plt.figure(figsize=(40, 10))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.savefig("DDPG.png")
        plt.close()

        x = [i for i in range(len(self.env.unwrapped.episode_reward))]
        reward = self.env.unwrapped.episode_reward
        rewardOfEnergy = self.env.unwrapped.episode_energy_reward
        rewardOfTT = self.env.unwrapped.episode_tt_reward
        rewardOfRemainingEnergy = self.env.unwrapped.episode_remainingEnergy_reward
        tt = self.env.unwrapped.episode_tt
        energy = self.env.unwrapped.episode_energy
        classicFLEnergy = self.env.unwrapped.episode_classicFL_energy
        classicFLTT = self.env.unwrapped.episode_classicFL_TT
        utils.saveGraphs(savePath=f"{ROOT_DIR}/DDPG/Graphs", energy=energy, rewardOfEnergy=rewardOfEnergy,
                         rewardOfTrainingTime=rewardOfTT,
                         trainingTime=tt, reward=reward, x=x, allEnergy=self.env.unwrapped.episode_energy,
                         allTrainingTime=self.env.unwrapped.episode_tt, classicFL_trainingTime=classicFLTT,
                         classicFL_energy=classicFLEnergy, rewardOfRemainingEnergy=rewardOfRemainingEnergy)
