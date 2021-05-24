import cv2
import gym
import torch
import numpy as np

from src.utils.general_functions import torch_from_frame


class AgentEvaluation:
    def __init__(self, model, path, device):
        self.checkpoint = torch.load(path)
        self.device = device

        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.eval()
        self.model = model

        self.env = None

    def play(self, n, env, skip_n=4, render=True):
        episodes_played = 0
        rewards = []
        is_done = True
        while episodes_played < n:
            if is_done:
                state = torch_from_frame(env.reset(), self.device)
                if render:
                    env.render()
                state = torch.cat([state] * 4, dim=1)
                is_done = False
                episode_reward = 0.0

            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            next_state = None
            for _ in range(skip_n):
                new_state, reward, is_done, _ = env.step(action.item())
                if render:
                    env.render()
                new_state = torch_from_frame(new_state, self.device)
                episode_reward += reward
                if is_done:
                    break
                next_state = (
                    new_state
                    if next_state is None
                    else torch.cat((next_state, new_state), dim=1)
                )
            state = next_state

            if is_done:
                rewards.append(episode_reward)
                episodes_played += 1
        return rewards

    def record(self, env, path):
        wrapped_env = gym.wrappers.Monitor(env, path, force=True)
        self.play(1, wrapped_env, render=False)
        wrapped_env.close()
        env.close()


class AtariEvaluator(AgentEvaluation):
    def play(self, n, env, skip_n=4, render=True):
        episodes_played = 0
        rewards = []
        is_done = True
        self.env = env
        while episodes_played < n:
                # If episode has finished, start a new one
            if is_done:
                episode_count += 1
                state = self._get_initial_state(skip_n)
                if render:
                    self.env.render()
                episode_reward = 0.0
                episode_loss = 0.0
                is_done = False
                #while(self.env.t<1):
                #    self.env.step(8)
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            new_state, reward, is_done, _ = env.step(action.item())
            episode_reward += reward
            if render:
                self.env.render()
            state = next_state
            
            if is_done:
                rewards.append(episode_reward)
                episodes_played += 1
                self.env.close()
 
        return rewards

    def _get_initial_state(self, skip_n):
        return self.env.reset()
    
    def _get_state_from_frame(self, frame):
        if frame is None:
            return frame

        state = np.array(frame)

        # Make it channels x height x width
        state = state.transpose((2, 0, 1))

        # Scale
        state = state.astype("float32")

        # To torch
        return torch.from_numpy(state).unsqueeze(0).to(self.device)
