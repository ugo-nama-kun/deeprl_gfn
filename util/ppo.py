import collections
import itertools
import os
from typing import List, Any, Optional

import pfrl
import torch
import torch.nn.functional as F
from pfrl.agent import AttributeSavingMixin
from torch import nn

from pfrl.agents import PPO
from pfrl.agents.ppo import _mean_or_nan, _elementwise_clip, _add_advantage_and_value_target_to_episodes, \
    _compute_explained_variance
from pfrl.utils.batch_states import batch_states


"""
PPO with KL
"""


class PPO_KL(PPO):
    """
    PPO with KL regularization
    """

    saved_attributes = ("model", "optimizer", "obs_normalizer")

    def __init__(
        self,
        model,
        optimizer,
        obs_normalizer=None,
        gpu=None,
        gamma=0.99,
        lambd=0.95,
        phi=lambda x: x,
        value_func_coef=1.0,
        entropy_coef=0.01,
        kl_coef=3.0,  # fixed kl coefficient. value adopted from https://arxiv.org/abs/2009.10897
        update_interval=2048,
        minibatch_size=64,
        epochs=10,
        clip_eps=0.2,
        clip_eps_vf=None,
        standardize_advantages=True,
        batch_states=batch_states,
        recurrent=False,
        max_recurrent_sequence_len=None,
        act_deterministically=False,
        max_grad_norm=None,
        value_stats_window=1000,
        entropy_stats_window=1000,
        value_loss_stats_window=100,
        policy_loss_stats_window=100,
        kl_divergence_stats_window=100,
    ):
        super(PPO_KL, self).__init__(
            model=model,
            optimizer=optimizer,
            obs_normalizer=obs_normalizer,
            gpu=gpu,
            gamma=gamma,
            lambd=lambd,
            phi=phi,
            value_func_coef=value_func_coef,
            entropy_coef=entropy_coef,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps=clip_eps,
            clip_eps_vf=clip_eps_vf,
            standardize_advantages=standardize_advantages,
            batch_states=batch_states,
            recurrent=recurrent,
            max_recurrent_sequence_len=max_recurrent_sequence_len,
            act_deterministically=act_deterministically,
            max_grad_norm=max_grad_norm,
            value_stats_window=value_stats_window,
            entropy_stats_window=entropy_stats_window,
            value_loss_stats_window=value_loss_stats_window,
            policy_loss_stats_window=policy_loss_stats_window
        )

        self.kl_coef = kl_coef
        self.kl_divergence_record = collections.deque(maxlen=kl_divergence_stats_window)

    def _lossfun(
        self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
    ):
        # additional KL term: Approximation from http://joschu.net/blog/kl-approx.html
        logr = log_probs - log_probs_old
        approx_kl = torch.mean((logr.exp() - 1) - logr)

        prob_ratio = torch.exp(log_probs - log_probs_old)

        loss_policy = -torch.mean(
            torch.min(
                prob_ratio * advs,
                torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs,
            ),
        )

        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred, vs_teacher)
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred, vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred, vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))
        self.kl_divergence_record.append(float(approx_kl))

        loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * loss_entropy
            + self.kl_coef * approx_kl
        )

        return loss

    def get_statistics(self):
        return [
            ("average_value", _mean_or_nan(self.value_record)),
            ("average_entropy", _mean_or_nan(self.entropy_record)),
            ("average_value_loss", _mean_or_nan(self.value_loss_record)),
            ("average_policy_loss", _mean_or_nan(self.policy_loss_record)),
            ("average_kl_loss", _mean_or_nan(self.kl_divergence_record)),
            ("n_updates", self.n_updates),
            ("explained_variance", self.explained_variance),
        ]

    def load(self, dirname: str, gpu_id=None) -> None:
        """Load internal states. gpu_id is int or None"""
        self.__load(dirname, [], gpu_id)

    def __load(self, dirname: str, ancestors: List[Any], gpu_id=None) -> None:
        if gpu_id is None:
            map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        else:
            # default
            map_location = torch.device("cpu") if not torch.cuda.is_available() else 'cuda:'+str(gpu_id)

        ancestors.append(self)
        for attr in self.saved_attributes:
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor for ancestor in ancestors
                ), "Avoid an infinite loop"
                attr_value.load(os.path.join(dirname, attr))
            else:
                if isinstance(
                    attr_value,
                    (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel),
                ):
                    attr_value = attr_value.module
                attr_value.load_state_dict(
                    torch.load(
                        os.path.join(dirname, "{}.pt".format(attr)), map_location
                    )
                )
        ancestors.pop()


"""
PPO with KL + Data Augmentation
"""


def vision_augmentation(state):
    aug_state = state
    return aug_state


def _add_log_prob_and_value_to_episodes_extended(
    episodes,
    model,
    phi,
    batch_states,
    obs_normalizer,
    device,
):

    dataset = list(itertools.chain.from_iterable(episodes))

    # Compute v_pred and next_v_pred
    states = batch_states([b["state"] for b in dataset], device, phi)
    next_states = batch_states([b["next_state"] for b in dataset], device, phi)

    # if obs_normalizer:
    #     states = obs_normalizer(states, update=False)
    #     next_states = obs_normalizer(next_states, update=False)

    with torch.no_grad(), pfrl.utils.evaluating(model):
        # prediction with data augmentation
        distribs, vs_pred = model(states, enable_aug=True)
        _, next_vs_pred = model(next_states, enable_aug=True)

        actions = torch.tensor([b["action"] for b in dataset], device=device)
        log_probs = distribs.log_prob(actions).cpu().numpy()
        vs_pred = vs_pred.cpu().numpy().ravel()
        next_vs_pred = next_vs_pred.cpu().numpy().ravel()

    for transition, log_prob, v_pred, next_v_pred in zip(
        dataset, log_probs, vs_pred, next_vs_pred
    ):
        transition["log_prob"] = log_prob
        transition["v_pred"] = v_pred
        transition["next_v_pred"] = next_v_pred


def _make_dataset(
    episodes, model, phi, batch_states, obs_normalizer, gamma, lambd, device
):
    """Make a list of transitions with necessary information."""

    _add_log_prob_and_value_to_episodes_extended(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        device=device,
    )

    _add_advantage_and_value_target_to_episodes(episodes, gamma=gamma, lambd=lambd)

    return list(itertools.chain.from_iterable(episodes))


class PPO_EXT(PPO_KL):
    """
    PPO, extended by the data-augmentation
    """

    saved_attributes = ("model", "optimizer", "obs_normalizer")

    def __init__(
        self,
        model,
        optimizer,
        obs_normalizer=None,
        gpu=None,
        gamma=0.99,
        lambd=0.95,
        phi=lambda x: x,
        value_func_coef=1.0,
        entropy_coef=0.01,
        kl_coef=3.0,  # fixed kl coefficient. value adopted from https://arxiv.org/abs/2009.10897
        update_interval=2048,
        minibatch_size=64,
        epochs=10,
        clip_eps=0.2,
        clip_eps_vf=None,
        standardize_advantages=True,
        batch_states=batch_states,
        recurrent=False,
        max_recurrent_sequence_len=None,
        act_deterministically=False,
        max_grad_norm=None,
        value_stats_window=1000,
        entropy_stats_window=1000,
        value_loss_stats_window=100,
        policy_loss_stats_window=100,
        kl_divergence_stats_window=100,
    ):
        super(PPO_EXT, self).__init__(
            model=model,
            optimizer=optimizer,
            obs_normalizer=obs_normalizer,
            gpu=gpu,
            gamma=gamma,
            lambd=lambd,
            phi=phi,
            value_func_coef=value_func_coef,
            entropy_coef=entropy_coef,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps=clip_eps,
            clip_eps_vf=clip_eps_vf,
            standardize_advantages=standardize_advantages,
            batch_states=batch_states,
            recurrent=recurrent,
            max_recurrent_sequence_len=max_recurrent_sequence_len,
            act_deterministically=act_deterministically,
            max_grad_norm=max_grad_norm,
            value_stats_window=value_stats_window,
            entropy_stats_window=entropy_stats_window,
            value_loss_stats_window=value_loss_stats_window,
            policy_loss_stats_window=policy_loss_stats_window
        )

        self.kl_coef = kl_coef
        self.kl_divergence_record = collections.deque(maxlen=kl_divergence_stats_window)

    def _update_if_dataset_is_ready(self):
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (
                0
                if self.batch_last_episode is None
                else sum(len(episode) for episode in self.batch_last_episode)
            )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                # TODO: Add Recurrent network training capability
                """
                dataset = _make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    device=self.device,
                )
                self._update_recurrent(dataset)
                """
                raise ValueError("PPO_EXT note: Recurrent mode is not supported.")
            else:
                dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = _compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory))
            )
            self.memory = []
