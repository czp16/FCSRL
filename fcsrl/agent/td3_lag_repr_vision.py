import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import pickle
from copy import deepcopy
import time
# from kornia.augmentation import RandomCrop

from fcsrl.agent import BaseAgent
from fcsrl.data import Batch
from fcsrl.network import MLP, ConvEncoder, ActorDeter, EnsembleCritic
from fcsrl.utils import Config, to_tensor, GaussianNoise, to_numpy, _nstep_return, \
    MeanStdNormalizer, PIDLagrangianUpdater, DiscDist, cosine_sim_loss, soft_update

# an accelerated version to compute the N-step feasibility
def _nstep_feasbility(cost, end_flag, target_f, indices, discount, n_step):
    N = n_step
    feasi_score = [target_f]
    for n in reversed(range(N)):
        curr_idx = indices[n]
        feasi_score_next = (1 - end_flag[curr_idx]) * feasi_score[-1] + end_flag[curr_idx] * target_f
        feasi_score.append(
            np.maximum(cost[curr_idx], discount * feasi_score_next) # (B,1)
        )
    feasi_score = np.stack(list(reversed(feasi_score[1:])), 0) # (N,B)
    return feasi_score
    
def center_crop_image(img, input_H, output_H):
    # img: (B,H,W,C)
    top = (input_H - output_H) // 2
    left = (input_H - output_H) // 2

    cropped_img = img[:, top:top+output_H, left:left+output_H, :]
    return cropped_img

class TD3LagReprVisionAgent(BaseAgent):

    def __init__(
        self, 
        config,
        obs_normalizer=MeanStdNormalizer(),
    ):
        super().__init__()

        agent_cfg = config.agent
        self._tau = agent_cfg.soft_update_tau
        self._gamma = agent_cfg.discount_gamma
        self._noise = GaussianNoise(sigma=agent_cfg.explore_noise_std)
        # self._rew_norm = agent_cfg.rew_norm
        self._nstep_return = agent_cfg.nstep_return
        self._policy_noise = agent_cfg.policy_noise
        self._noise_clip = agent_cfg.noise_clip
        self._update_actor_freq = agent_cfg.update_actor_freq
        self._unroll_length = agent_cfg.unroll_length
        self._cnt = 0
        action_range = agent_cfg.action_range
        self._act_bias = (action_range[1] + action_range[0]) / 2
        self._act_scale = (action_range[1] - action_range[0]) / 2

        self.obs_normalizer = obs_normalizer
        self.transformations = []

        lagrg_cfg = config.Lagrangian
        self.cost_limit = lagrg_cfg.constraint_threshold
        self.lagrg = lagrg_cfg.init_lagrg
        self.lagrg_updater = PIDLagrangianUpdater(
            lagrg_cfg.init_lagrg,
            [lagrg_cfg.KP, lagrg_cfg.KI, lagrg_cfg.KD],
            lagrg_cfg.max_lambda,
        )
        self.build_nets(config)

    def build_nets(self, config):
        net_cfg = config.network

        self.repr_type = net_cfg.repr_type
        assert self.repr_type == "FCSRL"
        
        self.augment = False
        s_shape, a_dim, z_dim = net_cfg.s_shape, net_cfg.a_dim, net_cfg.z_dim
        self.n_buckets = net_cfg.discrete_n_buckets
        self.bucket_low, self.bucket_high = net_cfg.discrete_range

        # also predict reward and sum cost starting from the first state of subtrajectoy
        self.encoder = ConvEncoder(s_shape, a_dim, z_dim, net_cfg.encoder_hidden_dim).to(Config.DEVICE)
        self.feasi_head = MLP(z_dim, net_cfg.encoder_hidden_dim, self.n_buckets).to(Config.DEVICE)
        self.proj_layer = MLP(z_dim, net_cfg.encoder_hidden_dim, z_dim // 2).to(Config.DEVICE)
        self.post_proj = nn.Linear(z_dim // 2, z_dim // 2, bias=False).to(Config.DEVICE)
        
        encoder_para = sum([list(net.parameters()) for net in [
            self.encoder, self.feasi_head, self.proj_layer, self.post_proj]
        ], [])
        
        self.actor = ActorDeter(
            z_dim, a_dim, net_cfg.actor_hidden_dim,
        ).to(Config.DEVICE)
        self.critic = EnsembleCritic(
            2, z_dim, a_dim, net_cfg.r_critic_hidden_dim,
        ).to(Config.DEVICE)
        self.cost_critic = EnsembleCritic(
            2, z_dim, a_dim, net_cfg.c_critic_hidden_dim,
        ).to(Config.DEVICE)

        self.encoder_optim = torch.optim.Adam(encoder_para, lr=net_cfg.encoder_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=net_cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=net_cfg.critic_lr)
        self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(), lr=net_cfg.critic_lr)

        self.fixed_encoder = deepcopy(self.encoder)
        self.actor_old = deepcopy(self.actor)
        self.critic_old = deepcopy(self.critic)
        self.cost_critic_old = deepcopy(self.cost_critic)

        self.fixed_encoder.train(False)
        self.actor_old.train(False)
        self.critic_old.train(False)
        self.cost_critic_old.train(False)

    @torch.no_grad()
    def transform(self, img):
        if self.augment:
            processed_img = img.transpose(0,3,1,2) # (B,C,H,W)
            processed_img = to_tensor(processed_img)
            for tsf in self.transformations:
                processed_img = tsf(processed_img)
        else:
            processed_img = to_tensor(img.transpose(0,3,1,2))
        return processed_img
    

    def train(self, mode=True):
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.cost_critic.train(mode)
        self.encoder.train(mode)

    def forward(self, batch, states=None, add_noise=True):
        model = self.actor
        obs = batch.obs
        if self.augment:
            obs = center_crop_image(obs, self.original_img_H, self.augmented_img_H)

        zs = self.fixed_encoder.zs(obs, unpermuted=True)
        logits = model(zs)
        # add noise
        if add_noise and self.training:
            logits += to_tensor(self._noise.sample(list(logits.shape)))
        logits = logits.clamp(-1.0, 1.0)

        logits = self._act_scale * logits + self._act_bias
        return Batch(act=logits, zs=zs), None
    
    def sync_weights(self):
        soft_update(self.actor_old, self.actor, self._tau)
        soft_update(self.critic_old, self.critic, self._tau)
        soft_update(self.cost_critic_old, self.cost_critic, self._tau)
        soft_update(self.fixed_encoder, self.encoder, self._tau*0.2)
        soft_update(self.fixed_encoder, self.encoder, self._tau*0.2)
        

    def process_fn(self, batch, replay, indices=None):
        
        batch.obs = self.transform(batch.obs)
        batch.obs_next = self.transform(batch.obs_next)

        batch.act = to_tensor(batch.act)
        batch.rew = to_tensor(batch.rew).unsqueeze(-1)
        batch.cost = to_tensor(batch.cost).unsqueeze(-1)
        batch.terminate = to_tensor(batch.terminate).unsqueeze(-1)

        B, N, H = len(indices), self._nstep_return, self._unroll_length
        
        indices_HxB = [indices]
        for _ in range(H - 1):
            indices_HxB.append((indices_HxB[-1]+1) % len(replay))
        indices_HxB = np.stack(indices_HxB, axis=0) # [N, B]
        indices_HxB = indices_HxB.reshape((-1,))
        
        batch_HxB = replay.get_batch(indices_HxB)

        batch_HxB.obs = self.transform(batch_HxB.obs)
        batch_HxB.obs_next = self.transform(batch_HxB.obs_next)
        batch_HxB.rew = batch_HxB.rew.reshape(H,B,1)
        batch_HxB.cost = batch_HxB.cost.reshape(H,B,1)
        batch_HxB.terminate = batch_HxB.terminate.reshape(H,B,1)
        batch_HxB.trunc = batch_HxB.trunc.reshape(H,B,1)

        indices_nstep = [indices]
        for _ in range(N - 1):
            indices_nstep.append(replay.next(indices_nstep[-1])) 
            # `replay.next` means it will end at the end_flag
            # which means if terminate[idx] or trunc[idx], replay.next(idx) = idx
        indices_nstep = np.stack(indices_nstep, axis=0) # [N, B]
        terminal_idx = indices_nstep[-1]

        terminal_batch = replay.get_batch(terminal_idx) # sampled from original replay buffer, needs normalizing
        terminal_batch.obs = self.transform(terminal_batch.obs) 
        terminal_batch.obs_next = self.transform(terminal_batch.obs_next)

        batch.return_r, batch.return_c = self.compute_nstep_return(terminal_batch, replay, indices_nstep, N)
        
        feasi_NxB = self.compute_nstep_feasibility(terminal_batch, replay, indices_nstep, N)
        batch.feasi_HxB = to_tensor(feasi_NxB[:H])

        # mask=0.0 after `trunc`=True or `terminate`=True
        end_HxB = np.logical_or(batch_HxB.terminate, batch_HxB.trunc)
        mask_ = 1.0 - (np.cumsum(end_HxB,0) > 0.5).astype(float)
        mask = np.concatenate([np.ones((1,B,1)), mask_[:-1]], 0) # (H,B,1)

        # (H, B, ...)
        batch.act_HxB = to_tensor(batch_HxB.act.reshape(H,B,-1))
        batch.rew_HxB = to_tensor(batch_HxB.rew)
        batch.cost_HxB = to_tensor(batch_HxB.cost)
        batch.obs_next_HxB = to_tensor(batch_HxB.obs_next.reshape(H, B, *batch_HxB.obs_next.shape[1:]))
        batch.mask_HxB = to_tensor(mask)

        return batch
            
    
    def compute_nstep_return(self, terminal_batch, replay, indices_nstep, n_step):
        end_flag = np.logical_or(replay.terminate, replay.trunc)
        
        returns = []
        for value_type in ['rew', 'cost']:
            r_or_c = getattr(replay, value_type)
            next_q = self._next_q(terminal_batch, value_type)  # (B, 1)
            next_q = to_numpy(next_q) * (1 - terminal_batch.terminate.reshape(-1,1))
            
            n_step_return = _nstep_return(r_or_c, end_flag, next_q, indices_nstep, self._gamma, n_step)
            returns.append(n_step_return)
        return returns
    
    def compute_nstep_feasibility(self, terminal_batch, replay, indices_nstep, n_step):
        N, B = n_step, indices_nstep.shape[1]
        end_flag = np.logical_or(replay.terminate, replay.trunc)
        cost = replay.cost
        discount = 0.9

        with torch.no_grad():
            zs_next = self.fixed_encoder.zs(terminal_batch.obs_next) # (B, ...)
            feasi_next_logits = self.feasi_head(zs_next).unsqueeze(0) # (1,B,...)
            feasi_next = DiscDist(feasi_next_logits, self.bucket_low, self.bucket_high, self.n_buckets).mean() # (1,B,1)
            feasi_next = to_numpy(feasi_next)[0,:,0] * (1 - terminal_batch.terminate) # (B, )

        feasi_score = _nstep_feasbility(cost, end_flag, feasi_next, indices_nstep, discount, N) # (N,B)
        feasi_score = feasi_score.reshape(N,B,1)
        return feasi_score
    

    def _next_q(self, batch, value_type='rew'):
        with torch.no_grad():
            zs_next = self.fixed_encoder.zs(batch.obs_next)
            a_next = self.actor_old(zs_next)
            a_noise = torch.randn(size=a_next.shape, device=a_next.device) * self._policy_noise
            if self._noise_clip > 0.0:
                a_noise = a_noise.clamp(-self._noise_clip, self._noise_clip)
            a_next = (a_next + a_noise).clamp(-1.0, 1.0)
            
            if value_type == 'rew':
                next_q, _ = self.critic_old(zs_next, a_next).min(dim=1, keepdim=True)
            elif value_type == 'cost':
                next_q, _ = self.cost_critic_old(zs_next, a_next).max(dim=1, keepdim=True)
        return next_q

    def _target_q(self, batch, value_type='rew'):
        next_q = self._next_q(batch)
        r_or_c = getattr(batch, value_type)
        target_q = r_or_c + (1.0-batch.terminate) * self._gamma * next_q
        return target_q

    def learn(self, batch, batch_size=None):
        metrics = {}
        met = self.train_encoder(batch)
        metrics.update(met)
        met = self.train_actor_critic(batch)
        metrics.update(met)
        return metrics
    
    def train_encoder(self, batch):
        metrics = {}

        dyn_coef = 1.0
        feasi_coef = 1.0
        H, B = batch.act_HxB.shape[0], batch.act_HxB.shape[1]
        
        pred_zs_1toH = [self.encoder.zs(batch.obs)]
        for h in range(H):
            pred_zs_1toH.append(self.encoder.zsa(pred_zs_1toH[-1], batch.act_HxB[h]))
        pred_zs_1toH = torch.stack(pred_zs_1toH[1:], 0) # (H+1, B, ...)
        
        # dynamics consistency
        pred_zs_1toH_flatten = pred_zs_1toH.flatten(0,1)
        pred_hz_1toH = self.post_proj(self.proj_layer(pred_zs_1toH_flatten)).reshape(H,B,-1)
        with torch.no_grad():
            obs_next_HxB = batch.obs_next_HxB.flatten(0,1)
            target_zs = self.fixed_encoder.zs(obs_next_HxB)
            target_hz_1toH = self.proj_layer(target_zs).reshape(H,B,-1).detach()
        dynamics_loss = - cosine_sim_loss(pred_hz_1toH, target_hz_1toH) * batch.mask_HxB
        dynamics_loss = dynamics_loss.mean()

        # predict per-step feasbility
        feasi_pred_logits = self.feasi_head(pred_zs_1toH_flatten).reshape(H,B,-1)
        feasi_pred_dist = DiscDist(feasi_pred_logits, self.bucket_low, self.bucket_high, self.n_buckets)
        feasi_pred_mean = feasi_pred_dist.mean().detach()
        feasi_loss = - feasi_pred_dist.log_prob(batch.feasi_HxB) * batch.mask_HxB
        feasi_loss = feasi_loss.mean()

        encoder_loss = (
            dyn_coef * dynamics_loss 
            + feasi_coef * feasi_loss
        )

        metrics['loss/encoder_dynamics'] = dynamics_loss.item()
        metrics['loss/encoder_pred_feasi'] = feasi_loss.item()
        metrics['loss/encoder_pred_feasi_MSE'] = F.mse_loss(feasi_pred_mean, batch.feasi_HxB).item()
        
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        self.encoder_optim.step()

        metrics['loss/encoder'] = encoder_loss.item()
        return metrics
    
    def train_actor_critic(self, batch):
        metrics = {}
        weight = getattr(batch, "weight", 1.0)
        weight = to_tensor(weight)
        
        # 1. train actor
        if self._nstep_return == 1:
            target_q_r = self._target_q(batch, 'rew')
            target_q_c = self._target_q(batch, 'cost')
        else:
            target_q_r = to_tensor(batch.return_r)
            target_q_c = to_tensor(batch.return_c)
        
        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(batch.obs)
        current_q_r = self.critic(fixed_zs, batch.act)
        current_q_c = self.cost_critic(fixed_zs, batch.act)

        td_r = current_q_r - target_q_r
        critic_loss = (td_r.pow(2).sum(-1) * weight).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        metrics['loss/critic_r'] = critic_loss.item()
        
        td_c = current_q_c - target_q_c
        cost_critic_loss = (td_c.pow(2).sum(-1) * weight).mean()
        self.cost_critic_optim.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optim.step()
        metrics['loss/critic_c'] = cost_critic_loss.item()
        
        # 2. train actor
        if self._cnt % self._update_actor_freq == 0:
            act = self.actor(fixed_zs) * self._act_scale + self._act_bias
            current_q_r = self.critic(fixed_zs, act).mean()
            current_q_c = self.cost_critic(fixed_zs, act).mean()

            # self.lagrg = 0.0
            self.lagrg = self.lagrg_updater.get_lagrg()
            actor_loss = -(current_q_r - self.lagrg * current_q_c)
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            self._last_qr = current_q_r.item()
            self._last_qc = current_q_c.item()
            metrics['loss/actor_qr'] = self._last_qr
            metrics['loss/actor_qc'] = self._last_qc
            
            self.sync_weights()
        self._cnt += 1

        return metrics

    def update_lagrangian_multiplier(self, Jc):
        # update cost coef
        self.lagrg_updater.update(Jc, self.cost_limit)