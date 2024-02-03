import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import pickle
from copy import deepcopy

from fcsrl.agent import BaseAgent
from fcsrl.data import Batch
from fcsrl.network import MLP, Encoder, EncodedActorDeter, EnsembleEncodedCritic
from fcsrl.utils import Config, to_tensor, GaussianNoise, to_numpy, _nstep_return, \
    MeanStdNormalizer, PIDLagrangianUpdater, DiscDist, soft_update, cosine_sim_loss


class TD3LagReprAgent(BaseAgent):
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

        s_dim, a_dim, z_dim = net_cfg.s_dim, net_cfg.a_dim, net_cfg.z_dim
        zsa_out_dim = z_dim
        self.n_buckets = net_cfg.discrete_n_buckets
        self.bucket_low, self.bucket_high = net_cfg.discrete_range

        self.encoder = Encoder(
            s_dim, a_dim, z_dim, zsa_out_dim, net_cfg.encoder_hidden_dim,
        ).to(Config.DEVICE)
        self.feasi_head = MLP(z_dim, net_cfg.encoder_hidden_dim, self.n_buckets).to(Config.DEVICE)
        self.proj_layer = MLP(z_dim, net_cfg.encoder_hidden_dim, z_dim // 2).to(Config.DEVICE)
        self.post_proj = nn.Linear(z_dim // 2, z_dim // 2, bias=False).to(Config.DEVICE)
        
        encoder_para = sum([list(net.parameters()) for net in [
            self.encoder, self.feasi_head, self.proj_layer, self.post_proj]
        ], [])
        
        self.actor = EncodedActorDeter(
            s_dim, a_dim, z_dim, net_cfg.actor_hidden_dim,
        ).to(Config.DEVICE)

        self.critic = EnsembleEncodedCritic(
            2, s_dim, a_dim, z_dim, net_cfg.r_critic_hidden_dim,
        ).to(Config.DEVICE)

        self.cost_critic = EnsembleEncodedCritic(
            2, s_dim, a_dim, z_dim, net_cfg.c_critic_hidden_dim,
        ).to(Config.DEVICE)

        encoder_para = self.encoder.parameters()

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
    

    def train(self, mode=True):
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.cost_critic.train(mode)
        self.encoder.train(mode)

    # def save_model(self, model_path):
    #     torch.save({
    #         'actor': self.actor.state_dict(),
    #         'critic': self.critic.state_dict(),
    #         'cost_critic': self.cost_critic.state_dict(),
    #         'encoder': self.encoder.state_dict(),
    #         'f_head': self.feasi_head.state_dict(),
    #     }, f'{model_path}/model.pt')

    #     with open(f'{model_path}/normalizer.stats', 'wb') as f:
    #         pickle.dump(self.obs_normalizer.state_dict(), f)
        

    # def load_model(self, model_path):
    #     models = torch.load(f'{model_path}/model.pt')
    #     self.actor.load_state_dict(models['actor'])
    #     self.critic.load_state_dict(models['critic'])
    #     self.cost_critic.load_state_dict(models['cost_critic'])
    #     self.encoder.load_state_dict(models['encoder'])
    #     self.feasi_head.load_state_dict(models['f_head'])

    #     self.fixed_encoder.load_state_dict(self.encoder.state_dict())

    #     with open(f'{model_path}/normalizer.stats', 'rb') as f:
    #         self.obs_normalizer.load_state_dict(pickle.load(f))

    def forward(self, batch, states=None, add_noise=True):
        model = self.actor
        obs = batch.obs

        zs = self.fixed_encoder.zs(obs)
        logits = model(obs, zs)
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
        

    def process_fn(self, batch, replay, indices=None):
        batch.obs = self.obs_normalizer(batch.obs)
        batch.obs_next = self.obs_normalizer(batch.obs_next)
        
        batch.obs = to_tensor(batch.obs)
        batch.act = to_tensor(batch.act)
        batch.obs_next = to_tensor(batch.obs_next)
        batch.rew = to_tensor(batch.rew).unsqueeze(-1)
        batch.cost = to_tensor(batch.cost).unsqueeze(-1)
        batch.terminate = to_tensor(batch.terminate).unsqueeze(-1)
        
        B, N, H = len(indices), self._nstep_return, self._unroll_length
        indices_NxB = [indices]
        for _ in range(N - 1):
            indices_NxB.append((indices_NxB[-1]+1) % len(replay))
        indices_NxB = np.stack(indices_NxB, axis=0) # [N, B]
        indices_NxB = indices_NxB.reshape((-1,))
        
        batch_NxB = replay.get_batch(indices_NxB)
        batch_NxB.obs = self.obs_normalizer(batch_NxB.obs)
        batch_NxB.obs_next = self.obs_normalizer(batch_NxB.obs_next)
        batch_NxB.rew = batch_NxB.rew.reshape(N,B,1)
        batch_NxB.cost = batch_NxB.cost.reshape(N,B,1)
        batch_NxB.terminate = batch_NxB.terminate.reshape(N,B,1)
        batch_NxB.trunc = batch_NxB.trunc.reshape(N,B,1)
        # mask=0.0 after `trunc`=True or `terminate`=True
        end_NxB = np.logical_or(batch_NxB.terminate, batch_NxB.trunc)
        mask_ = 1.0 - (np.cumsum(end_NxB,0) > 0.5).astype(float)
        mask = np.concatenate([np.ones((1,B,1)), mask_[:-1]], 0) # (N,B,1)

        return_r_NxB, return_c_NxB = self.compute_TD_lambda_return(batch_NxB)
        feasi_NxB = self.compute_nstep_feasibility(batch_NxB)

        # (B, 1)
        batch.return_r, batch.return_c = return_r_NxB[0], return_c_NxB[0]
        # (H, B, ...)
        batch.act_HxB = to_tensor(batch_NxB.act.reshape(N,B,-1)[:H])
        batch.cost_HxB = to_tensor(batch_NxB.cost[:H])
        batch.obs_next_HxB = to_tensor(batch_NxB.obs_next.reshape(N,B,-1)[:H])
        batch.mask_HxB = to_tensor(mask[:H])
        batch.return_r_HxB = to_tensor(return_r_NxB[:H])
        batch.return_c_HxB = to_tensor(return_c_NxB[:H])
        batch.feasi_HxB = to_tensor(feasi_NxB[:H])

        return batch
            
    
    def compute_nstep_return(self, replay, indices, n_step):
        end_flag = np.logical_or(replay.terminate, replay.trunc)

        B = len(indices)
        indices = [indices]

        for _ in range(n_step - 1):
            indices.append(replay.next(indices[-1]))
        indices = np.stack(indices, axis=0) # [N, B]
        
        terminal_idx = indices[-1]
        terminal_batch = replay.get_batch(terminal_idx) # sampled from original replay buffer, needs normalizing
        terminal_batch.obs = self.obs_normalizer(terminal_batch.obs) 
        terminal_batch.obs_next = self.obs_normalizer(terminal_batch.obs_next)
        
        returns = []
        for value_type in ['rew', 'cost']:
            r_or_c = getattr(replay, value_type)
            next_q = self._next_q(terminal_batch, value_type)  # (B, 1)
            next_q = to_numpy(next_q) * (1 - terminal_batch.terminate.reshape(-1,1))
            
            n_step_return = _nstep_return(r_or_c, end_flag, next_q, indices, self._gamma, n_step)
            returns.append(n_step_return)
        return returns
    
    def compute_TD_lambda_return(self, batch_NxB):
        N, B = batch_NxB.rew.shape[0], batch_NxB.rew.shape[1]
        TD_lambda = 1.0 # TD(1) is equivalent to n_step return

        if len(batch_NxB.obs_next.shape) > 2:
            batch_NxB.obs_next = batch_NxB.obs_next.reshape(N*B, -1)
        
        returns = []
        for value_type in ["rew", "cost"]:
            value_next = self._next_q(batch_NxB, value_type).reshape(N,B,1)
            value_next = to_numpy(value_next)
            r_or_c = getattr(batch_NxB, value_type)
            ret = [value_next[-1]]
            for n in reversed(range(N)):
                ret_next = (1-batch_NxB.trunc[n]) * ret[-1] + batch_NxB.trunc[n] * value_next[n]
                ret.append(
                    r_or_c[n] + self._gamma*(1-batch_NxB.terminate[n])*(
                        (1-TD_lambda)*value_next[n]+TD_lambda*ret_next
                    ) # [B,1]
                )
            ret = np.stack(list(reversed(ret[1:])), 0) # (N,B,1)
            returns.append(ret)
        return returns # List[(N,B,1), (N,B,1)]
    
    def compute_nstep_feasibility(self, batch_NxB):
        N, B = batch_NxB.rew.shape[0], batch_NxB.rew.shape[1]
        discount = 0.9

        with torch.no_grad():
            zs_next = self.fixed_encoder.zs(batch_NxB.obs_next) # (N*B, ...)
            feasi_next_logits = self.feasi_head(zs_next).reshape(N,B,-1)
            feasi_next = DiscDist(feasi_next_logits, self.bucket_low, self.bucket_high, self.n_buckets).mean()
            feasi_next = to_numpy(feasi_next)[1:] # (N-1, B, 1)

        c = batch_NxB.cost
        feasi_score = [feasi_next[-1]]
        for n in reversed(range(N-1)):
            feasi_score_next = (1-batch_NxB.trunc[n]) * feasi_score[-1] + batch_NxB.trunc[n] * feasi_next[n]
            feasi_score.append(
                np.maximum(c[n], discount * (1-batch_NxB.terminate[n])*feasi_score_next ) # [B,1]
            )
        feasi_score = np.stack(list(reversed(feasi_score)), 0) # (N,B,1)
        return feasi_score
    

    def _next_q(self, batch, value_type='rew'):
        with torch.no_grad():
            zs_next = self.fixed_encoder.zs(batch.obs_next)
            a_next = self.actor_old(batch.obs_next, zs_next)
            a_noise = torch.randn(size=a_next.shape, device=a_next.device) * self._policy_noise
            if self._noise_clip > 0.0:
                a_noise = a_noise.clamp(-self._noise_clip, self._noise_clip)
            a_next = (a_next + a_noise).clamp(-1.0, 1.0)
            
            zsa_next = self.fixed_encoder.zsa(zs_next, a_next)
            if value_type == 'rew':
                next_q, _ = self.critic_old(batch.obs_next, zs_next, a_next, zsa_next).min(dim=1, keepdim=True)
            elif value_type == 'cost':
                next_q, _ = self.cost_critic_old(batch.obs_next, zs_next, a_next, zsa_next).max(dim=1, keepdim=True)
            
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
        dyn_coef = 0.5
        feasi_coef = 1.0

        H, B = self._unroll_length, batch.act_HxB.shape[1]
        pred_zs_1toH = [self.encoder.zs(batch.obs)]
        for h in range(H):
            pred_zs_1toH.append(self.encoder.zsa(pred_zs_1toH[-1], batch.act_HxB[h]))
        pred_zs_1toH = torch.stack(pred_zs_1toH[1:], 0) # (H, B, ...)
        
        # dynamics consistency
        pred_zs_1toH_flatten = pred_zs_1toH.flatten(0,1)
        pred_hz_1toH = self.post_proj(self.proj_layer(pred_zs_1toH_flatten)).reshape(H,B,-1)
        with torch.no_grad():
            obs_next_HxB = batch.obs_next_HxB.flatten(0,1)
            target_zs = self.fixed_encoder.zs(obs_next_HxB)
            target_hz_1toH = self.proj_layer(target_zs).reshape(H,B,-1).detach()
        dynamics_loss = - cosine_sim_loss(pred_hz_1toH, target_hz_1toH) * batch.mask_HxB
        dynamics_loss = dynamics_loss.mean()

        # predict per-step feasibility
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
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, batch.act)
        current_q_r = self.critic(batch.obs, fixed_zs, batch.act, fixed_zsa)
        current_q_c = self.cost_critic(batch.obs, fixed_zs, batch.act, fixed_zsa)

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
            act = self.actor(batch.obs, fixed_zs) * self._act_scale + self._act_bias
            zsa = self.fixed_encoder.zsa(fixed_zs, act)
            current_q_r = self.critic(batch.obs, fixed_zs, act, zsa).mean()
            current_q_c = self.cost_critic(batch.obs, fixed_zs, act, zsa).mean()

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