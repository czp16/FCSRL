import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from copy import deepcopy

from fcsrl.agent import BaseAgent
from fcsrl.data import Batch
from fcsrl.network import MLP, Encoder, EncodedCritic, EncodedActorProb
from fcsrl.utils import DeviceConfig, to_tensor, to_numpy, MeanStdNormalizer, PIDLagrangianUpdater, \
    DiscDist, soft_update, cosine_sim_loss


class PPOLagReprAgent(BaseAgent):

    def __init__(
        self, 
        config,
        obs_normalizer=MeanStdNormalizer(),
    ):
        super().__init__()

        agent_cfg = config.agent
        self._tau = agent_cfg.soft_update_tau
        self._gamma = agent_cfg.discount_gamma
        self._f_discount = agent_cfg.feasibility_discount
        self._gae_lambda = agent_cfg.gae_lambda
        self._max_grad_norm = agent_cfg.max_grad_norm
        self._clip_ratio_eps = agent_cfg.clip_ratio_eps
        self._dual_clip_ratio = agent_cfg.dual_clip_ratio
        self._entropy_coef = agent_cfg.entropy_coef
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
        ).to(DeviceConfig.DEVICE)
        self.feasi_head = MLP(z_dim, net_cfg.encoder_hidden_dim, self.n_buckets).to(DeviceConfig.DEVICE)
        self.proj_layer = MLP(z_dim, net_cfg.encoder_hidden_dim, z_dim // 2).to(DeviceConfig.DEVICE)
        self.post_proj = nn.Linear(z_dim // 2, z_dim // 2, bias=False).to(DeviceConfig.DEVICE)
        
        encoder_para = sum([list(net.parameters()) for net in [
            self.encoder, self.feasi_head, self.proj_layer, self.post_proj]
        ], [])
        
        self.actor = EncodedActorProb(
            s_dim, a_dim, z_dim, net_cfg.actor_hidden_dim,
        ).to(DeviceConfig.DEVICE)

        self.critic = EncodedCritic(
            s_dim, 0, z_dim, net_cfg.r_critic_hidden_dim,
        ).to(DeviceConfig.DEVICE)

        self.cost_critic = EncodedCritic(
            s_dim, 0, z_dim, net_cfg.c_critic_hidden_dim,
        ).to(DeviceConfig.DEVICE)

        self.encoder_optim = torch.optim.Adam(encoder_para, lr=net_cfg.encoder_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=net_cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=net_cfg.critic_lr)
        self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(), lr=net_cfg.critic_lr)

        self.fixed_encoder = deepcopy(self.encoder)
        self.actor_old = deepcopy(self.actor)

        self.fixed_encoder.train(False)
        self.fixed_encoder.train(False)
        self.actor_old.train(False)

    def train(self, mode=True):
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.cost_critic.train(mode)
    
    # def save_model(self, model_path):
    #     torch.save({
    #         'actor': self.actor.state_dict(),
    #         'critic': self.critic.state_dict(),
    #         'actor_optim': self.actor_optim.state_dict(),
    #         'critic_optim': self.critic_optim.state_dict(),
    #     }, f'{model_path}.model')

    #     with open(f'{model_path}.stats', 'wb') as f:
    #         pickle.dump(self.obs_normalizer.state_dict(), f)
        

    # def load_model(self, model_path):
    #     models = torch.load(f'{model_path}.model')
    #     self.actor.load_state_dict(models['actor'])
    #     self.critic.load_state_dict(models['critic'])
    #     self.actor_optim.load_state_dict(models['actor_optim'])
    #     self.critic_optim.load_state_dict(models['critic_optim'])

    #     self.actor_old = deepcopy(self.actor)
    #     self.actor_old.eval()

    #     with open(f'{model_path}.stats', 'rb') as f:
    #         self.obs_normalizer.load_state_dict(pickle.load(f))
    

    def process_fn(self, batch, replay, indices=None):
        batch.obs = self.obs_normalizer(batch.obs)
        batch.obs_next = self.obs_normalizer(batch.obs_next)
        
        return_r, return_c = self._get_returns(batch)

        feasi = self._get_feasibility(batch)
        batch.feasi_score = to_tensor(feasi).unsqueeze(-1)

        batch.return_r = to_tensor(return_r).unsqueeze(-1)
        batch.return_c = to_tensor(return_c).unsqueeze(-1)
        batch.obs = to_tensor(batch.obs)
        batch.act = to_tensor(batch.act)
        batch.obs_next = to_tensor(batch.obs_next)
        batch.rew = to_tensor(batch.rew).unsqueeze(-1)
        batch.cost = to_tensor(batch.cost).unsqueeze(-1)
        batch.terminate = to_tensor(batch.terminate).unsqueeze(-1)
        batch.trunc = to_tensor(batch.trunc).unsqueeze(-1)
        return batch
    

    def _get_returns(self, batch):
        # assert batch.trunc[-1] == 1.0, \
        #     'The trajectory does not end with truncate=True or terminate=True.'
        returns = []
        for value_type in ["rew", "cost"]:
            ret = getattr(batch, value_type).copy()
            for i in reversed(range(len(ret))):
                if batch.trunc[i] or i == len(ret) - 1:
                    zs_next = self.fixed_encoder.zs(batch.obs_next[i:i+1])
                    next_value = self._next_v(batch.obs_next[i:i+1], zs_next)
                    ret[i] += to_numpy(next_value.squeeze())
                else:
                    ret[i] += self._gamma * (1-batch.terminate[i]) * ret[i+1]
            returns.append(ret)
        return returns
    
    def _get_feasibility(self, batch):
        feasi = np.zeros_like(batch.cost)
        for i in reversed(range(len(feasi))):
            if batch.trunc[i] or i == len(feasi) - 1:
                zs_next = self.fixed_encoder.zs(batch.obs_next[i:i+1])
                feasi_next_logits = self.feasi_head(zs_next).unsqueeze(0)
                feasi_next = DiscDist(feasi_next_logits, self.bucket_low, self.bucket_high, self.n_buckets).mean()
                feasi_next = to_numpy(feasi_next.squeeze())
                feasi[i] = np.maximum(batch.cost[i], self._f_discount * (1-batch.terminate[i]) * feasi_next)
            else:
                feasi[i] = np.maximum(batch.cost[i], self._f_discount * (1-batch.terminate[i]) * feasi[i+1])
        return feasi
    
    def forward(self, batch, states=None):
        model = self.actor
        zs = self.fixed_encoder.zs(batch.obs)
        
        logits = model(batch.obs, zs)
        policy_dist = D.Independent(D.Normal(*logits), 1)
        
        act = policy_dist.sample()
        act = act.clamp(-1.0, 1.0)
        log_p_act = policy_dist.log_prob(act)
        
        act = self._act_scale * act + self._act_bias
        return Batch(act=act, dist=policy_dist, log_p_act=log_p_act), None

    def sync_weights(self):
        for o, n in zip(self.actor_old.parameters(),
                        self.actor.parameters()):
            o.data.copy_(n.data)
        
        soft_update(self.fixed_encoder, self.encoder, self._tau)
        
    def _next_v(self, obs_next, zs_next, value_type="rew"):
        model = self.critic if value_type == "rew" else self.cost_critic
        value = model(obs_next, zs_next)
        return value

    def learn(self, batch, batch_size=None, repeat=1):
        
        # 1&2. compute reward & cost Advantage
        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(batch.obs)
            fixed_zs_next = self.fixed_encoder.zs(batch.obs_next)
            
            vs = self.critic(batch.obs, fixed_zs)
            vs_next = self.critic(batch.obs_next, fixed_zs_next)
            c_vs = self.cost_critic(batch.obs, fixed_zs)
            c_vs_next = self.cost_critic(batch.obs_next, fixed_zs_next)

            if self._gae_lambda is None:
                adv = batch.return_r - vs
            else: # use GAE
                adv = batch.rew + self._gamma * (1-batch.terminate) * vs_next - vs
                for i in reversed(range(adv.shape[0])):
                    if not (batch.trunc[i, 0] or i == adv.shape[0] - 1):
                        adv[i] += self._gamma * self._gae_lambda * adv[i+1]
            
            adv_std = adv.std() + 1e-12
            adv = ((adv - adv.mean()) / adv_std).detach()
            batch.update(adv=adv)

            if self._gae_lambda is None: 
                c_adv = batch.return_c - c_vs
            else: # use GAE
                c_adv = batch.cost + self._gamma * (1-batch.terminate) * c_vs_next - c_vs
                for i in reversed(range(c_adv.shape[0])):
                    if not (batch.trunc[i, 0] or i == c_adv.shape[0] - 1):
                        c_adv[i] += self._gamma * self._gae_lambda * c_adv[i+1]
            
            c_adv = ((c_adv - c_adv.mean()) / adv_std).detach() # align the scale with rew adv
            batch.update(c_adv=c_adv)

        # 3. start training
        metrics = {}
        for _ in range(repeat):
            for b in batch.sampler(batch_size):
                
                met = self.train_encoder(b)
                for k,v in met.items():
                    if f"loss/{k}" not in metrics:
                        metrics[f"loss/{k}"] = []
                    metrics[f"loss/{k}"].append(v)

                met = self.train_actor_critic(b)
                for k,v in met.items():
                    if f"loss/{k}" not in metrics:
                        metrics[f"loss/{k}"] = []
                    metrics[f"loss/{k}"].append(v)
        self.sync_weights()
        
        return metrics
    
    def train_encoder(self, batch):
        metrics = {}
        dyn_coef = 0.5
        feasi_coef = 1.0

        # dynamics consistency
        with torch.no_grad():
            target_zs = self.fixed_encoder.zs(batch.obs_next)
            target_ps = self.proj_layer(target_zs)
        zs = self.encoder.zs(batch.obs)
        pred_zs = self.encoder.zsa(zs, batch.act)
        pred_ps = self.post_proj(self.proj_layer(pred_zs))
        dynamics_loss = - cosine_sim_loss(pred_ps, target_ps)
        dynamics_loss = dynamics_loss.mean()

        # predict feasibility
        feasi_pred_logits = self.feasi_head(pred_zs).unsqueeze(0)
        feasi_pred_dist = DiscDist(feasi_pred_logits, self.bucket_low, self.bucket_high, self.n_buckets)
        feasi_pred_mean = feasi_pred_dist.mean().detach().squeeze(0)
        feasi_loss = - feasi_pred_dist.log_prob(batch.feasi_score.unsqueeze(0))
        feasi_loss = feasi_loss.mean()

        encoder_loss = (
            dyn_coef * dynamics_loss 
            + feasi_coef * feasi_loss
        )
        
        metrics['encoder_dynamics'] = dynamics_loss.item()
        metrics['encoder_pred_feasi'] = feasi_loss.item()
        metrics['encoder_pred_feasi_MSE'] = F.mse_loss(feasi_pred_mean, batch.feasi_score).item()

        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        self.encoder_optim.step()
            
        metrics['encoder'] = encoder_loss.item()
        return metrics

    def train_actor_critic(self, batch):
        metrics = {}
        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(batch.obs)
        dist = D.Independent(D.Normal(*self.actor(batch.obs, fixed_zs)), 1)
        dist_old = D.Independent(D.Normal(*self.actor_old(batch.obs, fixed_zs)), 1)
        act_normed = (batch.act - self._act_bias) / self._act_scale
        ratio = torch.exp(dist.log_prob(act_normed) - dist_old.log_prob(act_normed)).unsqueeze(-1) # (B, 1)
        
        # update critic
        vf_loss = F.mse_loss(self.critic(batch.obs, fixed_zs), batch.return_r)
        self.critic_optim.zero_grad()
        vf_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self._max_grad_norm)
        self.critic_optim.step()
        metrics["r_critic"] = vf_loss.item()

        # update cost_critic
        c_vf_loss = F.mse_loss(self.cost_critic(batch.obs, fixed_zs), batch.return_c)
        self.cost_critic_optim.zero_grad()
        c_vf_loss.backward()
        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self._max_grad_norm)
        self.cost_critic_optim.step()
        metrics["c_critic"] = c_vf_loss.item()

        # update actor
        rew_surr1 = ratio * batch.adv
        rew_surr2 = ratio.clamp(1. - self._clip_ratio_eps, \
            1. + self._clip_ratio_eps) * batch.adv
        rew_clip1 = torch.min(rew_surr1, rew_surr2)
        rew_clip2 = torch.max(rew_clip1, self._dual_clip_ratio * batch.adv)
        rew_surr = torch.where(batch.adv < 0, rew_clip2, rew_clip1).mean()
        
        cost_surr1 = ratio * batch.c_adv
        cost_surr2 = ratio.clamp(1. - self._clip_ratio_eps, \
            1. + self._clip_ratio_eps) * batch.c_adv
        cost_clip1 = torch.max(cost_surr1, cost_surr2)
        cost_clip2 = torch.min(cost_clip1, self._dual_clip_ratio * batch.c_adv)
        cost_surr = torch.where(batch.c_adv > 0, cost_clip2, cost_clip1).mean()

        entropy_loss = dist.entropy().mean()
        KL_loss = D.kl.kl_divergence(dist, dist_old).mean()
        self.lagrg = self.lagrg_updater.get_lagrg()
        actor_loss = - (rew_surr - self.lagrg * cost_surr) - self._entropy_coef * entropy_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self._max_grad_norm)
        self.actor_optim.step()

        metrics.update({
            "kld": KL_loss.item(),
            "entropy": entropy_loss.item(),
            "r_adv": rew_surr.item(),
            "c_adv": cost_surr.item(),
        })

        return metrics



    def update_lagrangian_multiplier(self, Jc):
        # update cost coef
        self.lagrg_updater.update(Jc, self.cost_limit)
        