import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from .q_learner import QLearner
import torch as th
from torch.optim import RMSprop


# import numpy as np
# import matplotlib.pyplot as plt


class LatentQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super(LatentQLearner, self).__init__(mac, scheme, logger, args)
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.role_save = 0
        self.role_save_interval = 10

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []

        self.mac.init_hidden(batch.batch_size)
        indicator, latent, latent_vae = self.mac.init_latent(batch.batch_size)

        reg_loss = 0
        dis_loss = 0
        ce_loss = 0
        for t in range(batch.max_seq_length):
            agent_outs, loss_, dis_loss_, ce_loss_ = self.mac.forward(batch, t=t, t_glob=t_env, train_mode=True)  # (bs,n,n_actions),(bs,n,latent_dim)
            reg_loss += loss_
            dis_loss += dis_loss_
            ce_loss += ce_loss_
            # loss_cs=self.args.gamma*loss_cs + _loss
            mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]
            # mac_out_latent.append((agent_outs_latent)) #[t,(bs,n,latent_dim)]

        reg_loss /= batch.max_seq_length
        dis_loss /= batch.max_seq_length
        ce_loss /= batch.max_seq_length

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # (bs,t,n,n_actions), Q values of n_actions

        # mac_out_latent=th.stack(mac_out_latent,dim=1)
        # (bs,t,n,latent_dim)
        # mac_out_latent=mac_out_latent.reshape(-1,self.args.latent_dim)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # (bs,t,n) Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)
        self.target_mac.init_latent(batch.batch_size)  # (bs,n,latent_size)

        for t in range(batch.max_seq_length):
            target_agent_outs, loss_cs_target, _, _ = self.target_mac.forward(batch,
                                                                        t=t)  # (bs,n,n_actions), (bs,n,latent_dim)
            target_mac_out.append(target_agent_outs)  # [t,(bs,n,n_actions)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index
        # (bs,t,n,n_actions)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # Q values

        # Max over target Q-Values
        if self.args.double_q:  # True for QMix
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()  # return a new Tensor, detached from the current graph
            mac_out_detach[avail_actions == 0] = -9999999
            # (bs,t,n,n_actions), discard t=0
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]  # indices instead of values
            # (bs,t,n,1)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            # (bs,t,n,n_actions) ==> (bs,t,n,1) ==> (bs,t,n) max target-Q
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            # (bs,t,1)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())  # no gradient through target net
        # (bs,t,1)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # entropy loss
        # mac_out_latent_norm=th.sqrt(th.sum(mac_out_latent*mac_out_latent,dim=1))
        # mac_out_latent=mac_out_latent/mac_out_latent_norm[:,None]
        # loss+=(th.norm(mac_out_latent)/mac_out_latent.size(0))*self.args.entropy_loss_weight
        loss += reg_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            #    if self.role_save % self.role_save_interval == 0:
            #        self.role_save = 0
            #        if self.args.latent_dim in [2, 3]:

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #           print(self.mac.agent.latent[:, :self.args.latent_dim],
            #                  self.mac.agent.latent[:, -self.args.latent_dim:])

            #    self.role_save += 1

            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_reg", reg_loss.item(), t_env)
            self.logger.log_stat("loss_dis", dis_loss.item(), t_env)
            self.logger.log_stat("loss_ce", ce_loss.item(), t_env)

            #indicator=[var_mean,mi.max(),mi.min(),mi.mean(),mi.std(),di.max(),di.min(),di.mean(),di.std()]
            self.logger.log_stat("var_mean", indicator[0].item(), t_env)
            self.logger.log_stat("mi_max", indicator[1].item(), t_env)
            self.logger.log_stat("mi_min", indicator[2].item(), t_env)
            self.logger.log_stat("mi_mean", indicator[3].item(), t_env)
            self.logger.log_stat("mi_std", indicator[4].item(), t_env)
            self.logger.log_stat("di_max", indicator[5].item(), t_env)
            self.logger.log_stat("di_min", indicator[6].item(), t_env)
            self.logger.log_stat("di_mean", indicator[7].item(), t_env)
            self.logger.log_stat("di_std", indicator[8].item(), t_env)

            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)

            if self.args.use_tensorboard:
                # log_vec(self,mat,metadata,label_img,global_step,tag)
                self.logger.log_vec(latent, list(range(self.args.n_agents)), t_env, "latent")
                self.logger.log_vec(latent_vae, list(range(self.args.n_agents)), t_env, "latent-VAE")
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
