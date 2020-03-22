from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th


# multi-agent controller with separete parameters for each agent.
class SeparateMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(SeparateMAC, self).__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        # for SeparateMAC
        self.latents = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, t_glob=0, train_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)  # (bs*n,(obs+act+id))
        avail_actions = ep_batch["avail_actions"][:, t]
        # (bs*n,(obs+act+id)), (bs,n,hidden_size), (bs,n,latent_dim)
        agent_outs, self.hidden_states, loss_cs, diss_loss, ce_loss = self.agent.forward(agent_inputs, self.hidden_states, t=t,
                                                                                batch=ep_batch, t_glob=t_glob, train_mode=train_mode)
        # (bs*n,n_actions), (bs*n,hidden_dim), (bs*n,latent_dim)
        # self.latents=self.latents.reshape(ep_batch.batch_size,self.n_agents,self.args.latent_dim) #(bs,n,latent_dim)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":  # q for QMix. Ignored

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), loss_cs, diss_loss, ce_loss
        # (bs,n,n_actions), (bs,n,latent_dim)

    def init_hidden(self, batch_size):
        if self.args.use_cuda:
            self.hidden_states = th.zeros(batch_size, self.n_agents,
                                          self.args.rnn_hidden_dim).cuda()  # (bs,n,hidden_dim)
        else:
            self.hidden_states = th.zeros(batch_size, self.n_agents, self.args.rnn_hidden_dim)

    # for SeparateMAC
    def init_latent(self, batch_size):
        return self.agent.init_latent(batch_size)
        # self.latents = th.randn(self.n_agents, self.args.latent_dim,requires_grad=True).unsqueeze(0).expand(batch_size,self.n_agents,-1) #(bs,n,latent_dim)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        if self.args.obs_last_action:  # True for QMix
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))  # last actions are empty
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_agent_id:  # True for QMix
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))  # onehot agent ID

        # inputs[i]: (bs,n,n)
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)  # (bs*n, act+obs+id)
        # inputs[i]: (bs*n,n); ==> (bs*n,3n) i.e. (bs*n,(obs+act+id))
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
