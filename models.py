import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution

_str_to_activation = {
    'relu': nn.ReLU(),
    'elu' : nn.ELU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class RSSM(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size, hidden_size, obs_embed_size,
                 activation, discrete=False, discrete_classes=32):

        super().__init__()

        self.action_size = action_size
        self.stoch_size  = stoch_size   
        self.deter_size  = deter_size   # GRU hidden units
        self.hidden_size = hidden_size  # intermediate fc_layers hidden units 
        self.embedding_size = obs_embed_size
        self.discrete = discrete
        self.discrete_classes = discrete_classes
        self.stoch_feature_size = (
            self.stoch_size * self.discrete_classes if self.discrete else self.stoch_size
        )

        self.act_fn = _str_to_activation[activation]
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)

        self.fc_state_action = nn.Linear(self.stoch_feature_size + self.action_size, self.deter_size)
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior = nn.Linear(self.hidden_size, self._dist_output_size)
        self.fc_embed_posterior = nn.Linear(self.embedding_size + self.deter_size, self.hidden_size)
        self.fc_state_posterior = nn.Linear(self.hidden_size, self._dist_output_size)

    @property
    def _dist_output_size(self):
        if self.discrete:
            # v2: output logits for each category
            return self.stoch_size * self.discrete_classes
        return 2 * self.stoch_size

    def init_state(self, batch_size, device):

        state = dict(
            stoch = torch.zeros(batch_size, self.stoch_size).to(device),
            deter = torch.zeros(batch_size, self.deter_size).to(device))
        if self.discrete:
            state['logits'] = torch.zeros(
                batch_size, self.stoch_size, self.discrete_classes).to(device)
            state['stoch'] = torch.zeros(batch_size, self.stoch_feature_size).to(device)
        else:
            state['mean'] = torch.zeros(batch_size, self.stoch_size).to(device)
            state['std'] = torch.zeros(batch_size, self.stoch_size).to(device)
        return state

    def get_dist(self, state):

        if self.discrete:
            # v2
            distribution = distributions.OneHotCategoricalStraightThrough(logits=state['logits'])
            distribution = distributions.independent.Independent(distribution, 1)
            return distribution
        # v1
        distribution = distributions.Normal(state['mean'], state['std'])
        distribution = distributions.independent.Independent(distribution, 1)
        return distribution

    def get_feat(self, state):

        return torch.cat([state['stoch'], state['deter']], dim=-1)

    def _dist_params(self, dist_input):

        if self.discrete:
            logits = torch.reshape(
                dist_input, (*dist_input.shape[:-1], self.stoch_size, self.discrete_classes))
            dist = self.get_dist({'logits': logits})
            sample = dist.rsample()
            sample = torch.reshape(sample, (*sample.shape[:-2], self.stoch_feature_size))
            return {'logits': logits, 'stoch': sample}

        mean, std = torch.chunk(dist_input, 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std
        return {'mean': mean, 'std': std, 'stoch': sample}

    def observe_step(self, prev_state, prev_action, obs_embed, nonterm=1.0):

        prior = self.imagine_step(prev_state, prev_action, nonterm)
        posterior_embed = self.act_fn(self.fc_embed_posterior(torch.cat([obs_embed, prior['deter']], dim=-1)))
        posterior = self._dist_params(self.fc_state_posterior(posterior_embed))
        # Observation updates stochastic state; deterministic RSSM state is shared.
        posterior['deter'] = prior['deter']
        return prior, posterior

    def imagine_step(self, prev_state, prev_action, nonterm=1.0):

        state_action = self.act_fn(self.fc_state_action(torch.cat([prev_state['stoch']*nonterm, prev_action], dim=-1)))
        deter = self.rnn(state_action, prev_state['deter']*nonterm)
        prior_embed = self.act_fn(self.fc_embed_prior(deter))
        prior = self._dist_params(self.fc_state_prior(prior_embed))
        prior['deter'] = deter
        return prior

    def observe_rollout(self, obs_embed, actions, nonterms, prev_state, horizon):

        priors = []
        posteriors = []

        for t in range(horizon):
            prev_action = actions[t]* nonterms[t]
            prior_state, posterior_state = self.observe_step(prev_state, prev_action, obs_embed[t], nonterms[t])
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        priors = self.stack_states(priors, dim=0)
        posteriors = self.stack_states(posteriors, dim=0)

        return priors, posteriors

    def imagine_rollout(self, actor, prev_state, horizon, return_actions=False, include_initial=False):

        rssm_state = prev_state
        next_states = [rssm_state] if include_initial else []
        actions = []

        if return_actions:
            feat = self.get_feat(rssm_state)
            actions.append(torch.zeros(*feat.shape[:-1], actor.action_size).to(feat.device))

        for t in range(horizon):
            action = actor(self.get_feat(rssm_state).detach())
            rssm_state = self.imagine_step(rssm_state, action)
            actions.append(action)
            next_states.append(rssm_state)

        next_states = self.stack_states(next_states)
        if return_actions:
            return next_states, torch.stack(actions)
        return next_states

    def stack_states(self, states, dim=0):

        return {
            key: torch.stack([state[key] for state in states], dim=dim)
            for key in states[0]
        }

    def detach_state(self, state):

        return {key: value.detach() for key, value in state.items()}

    def seq_to_batch(self, state):

        return {
            key: torch.reshape(value, (value.shape[0] * value.shape[1], *value.shape[2:]))
            for key, value in state.items()
        }


class ConvEncoder(nn.Module):

    def __init__(self, input_shape, embed_size, activation, depth=32):

        super().__init__()

        self.input_shape = input_shape
        self.act_fn = _str_to_activation[activation]
        self.depth = depth
        self.kernels = [4, 4, 4, 4]

        self.embed_size = embed_size
        
        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = input_shape[0] if i==0 else self.depth * (2 ** (i-1))
            out_ch = self.depth * (2 ** i)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=2))
            layers.append(self.act_fn)

        self.conv_block = nn.Sequential(*layers)
        self.fc = nn.Identity() if self.embed_size == 1024 else nn.Linear(1024, self.embed_size)

    def forward(self, inputs):
        reshaped = inputs.reshape(-1, *self.input_shape)
        embed = self.conv_block(reshaped)
        embed = torch.reshape(embed, (*inputs.shape[:-3], -1))
        embed = self.fc(embed)

        return embed

class ConvDecoder(nn.Module):
 
    def __init__(self, stoch_size, deter_size, output_shape, activation, depth=32):

        super().__init__()

        self.output_shape = output_shape
        self.depth = depth
        self.kernels = [5, 5, 6, 6]
        self.act_fn = _str_to_activation[activation]
        
        self.dense = nn.Linear(stoch_size + deter_size, 32*self.depth)

        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = 32*self.depth if i==0 else self.depth * (2 ** (len(self.kernels)-1-i))
            out_ch = output_shape[0] if i== len(self.kernels)-1 else self.depth * (2 ** (len(self.kernels)-2-i))
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2))
            if i!=len(self.kernels)-1:
                layers.append(self.act_fn)

        self.convtranspose = nn.Sequential(*layers)

    def forward(self, features):
        out_batch_shape = features.shape[:-1]
        out = self.dense(features)
        out = torch.reshape(out, [-1, 32*self.depth, 1, 1])
        out = self.convtranspose(out)
        mean = torch.reshape(out, (*out_batch_shape, *self.output_shape))

        out_dist = distributions.independent.Independent(
            distributions.Normal(mean, 1), len(self.output_shape))

        return out_dist

# used for reward and value models
class DenseDecoder(nn.Module):

    def __init__(self, stoch_size, deter_size, output_shape, n_layers, units, activation, dist):

        super().__init__()

        self.input_size = stoch_size + deter_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.dist = dist

        layers=[]

        for i in range(self.n_layers):
            in_ch = self.input_size if i==0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn) 

        layers.append(nn.Linear(self.units, int(np.prod(self.output_shape))))

        self.model = nn.Sequential(*layers)

    def forward(self, features):

        out = self.model(features)

        if self.dist == 'normal':
            return distributions.independent.Independent(
                distributions.Normal(out, 1), len(self.output_shape))
        if self.dist == 'binary':
            return distributions.independent.Independent(
                distributions.Bernoulli(logits =out), len(self.output_shape))
        if self.dist == 'none':
            return out

        raise NotImplementedError(self.dist)

class ActionDecoder(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size, n_layers, units, 
                        activation, min_std=1e-4, init_std=5, mean_scale=5):

        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.units = units  
        self.act_fn = _str_to_activation[activation]
        self.n_layers = n_layers

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        layers = []
        for i in range(self.n_layers):
            in_ch = self.stoch_size + self.deter_size if i==0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        layers.append(nn.Linear(self.units, 2*self.action_size))
        self.action_model = nn.Sequential(*layers)

    def get_dist(self, features):

        out = self.action_model(features)
        mean, std = torch.chunk(out, 2, dim=-1) 

        raw_init_std = np.log(np.exp(self._init_std)-1)
        action_mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
        action_std = F.softplus(std + raw_init_std) + self._min_std

        dist = distributions.Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = distributions.independent.Independent(dist, 1)
        dist = SampleDist(dist)
        return dist

    def forward(self, features, deter=False):

        dist = self.get_dist(features)

        if deter:
            return dist.mode()
        else:
            return dist.rsample()

    def add_exploration(self, action, action_noise=0.3):

        return torch.clamp(distributions.Normal(action, action_noise).rsample(), -1, 1)


class TanhBijector(distributions.Transform):

    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = constraints.real
        self.codomain = constraints.interval(-1.0, 1.0)

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))

class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample(self._samples)
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()
