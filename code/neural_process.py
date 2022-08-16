import torch
from torch import nn
from torch.nn import Module as TorchModule
import torch.nn.functional as F
from modules import GeneralModel
from utils import ContextTargetSpliter
from torch.distributions import Normal
from layers import EdgePredictor, TimeEncode, SineActivation
from torchdiffeq import odeint_adjoint


class NeuralProcess(GeneralModel):
    def __init__(self, gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, np_parm,
                 base_model, enable_ode, enable_determinstic, resize_ratio,
                 combined=False):
        super(NeuralProcess, self).__init__(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param,
                                            train_param,
                                            combined=combined)
        self.base_model = base_model
        self.enable_ode = enable_ode
        self.enabe_determinstic = enable_determinstic
        self.r_dim = np_parm['r_dim']
        self.z_dim = np_parm['z_dim']
        self.h_dim = np_parm['h_dim']
        self.t_dim = np_parm['t_dim']
        self.np_out_dim = np_parm['out_dim']
        self.l = np_parm['l']
        self.old_as_context = np_parm['old_as_context']
        self.r_tol = float(np_parm['r_tol'])
        self.a_tol = float(np_parm['a_tol'])
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.method = np_parm['method']
        self.resize_ratio = resize_ratio
        self.encoder = Encoder(
            gnn_param['dim_out'] * 2, np_parm['y_dim'], self.h_dim, self.r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(self.r_dim, self.z_dim)
        self.decoder = Decoder(
            gnn_param['dim_out'], self.z_dim,
            self.h_dim,
            self.np_out_dim)  # ODEDecoder(self.latent_odefunc, gnn_param['dim_out'], self.z_dim, self.h_dim, resize_ratio, tol=self.tol, method=self.method) if self.enable_ode else Decoder(gnn_param['dim_out'], self.z_dim, self.h_dim)
        self.time_encoder = TimeEncode(np_parm['t_dim'])
        if self.enable_ode:
            self.ode_solver = ODESolver(self.r_dim, self.h_dim, self.t_dim, self.time_encoder, r_tol=self.r_tol,
                                        a_tol=self.a_tol,
                                        method=self.method)
        if not self.old_as_context:
            self.context_spliter = ContextTargetSpliter(
                np_parm['context_split'])
        if self.base_model == 'snp':
            self.update_cell = nn.GRUCell(self.r_dim, self.r_dim)
        elif self.base_model == "anp":
            self.update_cell = nn.GRUCell(self.r_dim, self.r_dim)
            self.multi_atten = nn.MultiheadAttention(self.r_dim, num_heads=4, kdim=self.r_dim,
                                                     vdim=self.r_dim)
            self.history_memory = torch.zeros(
                1, self.r_dim, device=self.device)
        elif self.base_model == "mnp":
            self.memory_net = MemoryNet(np_parm['mem_size'], self.h_dim, self.t_dim, self.r_dim, self.time_encoder,
                                        device=self.device)
        if self.enabe_determinstic:
            self.deterministic_decoder = DeterminsticDecoder(gnn_param['dim_out'], self.r_dim, self.h_dim,
                                                             self.np_out_dim)
        self.register_buffer('running_r', torch.zeros(self.r_dim, device=self.device))
        self.register_buffer('num_batches_tracked', torch.tensor(
            0, dtype=torch.long, device=self.device))
        self.register_buffer('last_ts', torch.tensor(
            0, dtype=torch.float, device=self.device))
        self.test = False

    def detach(self):
        # Detach running_r for memory optimization
        self.running_r = self.running_r.detach()
        if self.base_model == "anp":
            self.history_memory.detach_()
        elif self.base_model == 'mnp':
            self.memory_net.memory.detach_()

    def reset(self):
        '''
        Reset running_r and num_batches_tracked
        :return:
        '''
        self.running_r = torch.zeros(self.r_dim, device=self.device)
        self.num_batches_tracked = torch.tensor(
            0, dtype=torch.long, device=self.device)
        self.last_ts = torch.tensor(0, dtype=torch.float, device=self.device)
        if self.base_model == 'mnp':
            self.memory_net.reset()
        elif self.base_model == 'anp':
            self.history_memory = torch.zeros(
                1, self.r_dim, device=self.device)

    def xy_to_mu_sigma(self, data, ts, mode, negative_sample=1):
        src, pos_dst, neg_dst = data
        ts_src, ts_pos_dst, ts_neg_dst = ts
        t = torch.cat([ts_src, ts_neg_dst], dim=-1)
        pos_pair = torch.cat([src, pos_dst], dim=-1)
        neg_pair = torch.cat([src.tile(negative_sample, 1), neg_dst], dim=-1)
        x = torch.cat([pos_pair, neg_pair], dim=0)
        y = torch.cat(
            [torch.ones(pos_pair.shape[0], device=self.device, dtype=torch.long),
             torch.zeros(neg_pair.shape[0], device=self.device, dtype=torch.long)],
            dim=0)
        r_i = self.encoder(x, y)
        r = self.aggregate(r_i, t, mode)
        if len(r.shape) < 2:
            r = r.unsqueeze(0)
        return self.r_to_mu_sigma(r)

    def forward(self, mfgs, ts, neg_samples=1, data=None):
        h = self.get_emb(mfgs)
        if self.base_model == "origin":
            if self.training:
                pos_pred, neg_pred = self.edge_predictor(
                    h, neg_samples=neg_samples)
                return pos_pred, neg_pred, None, None
            else:
                return self.edge_predictor(h, neg_samples=neg_samples)
        else:
            ts = torch.from_numpy(ts).to(self.device)
            ts = ts / self.resize_ratio
            num_edge = h.shape[0] // (neg_samples + 2)
            h_src = h[:num_edge]
            ts_src = ts[:num_edge]
            h_pos_dst = h[num_edge:2 * num_edge]
            ts_pos_dst = ts[num_edge:2 * num_edge]
            h_neg_dst = h[2 * num_edge:]
            ts_neg_dst = ts[2 * num_edge:]
            if self.training:
                if self.old_as_context:
                    if self.enable_ode and torch.max(ts) > self.last_ts:
                        if self.base_model == 'mnp':
                            t_0 = self.last_ts.expand(ts_src.shape[0], 1)
                            init_r = self.memory_net(ts_src.unsqueeze(-1))
                            context_r = self.ode_solver(init_r, t_0,
                                                        ts_src.unsqueeze(-1))
                        else:
                            context_r = self.ode_solver(self.running_r.unsqueeze(0), self.last_ts.view(1, 1),
                                                        torch.max(ts).view(1, 1))
                    else:
                        if self.base_model == "mnp":
                            context_r = self.memory_net(ts_src.unsqueeze(-1))
                        else:
                            context_r = self.running_r.unsqueeze(0)
                    mu_context, sigma_context = self.r_to_mu_sigma(context_r)
                else:
                    context_data, target_data, context_ts, target_ts = self.context_spliter(
                        [h_src, h_pos_dst, h_neg_dst, ts_src, ts_pos_dst, ts_neg_dst])
                    mu_context, sigma_context = self.xy_to_mu_sigma(context_data, context_ts,
                                                                    'context', negative_sample=neg_samples)
                mu_target, sigma_target = self.xy_to_mu_sigma([h_src, h_pos_dst, h_neg_dst],
                                                              [ts_src, ts_pos_dst,
                                                               ts_neg_dst], 'target',
                                                              negative_sample=neg_samples)
                if self.base_model == 'mnp':
                    target_r = self.memory_net(ts_src.unsqueeze(-1))
                    mu_target, sigma_target = self.r_to_mu_sigma(target_r)
                if self.enabe_determinstic:
                    r = self.running_r.expand(h.shape[0], self.running_r.shape[0])
                    pos_pred, neg_pred, dist = self.deterministic_decoder(h, r, neg_samples=neg_samples, training=self.training)
                    return pos_pred, neg_pred, dist
                # Sample from encoded distribution using reparameterization trick
                q_context = Normal(mu_context, sigma_context)
                q_target = Normal(mu_target, sigma_target)
                z_sample = q_target.rsample(torch.Size([self.l]))  # (l, z_dim)
                target_pos_pred, target_neg_pred = self.decoder(
                    h, z_sample, ts, self.last_ts, neg_samples=neg_samples)
                return target_pos_pred, target_neg_pred, q_target, q_context
            else:
                if (data == "train" or data == 'val') and self.test:
                    # Update running_r using all the train and val data at the beginning of test phase
                    mu_target, sigma_target = self.xy_to_mu_sigma([h_src, h_pos_dst, h_neg_dst],
                                                                  [ts_src, ts_pos_dst,
                                                                   ts_neg_dst], 'target',
                                                                  negative_sample=neg_samples)
                    if self.base_model == 'mnp':
                        target_r = self.memory_net(ts_src.unsqueeze(-1))
                        mu_target, sigma_target = self.r_to_mu_sigma(target_r)
                    if self.enabe_determinstic:
                        r = self.running_r.expand(h.shape[0], self.running_r.shape[0])
                        pos_pred, neg_pred = self.deterministic_decoder(h, r, neg_samples=neg_samples)
                        return pos_pred, neg_pred
                else:
                    if self.enable_ode and torch.max(ts) > self.last_ts:
                        if self.base_model == 'mnp':
                            init_r = self.memory_net(ts_src.unsqueeze(-1))
                            context_r = self.ode_solver(init_r, self.last_ts.expand(ts_src.shape[0]).unsqueeze(-1),
                                                        ts_src.unsqueeze(-1))
                        else:
                            r = self.running_r.expand(
                                ts_src.shape[0], self.running_r.shape[0])
                            t_0 = self.last_ts.expand(ts_src.shape[0])
                            context_r = self.ode_solver(
                                r, t_0.unsqueeze(-1), ts_src.unsqueeze(-1))
                    else:
                        if self.base_model == "mnp":
                            context_r = self.memory_net(ts_src.unsqueeze(-1))
                        else:
                            context_r = self.running_r.unsqueeze(0)
                    if self.enabe_determinstic:
                        r = context_r.expand(h.shape[0], context_r.shape[-1])
                        pos_pred, neg_pred = self.deterministic_decoder(h, r, neg_samples=neg_samples)
                        return pos_pred, neg_pred
                    mu_target, sigma_target = self.r_to_mu_sigma(context_r)
                q_target = Normal(mu_target, sigma_target)
                z_sample = q_target.rsample(torch.Size([self.l]))
                target_pos_pred, target_neg_pred = self.decoder(
                    h, z_sample, ts, self.last_ts, neg_samples=neg_samples)
                return target_pos_pred, target_neg_pred

    def aggregate(self, r_i, ts, mode):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batchsize * 2, r_dim, )
        ts: torch.Tensor
            Shape (batchsize * 2 )
        mode: string
            context or target
        """
        t_emb = self.time_encoder(ts)
        r_i = r_i + t_emb
        current_r = torch.mean(r_i, dim=0)
        # Record the maximum timestamp for current batch
        current_t = torch.max(ts)
        # Only the target data will update the running_r and num_batches
        if self.base_model == "snp":
            r = self.update_cell(self.running_r, current_r)
            if mode == "target":
                self.running_r = r
                self.last_ts = current_t
        elif self.base_model == "np":
            if mode == "target":
                self.num_batches_tracked += 1
                momentum = 1 / float(self.num_batches_tracked)
                self.running_r = (1 - momentum) * \
                                 self.running_r + momentum * current_r
                self.last_ts = current_t
                r = self.running_r
            else:
                momentum = 1 / float(self.num_batches_tracked + 1)
                r = (1 - momentum) * self.running_r + momentum * current_r
        elif self.base_model == "anp":
            current_h = current_r.unsqueeze(0)
            memory = torch.cat([self.history_memory, current_h], dim=0)
            attn_out, _ = self.multi_atten(current_h, memory, memory, need_weights=False)
            r = torch.mean(attn_out, dim=0)
            # r = self.update_cell(current_r, r.squeeze(0))
            if mode == "target":
                self.running_r = r
                self.history_memory = memory  # torch.cat([self.history_memory, torch.cat((current_r.unsqueeze(0), t_emb), dim=-1)], dim=0)
                self.last_ts = current_t
        elif self.base_model == 'mnp':
            ts.unsqueeze_(-1)
            mem_out = self.memory_net.update(ts, r_i)
            r = self.memory_net(ts, mem_out)
            if mode == "target":
                # self.running_r = r
                self.memory_net.memory = mem_out
                self.last_ts = current_t
        return r


class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.y_emb = nn.Embedding(2, y_dim)
        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.LeakyReLU(),
                  nn.Linear(h_dim, h_dim),
                  nn.LeakyReLU(),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        y = self.y_emb(y)
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, out_dim):
        super(Decoder, self).__init__()
        self.decode_fc = nn.Sequential(nn.Linear(x_dim + z_dim, h_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(h_dim, out_dim))
        self.edge_predictor = EdgePredictor(out_dim)

    def forward(self, x, z, ts=None, t_0=None, neg_samples=1):
        '''

        :param x: (batch_size * 2 + batch_size * neg_smp, x_dim)
        :param z: (l, batch_size + B * neg_smp, z_dim) or (l, 1, z_dim)
        :return:
        '''
        z = z.transpose(0, 1)
        if z.shape[0] == 1:
            z = z.expand(x.shape[0], z.shape[1], z.shape[2])
        else:
            z = z.tile(neg_samples + 2, 1, 1)
        x = x.unsqueeze(1).expand(x.shape[0], z.shape[1], x.shape[1])
        x = torch.cat([x, z], dim=-1)  # 3B, L, x_dim + z_dim
        h = self.decode_fc(x)
        pos_pred, neg_pred = self.edge_predictor(h, neg_samples=neg_samples)
        return torch.mean(pos_pred, dim=1), torch.mean(neg_pred, dim=1)


class DeterminsticDecoder(nn.Module):
    '''CNP decoder'''

    def __init__(self, x_dim, r_dim, h_dim, out_dim):
        super(DeterminsticDecoder, self).__init__()
        self.decode_fc = nn.Sequential(nn.Linear(x_dim + r_dim, h_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(h_dim, out_dim))
        self.edge_predictor = EdgePredictor(out_dim, dim_out=2)

    def forward(self, x, r, ts=None, t_0=None, neg_samples=1, training=False):
        '''
        :param x: (batch_size * 2 + batch_size * neg_smp, x_dim)
        :param r: (batch_size * 2 + batch_size * neg_smp, r_dim)
        :return:
        '''
        x = torch.cat([x, r], dim=-1)  # batch_size * 2 + batch_size * neg_smp, x_dim + r_dim
        h = self.decode_fc(x)
        pos_pred, neg_pred = self.edge_predictor(h, neg_samples=neg_samples)  # Pos_size, 2, Neg_size, 2
        pos_mean, pos_var = pos_pred.split(1, dim=-1)
        neg_mean, neg_var = neg_pred.split(1, dim=-1)
        mu = torch.cat([pos_mean, neg_mean])
        sigma = torch.cat([pos_var, neg_var])
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        dist = Normal(mu, sigma)
        if training:
            return pos_mean, neg_mean, dist
        else:
            return pos_mean, neg_mean


class ODESolver(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, r_dim, h_dim, t_dim, time_enc, r_tol=1e-6, a_tol=1e-7, method="dopri5"):
        super(ODESolver, self).__init__()
        self.a_tol = a_tol
        self.r_tol = r_tol
        self.method = method
        self.ode_func = ODEFunc(r_dim, h_dim, t_dim,
                                time_enc, self.start_time, self.end_time)

    def forward(self, r, start_t, end_t):
        '''

        :param r: B * r_dim
        :param start_t: B,
        :param end_t: B,
        :return:
        '''
        initial_state = (r, end_t, start_t)
        tt = torch.tensor([self.start_time, self.end_time]).to(r)  # [0, 1]
        solution = odeint_adjoint(
            self.ode_func,
            initial_state,
            tt,
            rtol=self.r_tol,
            atol=self.a_tol,
            method=self.method,
            # options=None if self.method == "dopri5" else {'step_size': 0.125}
        )
        r_final, _, _ = solution
        r_final = r_final[-1, :]
        return r_final


class ODEDecoder(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, latent_odefunc, x_dim, z_dim, h_dim, resize_ratio, tol=1e-6, method="dopri5", ):
        super(ODEDecoder, self).__init__()
        self.tol = tol
        self.method = method

        self.resize_ratio = resize_ratio
        self.ode_func = ODEFunc(z_dim, h_dim, self.start_time, self.end_time)
        self.decode_fc = nn.Linear(x_dim + z_dim, h_dim)
        self.edge_predictor = EdgePredictor(h_dim)

    def forward(self, x, z, ts, t_0=None):
        '''

        :param x: B * 3, x_dim
        :param z: l, z_dim
        :param ts: B * 3,
        :param init_t:
        :return:
        '''
        z = z.expand(x.shape[0], z.shape[0], z.shape[1])  # B * 3, l, z_dim
        x = x.unsqueeze(1).expand(
            x.shape[0], z.shape[1], x.shape[1])  # B * 3, l, z_dim
        ts = ts.unsqueeze(-1)  # B * 3, 1
        if t_0 is None:
            t_0 = torch.zeros_like(ts).to(ts)
        t_0 = t_0 / float(self.resize_ratio)
        ts = ts / float(self.resize_ratio)  # Resize the timestamp
        initial_state = (z, ts, t_0)
        tt = torch.tensor([self.start_time, self.end_time]).to(z)
        solution = odeint_adjoint(
            self.ode_func,
            initial_state,
            tt,
            method=self.method
        )

        z_final, _, _ = solution
        z_final = z_final[-1, :]
        x = torch.cat([x, z_final], dim=-1)  # 3B * L, x_dim + z_dim
        b, l, dim = x.shape
        x = x.view(b * l, -1)
        h = self.decode_fc(x)
        pos_pred, neg_pred = self.edge_predictor(h)
        pos_pred = pos_pred.view(b // 3, l, -1)
        neg_pred = neg_pred.view(b // 3, l, -1)
        return torch.mean(pos_pred, dim=1), torch.mean(neg_pred, dim=1)

    def train_decoder(self, x, z):
        '''
        :param x: (batch_size * 3, x_dim)
        :param z: (l, z_dim)
        :return:
        '''
        z = z.expand(x.shape[0], z.shape[0], z.shape[1])
        x = x.unsqueeze(1).expand(x.shape[0], z.shape[1], x.shape[1])
        x = torch.cat([x, z], dim=-1)  # 3B * L, x_dim + z_dim
        b, l, dim = x.shape
        x = x.view(b * l, -1)
        h = self.decode_fc(x)
        pos_pred, neg_pred = self.edge_predictor(h)
        pos_pred = pos_pred.view(b // 3, l, -1)
        neg_pred = neg_pred.view(b // 3, l, -1)
        return torch.mean(pos_pred, dim=1), torch.mean(neg_pred, dim=1)


class ODEFunc(nn.Module):
    def __init__(self, z_dim, h_dim, t_dim, time_enc, start_time, end_time):
        super(ODEFunc, self).__init__()
        self.start_time = start_time
        self.end_time = end_time
        # Timestamp's dimension =1
        self.time_enc = time_enc
        ode_layers = [nn.Linear(z_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, z_dim)]
        self.latent_odefunc = nn.Sequential(*ode_layers)

    def forward(self, s, x):
        '''
        Using dummy variable s to integrate between 0-1
        :param s:
        :param x:
        :return:
        '''
        z, ts, t_0 = x
        ratio = (ts - t_0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t_0
        with torch.enable_grad():
            z = z.requires_grad_(True)
            t = t.requires_grad_(True)
            t = self.time_enc(t)
            # inp = torch.cat([z, t], dim=-1)
            dz = self.latent_odefunc(z + t)
            dz = dz * ratio
        return dz, ts, t_0


class EncODEFunc(nn.Module):
    def __init__(self, latent_odefunc):
        super(EncODEFunc, self).__init__()
        self.latent_odefunc = latent_odefunc

    def forward(self, t, x):
        t = t.view(1, 1)
        return self.latent_odefunc(torch.cat([x, t], dim=-1))


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class MemoryNet(nn.Module):
    def __init__(self, mem_size, h, ts_dim, inp_dim, time_enc, device):
        super(MemoryNet, self).__init__()
        self.mem_size = mem_size
        self.h = h
        self.inp_dim = inp_dim
        self.time_encoder = time_enc
        self.device = device
        self.memory = torch.zeros(mem_size, h, device=self.device)
        self.Q_linear = nn.Linear(ts_dim, h, bias=False)
        self.K_linear = nn.Linear(h, h, bias=False)
        self.V_linear = nn.Linear(inp_dim, h, bias=False)
        self.gru = nn.GRUCell(h, h)
        self.feat_droupout = nn.Dropout2d(0.2)
        self.reset()

    def update(self, ts, input):
        '''
        Update memory
        :param
            ts: B
            input: B, inp_dim.
        :return: new_mem: mem_size, h
        '''
        t_emb = self.time_encoder(ts)
        q_h = self.Q_linear(t_emb)  # (B, h)
        k_h = self.K_linear(self.memory)  # (mem_size, h)
        weight = torch.softmax(torch.matmul(q_h, k_h.T),
                               dim=1)  # (B, mem_size)
        v_h = self.V_linear(input)  # (B, h)
        # v_h = self.feat_droupout(v_h.unsqueeze(0)).squeeze(0)
        new_mem = (weight.unsqueeze(2) * v_h.unsqueeze(1)).sum(0)
        mem = self.memory + new_mem
        # mem = self.gru(self.memory, new_mem)
        return F.normalize(mem, p=2, dim=1)

    def forward(self, ts, mem=None):
        '''
        Read Memory
        :param ts: B, t_dim
        :param mem: mem_size, h
        :return read_r: B, h

        '''
        if mem is None:
            mem = self.memory
        t_emb = self.time_encoder(ts)
        q_h = self.Q_linear(t_emb)  # (B, h)
        k_h = self.K_linear(mem)  # (mem_size, h)
        weight = torch.softmax(torch.matmul(q_h, k_h.T) / (k_h.shape[-1] ** (1 / 2)),
                               dim=1)  # (B, mem_size)
        read_r = torch.matmul(weight, mem)
        return read_r

    def reset(self):
        # Uniform Initialization of 1e-6
        self.memory = torch.randn_like(self.memory, device=self.device)
