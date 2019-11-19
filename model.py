import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, args, device):
        super(Net, self).__init__()
        self.time_step = args.time_step
        self.infer_model = args.infer_model
        self.x_dim = args.x_dim
        self.class_dim = args.class_dim
        self.z_dim = args.z_dim
        self.rnn_dim = args.rnn_dim
        self.rnn_layer = args.rnn_layer
        self.z2comb_dim = args.z2comb_dim
        self.hcomb_dim = args.hcomb_dim
        self.comb2z_dim = args.comb2z_dim
        self.emi_dim = args.emi_dim
        self.trans_dim = args.trans_dim
        self.disc_dim = args.disc_dim
        self.data_bn = nn.BatchNorm1d(args.x_dim//2)
        self.h0 = nn.Parameter(torch.zeros(2*args.rnn_layer, 1, self.rnn_dim, dtype=torch.float32))
        self.dropout_p = args.dropout_p
        self.gru_x2h = nn.GRU(
            input_size=self.x_dim,
            hidden_size=self.rnn_dim,
            num_layers=self.rnn_layer,
            dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )
        self.combiner = Combiner_zx2z(
            self.dropout_p, self.infer_model, self.time_step, self.z_dim, self.hcomb_dim, self.rnn_dim)
        self.emitter = Emitter_xz2x(
            self.dropout_p, self.x_dim, self.z_dim, self.emi_dim, device)
        self.trans = GatedTransition_z2z(
            self.dropout_p, self.time_step, self.z_dim, self.trans_dim)
        self.discriminator = Discriminator(
            self.dropout_p, self.rnn_dim, self.rnn_layer, self.z_dim, self.disc_dim, self.class_dim)

    def forward(self, x, discriminate, train):
        batch_size, dimensionality, time_step, joints, people = x.size()
        # x = x.permute(0, 2, 4, 3, 1).contiguous().view(batch_size, time_step, people * dimensionality * joints)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(batch_size * people, joints * dimensionality, time_step)
        x = self.data_bn(x)
        x = x.view(batch_size, people, joints, dimensionality, time_step)
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(batch_size, time_step, people * dimensionality * joints)

        # x -> h_left, h_right
        h_rnn, _ = self.gru_x2h(x, self.h0.repeat(1, batch_size, 1))

        # generative training
        if not discriminate:
            qz, mu_qz, logvar_qz = self.combiner(x, h_rnn, train)
            pz, mu_pz, logvar_pz = self.trans(qz, train)
            recon_x = self.emitter(qz[:, :-1, :], x)
            return recon_x, qz[:, :-1, :], mu_qz, logvar_qz, pz[:, :-1, :], mu_pz, logvar_pz
        # discriminative training
        else:
            qz, _, _ = self.combiner(x, h_rnn, train)
            y = self.discriminator(qz[:, :-1, :])
            return y


class Combiner_zx2z(nn.Module):
    def __init__(self, dropout_p, infer_model, time_step, z_dim, hcomb_dim, rnn_dim):
        super(Combiner_zx2z, self).__init__()
        self.infer_model = infer_model
        self.time_step = time_step
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim
        if self.infer_model[:2] == 'ST':
            self.fc_infer_z2comb = nn.Linear(z_dim, hcomb_dim)
            self.fc_infer_comb2z_mu = nn.Linear(hcomb_dim, z_dim)
            self.fc_infer_comb2z_logvar = nn.Linear(hcomb_dim, z_dim)
        elif self.infer_model[:2] == 'MF':
            self.fc_infer_mu_r = nn.Linear(rnn_dim, z_dim)
            self.fc_infer_logvar_r = nn.Linear(rnn_dim, z_dim)
            self.fc_infer_mu_l = nn.Linear(rnn_dim, z_dim)
            self.fc_infer_logvar_l = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)
        self.eps = 1e-8
        self.z_q_0 = nn.Parameter(torch.zeros(1, self.z_dim, dtype=torch.float32))

    def reparameterize(self, mu, logvar, train=True):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if train:
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, h_rnn, train):
        batch_size = x.size(0)

        # (batch_size, time_step, 2*hidden_size) -> (batch_size, time_step, 2, hidden_size)
        h_rnn = h_rnn.view(batch_size, self.time_step, 2, self.rnn_dim)
        h_left, h_right = h_rnn[:, :, 0, :], h_rnn[:, :, 1, :]

        if self.infer_model[:2] == 'ST':
            z = [None for i in range(self.time_step+1)]
            z_mu = [None for i in range(self.time_step)]
            z_logvar = [None for i in range(self.time_step)]
            z[-1] = self.z_q_0.repeat(batch_size, 1)

        if self.infer_model[:2] == 'ST':
            for t in range(self.time_step):
                # z(t-1) (+x) -> h_comb
                if self.infer_model == 'STLR':
                    h_comb = 1/3 * (self.dropout(self.tanh(self.fc_infer_z2comb(z[t-1]))
                           + h_left[:, t, :] + h_right[:, t, :]))
                elif self.infer_model == 'STR':
                    h_comb = 0.5 * (self.dropout(self.tanh(self.fc_infer_z2comb(z[t-1]))
                           + h_right[:, t, :]))
                elif self.infer_model == 'STL':
                    h_comb = 0.5 * (self.dropout(self.tanh(self.fc_infer_z2comb(z[t-1]))
                           + h_left[:, t, :]))
                # h_comb -> z(t)
                z_mu[t] = self.fc_infer_comb2z_mu(h_comb)
                z_logvar[t] = self.fc_infer_comb2z_logvar(h_comb)
                z[t] = self.reparameterize(z_mu[t], z_logvar[t], train)
            z = torch.stack(z, dim=1)
            z_mu = torch.stack(z_mu, dim=1)
            z_logvar = torch.stack(z_logvar, dim=1)
        elif self.infer_model == 'MFLR':
            # h_left, h_right -> z
            mu_r = self.fc_infer_mu_r(h_right)
            logvar_r = self.fc_infer_logvar_r(h_right)
            var_r = torch.exp(logvar_r)
            mu_l = self.fc_infer_mu_l(h_left)
            logvar_l = self.fc_infer_logvar_l(h_left)
            var_l = torch.exp(logvar_l)

            z_mu = (mu_r * var_l + mu_l * var_r) / (var_r + var_l + self.eps)
            z_logvar = logvar_r + logvar_l - torch.log(var_r + var_l + self.eps)
            z = self.reparameterize(z_mu, z_logvar, train)
            z = torch.cat((z, self.z_q_0.repeat(batch_size, 1, 1)), 1)
        return z, z_mu, z_logvar


class Emitter_xz2x(nn.Module):
    def __init__(self, dropout_p, x_dim, z_dim, emi_dim, device):
        super(Emitter_xz2x, self).__init__()
        self.x_dim = x_dim
        self.device = device
        self.fc_emi_gate_xz2h = nn.Linear(x_dim+z_dim, emi_dim)
        self.fc_emi_gate_h2xz = nn.Linear(emi_dim, x_dim)
        self.fc_emi_proposed_mean_z2h = nn.Linear(z_dim, emi_dim)
        self.fc_emi_proposed_mean_h2z = nn.Linear(emi_dim, x_dim)
        self.fc_emi_x2mu = nn.Linear(x_dim, x_dim)
        self.fc_emi_z2logvar = nn.Linear(x_dim, x_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, z, x):
        batch_size = z.size(0)

        x_0 = torch.zeros(batch_size, 1, self.x_dim).to(self.device)
        x = torch.cat((x_0, x[:, :-1, :]), dim=1)
        xz = torch.cat((x, z), dim=2)

        _gate = self.dropout(self.relu(self.fc_emi_gate_xz2h(xz)))
        gate = self.sigmoid(self.fc_emi_gate_h2xz(_gate))
        _proposed_mean = self.dropout(self.relu(self.fc_emi_proposed_mean_z2h(z)))
        proposed_mean = self.fc_emi_proposed_mean_h2z(_proposed_mean)
        recon_x_mu = (1 - gate) * self.fc_emi_x2mu(x) + gate * proposed_mean
        recon_x_logvar = self.fc_emi_z2logvar(self.dropout(self.relu(proposed_mean)))
        recon_x = torch.cat([recon_x_mu, recon_x_logvar], dim=2)
        return recon_x


class GatedTransition_z2z(nn.Module):
    def __init__(self, dropout_p, time_step, z_dim, trans_dim):
        super(GatedTransition_z2z, self).__init__()
        self.time_step = time_step
        self.fc_gt_gate_z2h = nn.Linear(z_dim, trans_dim)
        self.fc_gt_gate_h2z = nn.Linear(trans_dim, z_dim)
        self.fc_gt_proposed_mean_z2h = nn.Linear(z_dim, trans_dim)
        self.fc_gt_proposed_mean_h2z = nn.Linear(trans_dim, z_dim)
        self.fc_gt_z2mu = nn.Linear(z_dim, z_dim)
        self.fc_gt_z2logvar = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)

    def reparameterize(self, mu, logvar, train=True):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if train:
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, z_q, train):
        batch_size = z_q.size(0)
        # Initialize array
        z_mu = [None for i in range(self.time_step)]
        z_logvar = [None for i in range(self.time_step)]
        z_p = [None for i in range(self.time_step)]

        for t in range(self.time_step):
            z_t_1 = z_q[:, t-1, :]
            _gate = self.dropout(self.relu(self.fc_gt_gate_z2h(z_t_1)))
            gate = self.sigmoid(self.fc_gt_gate_h2z(_gate))
            _proposed_mean = self.dropout(self.relu(self.fc_gt_proposed_mean_z2h(z_t_1)))
            proposed_mean = self.fc_gt_proposed_mean_h2z(_proposed_mean)
            z_mu[t] = (1 - gate) * self.fc_gt_z2mu(z_t_1) + gate * proposed_mean
            z_logvar[t] = self.fc_gt_z2logvar(self.dropout(self.relu(proposed_mean)))
            z_p[t] = self.reparameterize(z_mu[t], z_logvar[t], train)

        z_mu = torch.stack(z_mu, dim=1)
        z_logvar = torch.stack(z_logvar, dim=1)
        z_p = torch.stack(z_p, dim=1)
        return z_p, z_mu, z_logvar


class Discriminator(nn.Module):
    def __init__(self, dropout_p, rnn_dim, rnn_layer, z_dim, disc_dim, class_dim):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=z_dim,
            hidden_size=rnn_dim,
            num_layers=rnn_layer,
            dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_classifier1 = nn.Linear(rnn_dim*2, disc_dim)
        self.fc_classifier2 = nn.Linear(disc_dim, disc_dim)
        self.fc_classifier3 = nn.Linear(disc_dim, class_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, z):
        output, hidden = self.lstm(z)
        output = F.relu(self.fc_classifier1(output[:, -1, :]))
        output = self.dropout(output)
        output = F.relu(self.fc_classifier2(output))
        output = self.dropout(output)
        output = self.fc_classifier3(output)
        return F.log_softmax(output, dim=1)
