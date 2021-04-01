"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import env.grid_4x4_det as env
# import env.experimental as env

# from graphviz import Source
# from torchviz import make_dot

import matplotlib.pyplot as plt

import os
import time
path_identifier = int(time.time())
result_path = f'results/{path_identifier}/'

try:
    os.mkdir('results/')
except OSError:
    pass

try:
    os.mkdir(result_path)
except OSError:
    pass

torch.autograd.set_detect_anomaly(True)

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

avi_range = 1000
avi_eps = 1e-4
pp_iter = 50
min_loss = float('inf')
min_loss_policy, policy_prev = torch.zeros(env.num_states, env.num_actions), torch.zeros(env.num_states, env.num_actions)
min_loss_rewards = torch.zeros(env.num_states, env.num_actions)
min_loss_esvc, prev_esvc = torch.zeros(env.num_states, 1), torch.zeros(env.num_states, 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(env.d_feature+1, 300)
        self.fc2 = nn.Linear(300, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, x):  # x are the features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)

optimizer = optim.SGD(net.parameters(), lr=1e-3)


def avi(s, a, t, ft, r, g_ids):  # ft: forward_transitions
    with torch.no_grad():
        num_s = s.size()[0]
        num_a = a.size()[0]
        q = torch.zeros((num_s, num_a))
        v = torch.ones(num_s) * -float('inf')
        pred_v = v.data.clone()

        converged = False
        i = 0
        softmax = nn.Softmax(dim=1)
        while not converged:
            q[:] = 0
            v[g_ids] = 0
            for sid, state in enumerate(s):
                if sid in g_ids:
                    continue
                for aid, action in enumerate(a):
                    for next_sid in ft[sid][aid]:
                        q[sid][aid] += r[sid][aid] + t[sid][aid][next_sid] * v[next_sid]
            v, _ = torch.max(softmax(q), dim=1)
            v[v != v] = -float('inf')  # replace nan's with -float('inf')

            # print(pred_v, v)
            if torch.max(torch.abs(v-pred_v)) < avi_eps:
                converged = True
            else:
                pred_v = v.data.clone()

            i += 1
            if i > avi_range:
                exit("AVI diverged!!!")

        policy = torch.exp(q.sub(v[:, None]))
        return policy


def policy_prop(s, a, t, bt, pol):
    with torch.no_grad():
        num_s = s.size()[0]
        esvc = torch.zeros(pp_iter, num_s, requires_grad=True)
        esvc_cp = torch.zeros(pp_iter, num_s, requires_grad=True)
        esvc[0, env.start_id] = 1
        esvc_cp[0, :] = esvc[0, :].clone()
        for i in range(pp_iter-1):
            esvc[i, env.goal_ids] = 0
            for sid, state in enumerate(s):
                for aid, action in enumerate(a):
                    for source_state in bt[sid][aid]:
                        esvc[i+1, sid] += t[source_state][aid][sid] * pol[source_state, aid] * esvc[i, source_state]
            esvc_cp[i+1, :] = esvc[i+1, :].clone()
        sum_esvc = torch.sum(esvc_cp, dim=0)
        normalized_esvc = sum_esvc / torch.sum(esvc_cp)
        return normalized_esvc


def calculate_loss(rews, rew_ids, new_rews, save=False):
    global min_loss_policy, min_loss_rewards, policy_prev, min_loss_esvc, prev_esvc

    if save:
        min_loss_rewards = rews.clone()
        min_loss_policy = policy_prev.clone()
        min_loss_esvc = prev_esvc

    for i in range(len(new_rews)):
        state = (rew_ids[0][0][i] / env.num_actions).int()
        action = (rew_ids[0][0][i] % env.num_actions).int()
        rews[state, action] = new_rews[i]

    policy = avi(env.states, env.actions, env.transitions, env.forward_transitions, rews, env.goal_ids)
    policy[policy == torch.tensor(float('inf'))] = 1.79769e+30
    policy /= torch.sum(policy, dim=1).view(-1, 1)

    policy_prev = policy.clone()

    exp_svc = policy_prop(env.states, env.actions, env.transitions, env.backward_transitions, policy)
    exp_svc_norm = exp_svc / torch.sum(exp_svc)
    prev_esvc = exp_svc_norm.clone()

    # diff = F.mse_loss(exp_svc_norm, env.emp_fc)
    diff = env.emp_fc - exp_svc_norm

    return diff.repeat_interleave(env.num_actions), rews


def dme():
    global min_loss, min_loss_policy, min_loss_rewards
    save = False

    epochs = 1000000
    rewards = torch.rand(env.num_states, env.num_actions)  # uniformly random rewards
    losses = torch.zeros(epochs, 1)
    for epoch in range(epochs):
        train_x, train_x_ids = env.get_train_data()
        for b_id, batch in enumerate(train_x):
            net.zero_grad()

            # reward_ids = [(train_x_ids[b_id][0] / env.num_actions).int(),
            # (train_x_ids[b_id][0] % env.num_actions).int()]
            outputs = net(batch)  # forward pass of a batch
            loss, rewards = calculate_loss(rewards, train_x_ids, outputs, save)
            save = False
            outputs.backward(loss.view(env.batch_size, 1))
            optimizer.step()

            sum_loss = torch.sum(loss.pow(2))
            if min_loss > sum_loss:
                min_loss = sum_loss
                save = True
            print(f'{epoch}: {sum_loss}')
            losses[epoch] = sum_loss

#    print(f'policy: {min_loss_policy}')
#    print(f'rewards: {min_loss_rewards}')
#    print(f'esvc: {min_loss_esvc}')
#    print(f'emp_fc: {env.emp_fc}')
    f = open(result_path+'out.txt', 'a')
    f.write(f'policy: {min_loss_policy}\n')
    f.write(f'rewards: {min_loss_rewards}\n')
    f.write(f'esvc: {min_loss_esvc}\n')
    f.write(f'emp_fc: {env.emp_fc}\n')
    f.close()

    plt.plot(range(epochs), losses)
    plt.savefig(result_path+'result.png')


dme()
# print(net(train_x[0]))
