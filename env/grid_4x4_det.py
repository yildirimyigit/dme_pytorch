"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
import math


d_feature = 1  # feature is manh dist. {r(s, a) where s is 1 dim, a is 1}

num_states = 16
num_actions = 4
states = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]).\
    float().view(num_states, 1)
actions = torch.tensor([0, 1, 2, 3]).float().view(num_actions, 1)
goal_ids = [15]
start_id = 0
batch_size = int(num_states * num_actions)

forward_transitions = [[[0], [1], [4], [0]], [[1], [2], [5], [0]], [[2], [3], [6], [1]], [[3], [3], [7], [2]],
                       [[0], [5], [8], [4]], [[1], [6], [9], [4]], [[2], [7], [10], [5]], [[3], [7], [11], [6]],
                       [[4], [9], [12], [8]], [[5], [10], [13], [8]], [[6], [11], [14], [9]], [[7], [11], [15], [10]],
                       [[8], [13], [12], [12]], [[9], [14], [13], [12]], [[10], [15], [14], [13]],
                       [[11], [15], [15], [14]]]

backward_transitions = [[[0, 4], [], [], [0, 1]], [[1, 5], [0], [], [2]], [[2, 6], [1], [], [3]],
                        [[3, 7], [2, 3], [], []], [[8], [], [0], [4, 5]], [[9], [4], [1], [6]], [[10], [5], [2], [7]],
                        [[11], [6, 7], [3], []], [[12], [], [4], [8, 9]], [[13], [8], [5], [10]],
                        [[14], [9], [6], [11]], [[15], [10, 11], [7], []], [[], [], [8, 12], [12, 13]],
                        [[], [12], [9, 13], [14]], [[], [13], [10, 14], [15]], [[], [14, 15], [11, 15], []]]

transitions = torch.zeros(num_states, num_actions, num_states)
for i in range(num_states):
    for j in range(num_actions):
        transitions[i][j][forward_transitions[i][j]] = 1

rewards = -torch.ones(num_states, num_actions)
for i in range(num_states):
    for j in range(num_actions):
        if forward_transitions[i][j] == goal_ids:
            rewards[i][j] = 100


def prepare_states(all_states, batch_sz):
    with torch.no_grad():
        num_s = all_states.size()[0]
        states_to_feats = state_to_feature(all_states)
        batches = []
        batches_ids = []
        shuffled_ids = torch.randperm(num_s)
        start_batch = 0
        for _ in range(math.ceil(num_s/batch_sz)):
            batch_ids = shuffled_ids[start_batch:min(start_batch+batch_sz, num_s)]
            batches.append(states_to_feats[batch_ids])
            batches_ids.append([batch_ids])
            start_batch = start_batch+batch_sz
        return batches, batches_ids


def state_to_feature(state_action_batch):  # feature is 1D: manhattan distance to goal
    dists = state_action_batch.clone()
    dists[:, 0] = goal_ids[0] - state_action_batch[:, 0]
    features = (dists/4).int() + (dists % 4)

    # normalization
    features[:, 0] /= torch.max(features[:, 0])
    features[:, 1] /= torch.max(features[:, 1])
    return features


def get_train_data():
    state_action_set = torch.cartesian_prod(states.flatten(), actions.flatten())
    t_x, t_x_ids = prepare_states(state_action_set, batch_size)
    return t_x, t_x_ids


# train_x, train_x_ids = get_train_data()


def calculate_emp_fc():
    # trajectories = torch.tensor([[[0], [4], [8], [12], [13], [14], [15]], [[0], [1], [2], [3], [7], [11], [15]],
    #                              [[0], [4], [5], [9], [10], [14], [15]], [[0], [1], [5], [6], [10], [11], [15]],
    #                              [[0], [4], [8], [9], [10], [14], [15]], [[0], [1], [2], [6], [10], [11], [15]],
    #                              [[0], [4], [5], [6], [10], [14], [15]], [[0], [1], [5], [9], [13], [14], [15]]])
    trajectories = torch.tensor([[[0], [4], [8], [12], [13], [14], [15]]])
    empfc = torch.zeros(num_states)
    for traj in trajectories:
        for sid in range(num_states):
            if sid in traj:
                empfc[sid] += 1

    empfc = empfc / torch.sum(empfc)
    return empfc


emp_fc = calculate_emp_fc()
