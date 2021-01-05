"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
import math


d_feature = 3  # r(s, a) where s is 2 dim, a is 1

num_states = 6
num_actions = 2
states = torch.tensor([[-1, 1], [0, 1], [1, 1], [2, 1], [2, 2], [3, 2]]).float()
actions = torch.tensor([-1, 1]).float().view(2, 1)
goal_ids = [4]
start_id = 0
goal_pose = 0.45
batch_size = num_states * num_actions

forward_transitions = [[[0], [1]], [[0], [2]], [[1], [3]], [[2], [4]], [[3], [5]], [[4], [5]]]
backward_transitions = [[[0, 1], []], [[2], [0]], [[3], [1]], [[4], [2]], [[5], [3]], [[], [5]]]
transitions = [[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
               [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]],
               [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]], [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]]
rewards = torch.tensor([[-1, -1], [-1, -1], [-1, -1], [-1, 100], [-1, -1], [100, -1]]).float()


def prepare_states(all_states, batch_sz):  # batch of features
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


def state_to_feature(state_batch):
    state_batch[:, 0] = goal_pose - state_batch[:, 0]   # features = [dist_to_goal, velocity]
    # normalization
    state_batch[:, 0] /= 2.55
    state_batch[:, 1] /= 2.0
    # normalization
    return state_batch


def get_train_data():
    state_action_set = torch.zeros(num_states * num_actions, d_feature)
    for i in range(num_states):
        for j in range(num_actions):
            state_action_set[i*num_actions+j, :] = torch.cat((states[i], actions[j]))
    t_x, t_x_ids = prepare_states(state_action_set, batch_size)
    return t_x, t_x_ids
    

# train_x, train_x_ids = get_train_data()


def calculate_emp_fc():
    trajectories = [[[0], [1], [2], [3], [4]], [[0], [1], [2], [3], [4]], [[0], [1], [2], [3], [4]],
                    [[1], [2], [3], [4]], [[1], [2], [3], [4]],
                    [[2], [3], [4]], [[2], [3], [4]], [[3], [4]], [[5], [4]], [[5], [4]], [[5], [4]]]
    # trajectory = torch.tensor([[0], [1], [2], [3], [4]])
    empfc = torch.zeros(num_states)
    for traj in trajectories:
        for sid in range(num_states):
            if [sid] in traj:
                empfc[sid] += 1

    empfc = empfc / torch.sum(empfc)
    return empfc


emp_fc = calculate_emp_fc()
