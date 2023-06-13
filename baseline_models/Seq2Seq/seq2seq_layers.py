import torch
import numpy as np
import copy

from alfworld.agents.agent import BaseAgent
import alfworld.agents.modules.memory as memory
from alfworld.agents.modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule, BeamSearchNode
from alfworld.agents.modules.layers import NegativeLogLoss, masked_mean, compute_mask, GetGenerationQValue, CQAttention


def add_layers(net):
    net.aggregation_attention2 = CQAttention(block_hidden_dim=net.block_hidden_dim, dropout=net.dropout)
    net.aggregation_attention_proj2 = torch.nn.Linear(net.block_hidden_dim * 4, net.block_hidden_dim)
    net.cuda()


def agent_update_dagger(agent):
    if agent.recurrent:
        return agent.train_dagger_recurrent()
    else:
        return agent_train_dagger(agent)


def agent_train_dagger(agent):
    if len(agent.dagger_memory) < agent.dagger_replay_batch_size:
        return None
    # import ipdb; ipdb.set_trace()
    transitions = agent.dagger_memory.sample(agent.dagger_replay_batch_size)
    if transitions is None:
        return None
    batch = memory.dagger_transition(*zip(*transitions))

    if agent.action_space == "generation":
        return agent_command_generation_teacher_force(agent, batch.observation_list, batch.task_list, batch.target_list)
    elif agent.action_space in ["admissible", "exhaustive"]:
        return agent.admissible_commands_teacher_force(batch.observation_list, batch.task_list,
                                                       batch.action_candidate_list, batch.target_indices)
    else:
        raise NotImplementedError()


def agent_command_generation_teacher_force(agent, observation_strings, task_desc_strings, target_strings):
    input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in target_strings]
    output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in target_strings]

    partial_observation_strings = [observation_strings[i][0] for i in range(len(observation_strings))]
    full_observation_strings = [observation_strings[i][1] for i in range(len(observation_strings))]

    partial_input_obs = agent.get_word_input(partial_observation_strings)       # observation
    full_input_obs = agent.get_word_input(full_observation_strings)
    input_obs = torch.cat((partial_input_obs, full_input_obs), dim=-1)

    h_obs, obs_mask = agent.encode(partial_observation_strings, use_model="online")
    h_td, td_mask = agent.encode(task_desc_strings, use_model="online")
    h_full_obs, full_obs_mask = agent.encode(full_observation_strings, use_model="online")
    both_obs_mask = torch.cat((obs_mask, full_obs_mask), dim=-1)

    # import ipdb; ipdb.set_trace()
    aggregated_obs_representation = net_aggregate_information(agent.online_net, h_obs, obs_mask, h_td,
                                                              td_mask, h_full_obs, full_obs_mask)  # batch x obs_length x hid

    input_target = agent.get_word_input(input_target_strings)
    ground_truth = agent.get_word_input(output_target_strings)  # batch x target_length
    target_mask = compute_mask(input_target)  # mask of ground truth should be the same
    pred = agent.online_net.decode(input_target, target_mask, aggregated_obs_representation, both_obs_mask, None,
                                   input_obs)  # batch x target_length x vocab

    batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask,
                                 smoothing_eps=agent.smoothing_eps)
    loss = torch.mean(batch_loss)

    if loss is None:
        return None
    # Backpropagate
    agent.online_net.zero_grad()
    agent.optimizer.zero_grad()
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), agent.clip_grad_norm)
    agent.optimizer.step()  # apply gradients
    return to_np(loss)


def net_aggregate_information(net, h_obs, obs_mask, h_td, td_mask, h_full_obs, full_obs_mask):
    aggregated_obs_representation1 = net.aggregation_attention(h_obs, h_td, obs_mask, td_mask)         # batch x obs_length x 4*hid
    aggregated_obs_representation1 = net.aggregation_attention_proj(aggregated_obs_representation1)     # batch x obs_length x hid
    aggregated_obs_representation1 = torch.tanh(aggregated_obs_representation1)
    aggregated_obs_representation1 = aggregated_obs_representation1 * obs_mask.unsqueeze(-1)


    aggregated_obs_representation2 = net.aggregation_attention2(h_full_obs, h_td, full_obs_mask, td_mask)  # batch x obs_length x 4*hid
    aggregated_obs_representation2 = net.aggregation_attention_proj2(aggregated_obs_representation2)  # batch x obs_length x hid
    aggregated_obs_representation2 = torch.tanh(aggregated_obs_representation2)
    aggregated_obs_representation2 = aggregated_obs_representation2 * full_obs_mask.unsqueeze(-1)

    aggregated_obs_representation = torch.cat((aggregated_obs_representation1, aggregated_obs_representation2), dim=-2)

    return aggregated_obs_representation


def agent_command_generation_greedy_generation(agent, observation_strings, task_desc_strings, previous_dynamics):
    with torch.no_grad():
        batch_size = len(observation_strings)

        partial_observation_strings = [observation_strings[i][0] for i in range(len(observation_strings))]
        full_observation_strings = [observation_strings[i][1] for i in range(len(observation_strings))]
        # both_observation_strings = [observation_strings[i][0] + ' ' + observation_strings[i][1] for i in range(len(observation_strings))]

        partial_input_obs = agent.get_word_input(partial_observation_strings)  # observation
        full_input_obs = agent.get_word_input(full_observation_strings)
        input_obs = torch.cat((partial_input_obs, full_input_obs), dim=-1)

        h_obs, obs_mask = agent.encode(partial_observation_strings, use_model="online")
        h_td, td_mask = agent.encode(task_desc_strings, use_model="online")
        h_full_obs, full_obs_mask = agent.encode(full_observation_strings, use_model="online")
        both_obs_mask = torch.cat((obs_mask, full_obs_mask), dim=-1)

        # import ipdb; ipdb.set_trace()
        aggregated_obs_representation = net_aggregate_information(agent.online_net, h_obs, obs_mask, h_td,
                                                                  td_mask, h_full_obs,
                                                                  full_obs_mask)  # batch x obs_length x hid

        if agent.recurrent:
            averaged_representation = agent.online_net.masked_mean(aggregated_obs_representation,
                                                                  obs_mask)  # batch x hid
            current_dynamics = agent.online_net.rnncell(averaged_representation,
                                                       previous_dynamics) if previous_dynamics is not None else agent.online_net.rnncell(
                averaged_representation)
        else:
            current_dynamics = None

        # greedy generation
        input_target_list = [[agent.word2id["[CLS]"]] for i in range(batch_size)]
        eos = np.zeros(batch_size)
        for _ in range(agent.max_target_length):

            input_target = copy.deepcopy(input_target_list)
            input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
            input_target = to_pt(input_target, agent.use_cuda)
            target_mask = compute_mask(input_target)  # mask of ground truth should be the same
            pred = agent.online_net.decode(input_target, target_mask, aggregated_obs_representation, both_obs_mask,
                                          current_dynamics, input_obs)  # batch x target_length x vocab
            # pointer softmax
            pred = to_np(pred[:, -1])  # batch x vocab
            pred = np.argmax(pred, -1)  # batch
            for b in range(batch_size):
                new_stuff = [pred[b]] if eos[b] == 0 else []
                input_target_list[b] = input_target_list[b] + new_stuff
                if pred[b] == agent.word2id["[SEP]"]:
                    eos[b] = 1
            if np.sum(eos) == batch_size:
                break
        res = [agent.tokenizer.decode(item) for item in input_target_list]
        res = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in res]
        res = [item.replace(" in / on ", " in/on ") for item in res]
        return res, current_dynamics
