import os
import argparse
import yaml
import torch
from baseline_models.Seq2Seq.seq2seq_trainer import train

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


def load_config():
    '''from alfworld.agents.modules.generic'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default='./scripts/seq_config.yaml', help="path to config file")
    parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                        help="override params of the config file,"
                             " e.g. -p 'training.gamma=0.95'")

    # add for HandMeThat
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--data_path', default='./datasets/v2')
    parser.add_argument('--data_dir_name', default='HandMeThat_with_expert_demonstration')
    parser.add_argument('--save_path', default='./checkpoints')
    parser.add_argument('--observability', default='fully', type=str)
    parser.add_argument('--load_pretrained', default=0, type=int)
    parser.add_argument('--load_from_tag', default=None, type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='Seq2Seq', type=str)
    parser.add_argument('--max_train_step', default=100000, type=int)
    parser.add_argument('--report_frequency', default=5000, type=int)
    parser.add_argument('--given_goal', default=0, type=int)
    parser.add_argument('--given_subgoal', default=0, type=int)

    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.load(value)
    # print(config)
    return config, args


def main():
    config, args = load_config()
    MODEL_DIR = os.path.join(args.save_path, args.model, args.observability)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    train(config, args)


if __name__ == '__main__':
    print(device)
    main()
