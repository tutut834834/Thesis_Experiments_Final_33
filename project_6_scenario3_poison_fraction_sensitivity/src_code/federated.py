import torch
import utils
import models
import math
import copy
import numpy as np
import os
import re
import sys
import random

from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from time import strftime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import H5Dataset


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def make_windows_safe_filename(name):
    """
    Windows does not allow these characters in file/folder names:
    < > : " / \\ | ? *
    """
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.replace(" ", "_")
    return name


def set_all_seeds(seed):
    """
    Makes all poison-fraction sensitivity runs comparable.
    Use the same --seed for every poison_frac run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TeeLogger:
    """
    Writes everything both to the terminal and to a text file.
    This produces the thesis txt logs for each poison-fraction run.
    """

    def __init__(self, terminal_stream, log_file):
        self.terminal_stream = terminal_stream
        self.log_file = log_file

    def write(self, message):
        self.terminal_stream.write(message)
        self.log_file.write(message)
        self.terminal_stream.flush()
        self.log_file.flush()

    def flush(self):
        self.terminal_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal_stream.isatty()


if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0

    set_all_seeds(args.seed)

    poison_frac_tag = str(args.poison_frac).replace(".", "p")

    run_name = (
        f"hyp3_poisonfrac_sensitivity"
        f"_{args.data}"
        f"_r{args.rounds}"
        f"_cpa{args.class_per_agent}"
        f"_pf{poison_frac_tag}"
        f"_pflevel{utils.get_poison_fraction_level(args.poison_frac)}"
        f"_base{args.base_class}"
        f"_target{args.target_class}"
        f"_cor{args.num_corrupt}"
        f"_rlr{args.robustLR_threshold}"
        f"_cl{getattr(args, 'clean_label', 0)}"
        f"_cltype{getattr(args, 'clean_label_type', 0)}"
        f"_seed{args.seed}"
        f"_{strftime('%Y%m%d_%H%M%S')}"
    )

    run_name = make_windows_safe_filename(run_name)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("output_logs", exist_ok=True)

    txt_log_path = os.path.join("output_logs", f"{run_name}_console_output.txt")
    txt_log_file = open(txt_log_path, "a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeLogger(original_stdout, txt_log_file)
    sys.stderr = TeeLogger(original_stderr, txt_log_file)

    print(f"Console output will be saved to: {txt_log_path}")
    print("POISON_FRACTION_SENSITIVITY_RUN_START")
    print(f"scenario=Hypothesis 3 - poison fraction sensitivity")
    print(f"poison_frac_grid={args.poison_frac_grid}")
    print(f"current_poison_frac={args.poison_frac}")
    print(f"current_poison_fraction_level={utils.get_poison_fraction_level(args.poison_frac)}")
    print("POISON_FRACTION_SENSITIVITY_RUN_END")

    utils.print_exp_details(args)

    log_dir = os.path.join("logs", run_name)
    writer = SummaryWriter(log_dir)

    print(f"TensorBoard logs will be saved to: {log_dir}")

    cum_poison_acc_mean = 0

    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    if args.data != 'fedemnist':
        user_groups = utils.distribute_data(train_dataset, args)

    # Poisoned validation set:
    # Even for clean-label training, validation must test base_class + trigger -> target_class.
    # Therefore force_dirty_label=True.
    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)

    utils.poison_dataset(
        poisoned_val_set.dataset,
        args,
        idxs,
        poison_all=True,
        force_dirty_label=True,
        context="poisoned_validation_eval"
    )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    global_model = models.get_model(args.data).to(args.device)
    agents, agent_data_sizes = [], {}

    for _id in range(0, args.num_agents):
        if args.data == 'fedemnist':
            agent = Agent(_id, args)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id])

        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

    n_model_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(
        agent_data_sizes,
        n_model_params,
        poisoned_val_loader,
        args,
        writer
    )
    criterion = nn.CrossEntropyLoss().to(args.device)

    for rnd in tqdm(range(1, args.rounds + 1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}

        for agent_id in np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False
        ):
            update = agents[agent_id].local_train(global_model, criterion)
            agent_updates_dict[agent_id] = update

            # make sure every agent gets same copy of the global model in a round
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())

        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)

        txt_log_file.flush()

        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    val_loader,
                    args
                )

                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)

                print(f'| Round: {rnd} |')
                print(f'| Poison_Frac: {args.poison_frac} | Poison_Fraction_Level: {utils.get_poison_fraction_level(args.poison_frac)} |')
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} |')

                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_loader,
                    args
                )

                cum_poison_acc_mean += poison_acc

                writer.add_scalar(
                    'Scenario3/Poison_Fraction',
                    args.poison_frac,
                    rnd
                )
                writer.add_scalar(
                    'Poison/Base_Class_Accuracy',
                    val_per_class_acc[args.base_class],
                    rnd
                )
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar(
                    'Poison/Cumulative_Poison_Accuracy_Mean',
                    cum_poison_acc_mean / rnd,
                    rnd
                )

                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                print("POISON_FRACTION_SENSITIVITY_METRICS_NOTE: compare Poison Acc, Val_Acc, and labels_changed across pf=0.05,0.10,0.25,0.50.")
                txt_log_file.flush()

    writer.close()
    print('Training has finished!')
    print(f'Full console output saved to: {txt_log_path}')

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    txt_log_file.close()
