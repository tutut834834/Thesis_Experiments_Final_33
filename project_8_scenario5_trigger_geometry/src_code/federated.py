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
    Makes all trigger-geometry runs comparable.
    Use the same --seed for every geometry run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TeeLogger:
    """
    Writes everything both to the terminal and to a text file.
    This produces thesis txt logs for each trigger geometry run.
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

    args = utils.validate_trigger_geometry_args(args)

    set_all_seeds(args.seed)

    trigger_geometry = utils.get_geometry_from_args(args)
    trigger_metrics = utils.compute_trigger_geometry_metrics(trigger_geometry, dataset=args.data)

    run_name = (
        f"hyp5_trigger_geometry"
        f"_{args.data}"
        f"_r{args.rounds}"
        f"_trig{trigger_geometry}"
        f"_area{trigger_metrics['mask_area']}"
        f"_cpa{args.class_per_agent}"
        f"_base{args.base_class}"
        f"_target{args.target_class}"
        f"_cor{args.num_corrupt}"
        f"_pf{str(args.poison_frac).replace('.', 'p')}"
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
    print("TRIGGER_GEOMETRY_RUN_START")
    print(f"scenario=Hypothesis 5 - trigger geometry")
    print(f"trigger_geometry={trigger_geometry}")
    print(f"trigger_geometry_grid={args.trigger_geometry_grid}")
    print(f"trigger_geometry_index={utils.get_trigger_geometry_index(trigger_geometry)}")
    print(f"trigger_geometry_family={trigger_metrics['geometry_family']}")
    print(f"trigger_mask_area={trigger_metrics['mask_area']}")
    print(f"trigger_mask_density={trigger_metrics['mask_density']}")
    print(f"trigger_compactness_score={trigger_metrics['compactness_score']}")
    print(f"expected_visibility={trigger_metrics['expected_visibility']}")
    print(f"expected_pixel_scale={trigger_metrics['expected_pixel_scale']}")
    print("TRIGGER_GEOMETRY_RUN_END")

    utils.all_trigger_geometry_setup_logs(args)
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
                writer.add_scalar('Scenario5/Trigger_Geometry_Index', utils.get_trigger_geometry_index(trigger_geometry), rnd)
                writer.add_scalar('Scenario5/Trigger_Mask_Area', trigger_metrics['mask_area'], rnd)
                writer.add_scalar('Scenario5/Trigger_Mask_Density', trigger_metrics['mask_density'], rnd)
                writer.add_scalar('Scenario5/Trigger_Compactness', trigger_metrics['compactness_score'], rnd)

                print(f'| Round: {rnd} |')
                print(f'| Trigger_Geometry: {trigger_geometry} | Geometry_Index: {utils.get_trigger_geometry_index(trigger_geometry)} |')
                print(f'| Trigger_Summary: {utils.trigger_geometry_summary_line(args)} |')
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
                print("TRIGGER_GEOMETRY_METRICS_NOTE: compare Poison Acc, Val_Acc, changed_pixels_first_sample, and trigger_mask_area across plus, square, apple.")
                txt_log_file.flush()

    writer.close()
    print('Training has finished!')
    print(f'Full console output saved to: {txt_log_path}')

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    txt_log_file.close()
