import argparse
import os
from copy import deepcopy

import yaml

from evals.main_distributed import launch_evals_with_parsed_args


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Where to save evaluation logs",
        required=True,
    )
    parser.add_argument(
        "--fname",
        type=str,
        help="Location of the pretrained model yaml config",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Location of the pretrained model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        help="Specify model checkpoint key if needed",
        default="teacher",
    )
    parser.add_argument(
        "--pred-checkpoint-key",
        type=str,
        help="Specify model checkpoint key if needed",
        default="predictor",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        help="Specify eval tag prefix",
        default=None,
    )
    parser.add_argument(
        "--evals",
        type=str,
        nargs="+",
        help="List of eval configs to run",
        required=True,
    )
    parser.add_argument(
        "--modelcustom",
        nargs="+",
        type=str,
        help="List of modelcustom to use",
        required=True,
    )
    parser.add_argument("--exclude", type=str, help="nodes to exclude from training", default=None)
    parser.add_argument(
        "--account",
        type=str,
        default="jepa",
        help="Cluster account to use when submitting jobs",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="learn",
        help="cluster partition to submit jobs on",
    )
    parser.add_argument(
        "--qos",
        type=str,
        default="lowest",
        help="If specified, qos value to use when submitting jobs",
    )
    parser.add_argument("--time", type=int, default=4300, help="time in minutes to run job")
    parser.add_argument("--printconfig", type=bool, default=True)
    return parser


def launch_job(
    config_fname,
    folder,
    checkpoint,
    modelcustom,
    model_args=None,
    tag_prefix=None,
    account="jepa",
    partition="learn",
    qos="jepa_pretrain",
    time=4320,
    exclude=None,
    checkpoint_key=None,
    pred_checkpoint_key=None,
):

    with open(config_fname, "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Launching {config_fname} for {checkpoint}")
    config = deepcopy(base_config)
    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint {checkpoint} does not exist")
    if "model_kwargs" not in config:
        config["model_kwargs"] = {}
    config["model_kwargs"]["checkpoint"] = checkpoint
    config["model_kwargs"]["module_name"] = modelcustom
    if "pretrain_kwargs" not in config["model_kwargs"]:
        config["model_kwargs"]["pretrain_kwargs"] = {}
    for k, v in model_args.items():
        config["model_kwargs"]["pretrain_kwargs"][k] = v

    if checkpoint_key is not None:
        config["model_kwargs"]["pretrain_kwargs"]["checkpoint_key"] = checkpoint_key
    if pred_checkpoint_key is not None:
        config["model_kwargs"]["pretrain_kwargs"]["pred_checkpoint_key"] = pred_checkpoint_key
    if "folder" in config:
        config["folder"] = folder

    if tag_prefix is not None:
        config["tag"] = f"{tag_prefix}-" + config["tag"]
    nodes = config["nodes"] if "nodes" in config else 8
    tasks_per_node = config["tasks_per_node"] if "tasks_per_node" in config else 8
    cpus_per_task = config["cpus_per_task"] if "cpus_per_task" in config else 32
    if args.printconfig:
        print(config)

    launch_evals_with_parsed_args(
        args_for_evals=config,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        submitit_folder=os.path.join(folder, "submitit-evals"),
        account=account,
        partition=partition,
        qos=qos,
        delay_seconds=5,
        save_configs=True,
        timeout=time,
        dependency=None,
        exclude_nodes=exclude,
    )


def launch_evals(args):
    folder = args.folder
    checkpoint = args.checkpoint
    with open(args.fname, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    model_args = None
    if "model" in model_config:
        model_args = model_config["model"]

    for eval_config, modelcustom in zip(args.evals, args.modelcustom):
        eval_name = os.path.basename(eval_config).split(".")[0]
        print(f"Launch evals {eval_name}")
        launch_job(
            eval_config,
            folder=folder,
            checkpoint=checkpoint,
            model_args=model_args,
            checkpoint_key=args.checkpoint_key,
            pred_checkpoint_key=args.pred_checkpoint_key,
            modelcustom=modelcustom,
            tag_prefix=args.tag_prefix,
            account=args.account,
            partition=args.partition,
            qos=args.qos,
            time=args.time,
            exclude=args.exclude,
        )


if __name__ == "__main__":
    args = create_parser().parse_args()
    launch_evals(args)
