"""
Generate evaluation configs for different environments.

Usage:
    python -m evals.simu_env_planning.run_eval_grid --env droid
    python -m evals.simu_env_planning.run_eval_grid --env metaworld
    python -m evals.simu_env_planning.run_eval_grid --env robocasa
    python -m evals.simu_env_planning.run_eval_grid --env maze
    python -m evals.simu_env_planning.run_eval_grid --env pusht
    python -m evals.simu_env_planning.run_eval_grid --env wall
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import yaml
from ruamel.yaml import YAML

yaml_rt = YAML(typ="rt")
yaml_rt.preserve_quotes = True

# ============================================================================
# Environment presets - each defines: variants (primary dimension), planners,
# objectives, epochs, and how to apply variant-specific config changes.
# ============================================================================
ENV_PRESETS = {
    "droid": {
        "default_conf": Path("configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml"),
        "variants": [("H1", 1), ("H3", 3), ("H6", 6)],
        "planners": [("ng", "nevergrad"), ("cem", "cem")],
        "objectives": [("L1", "repr_l1"), ("L2", "repr_dist")],
        "epochs": list(np.arange(0, 315, 6)),
        "tag_transform": lambda tag, v: tag.replace("H1", v[0])
        .replace("gH1", f"gH{v[1]}")
        .replace("nas1", f"nas{v[1]}"),
        "apply_variant": lambda conf, v: conf["planner"].update({"horizon": v[1], "num_act_stepped": v[1]})
        or conf["task_specification"].update({"goal_H": v[1]}),
    },
    "metaworld": {
        "default_conf": Path("configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml"),
        "variants": [("reach-wall", "mw-reach-wall"), ("reach", "mw-reach")],
        "planners": [("ng", "nevergrad"), ("cem", "cem")],
        "objectives": [("L1", "repr_l1"), ("L2", "repr_dist")],
        "epochs": list(range(0, 50)),
        "tag_transform": lambda tag, v: tag.replace("reach_", v[0] + "_"),
        "apply_variant": lambda conf, v: conf["task_specification"].update({"task": v[1]}),
    },
    "robocasa": {
        "default_conf": Path("configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml"),
        "variants": [(s, s) for s in ["reach", "pick", "place", "reach-pick", "pick-place", "reach-pick-place"]],
        "planners": [("cem", "cem")],
        "objectives": [("L1", "repr_l1"), ("L2", "repr_dist")],
        "epochs": list(np.arange(0, 315, 6)),
        "tag_transform": lambda tag, v: tag.replace("reach_", v[0] + "_"),
        "apply_variant": lambda conf, v: conf["task_specification"]["env"].update({"subtask": v[1]}),
    },
    "maze": {
        "default_conf": Path("configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml"),
        "variants": [("randstate", "random_state")],
        "planners": [("ng", "nevergrad"), ("cem", "cem")],
        "objectives": [("L1", "repr_l1"), ("L2", "repr_dist")],
        "epochs": list(range(0, 50)),
        "tag_transform": lambda tag, v: tag,
        "apply_variant": lambda conf, v: conf["task_specification"].update({"goal_source": v[1]}),
    },
    "pusht": {
        "default_conf": Path("configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml"),
        "variants": [("dset", "dset")],
        "planners": [("ng", "nevergrad"), ("cem", "cem")],
        "objectives": [("L1", "repr_l1"), ("L2", "repr_dist")],
        "epochs": list(range(0, 50)),
        "tag_transform": lambda tag, v: tag,
        "apply_variant": lambda conf, v: conf["task_specification"].update({"goal_source": v[1]}),
    },
    "wall": {
        "default_conf": Path("configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml"),
        "variants": [("randstate", "random_state")],
        "planners": [("ng", "nevergrad"), ("cem", "cem")],
        "objectives": [("L1", "repr_l1"), ("L2", "repr_dist")],
        "epochs": list(range(0, 50)),
        "tag_transform": lambda tag, v: tag,
        "apply_variant": lambda conf, v: conf["task_specification"].update({"goal_source": v[1]}),
    },
}


def generate_configs(preset: dict, out_dir: Path) -> int:
    """Generate configs by iterating over variants × planners × objectives × epochs."""
    ctr = 0
    with open(preset["default_conf"], "r") as f:
        base_conf = yaml_rt.load(f)
    base_tag = base_conf["tag"].rsplit("/", 1)[0].removesuffix("_decode")

    for variant in preset["variants"]:
        for planner_abbrev, planner_name in preset["planners"]:
            for obj_abbrev, objective in preset["objectives"]:
                for epoch in preset["epochs"]:
                    with open(preset["default_conf"], "r") as f:
                        conf = yaml_rt.load(f)

                    # Build tag
                    tag = (
                        re.sub(r"L[12]", obj_abbrev, base_tag)
                        .replace("ng", planner_abbrev)
                        .replace("cem", planner_abbrev)
                    )
                    tag = preset["tag_transform"](tag, variant)
                    if conf["planner"].get("decode_each_iteration"):
                        tag += "_decode"
                    conf["tag"] = f"{tag}/epoch-{epoch+1}"

                    # Common config updates
                    conf["model_kwargs"]["checkpoint"] = f"jepa-e{epoch}.pth.tar"
                    conf["planner"]["planner_name"] = planner_name
                    conf["planner"]["optimizer_name"] = "NGOpt"
                    conf["planner"]["planning_objective"]["objective_type"] = objective

                    # Variant-specific updates
                    preset["apply_variant"](conf, variant)

                    with open(out_dir / f"{ctr}.yaml", "w") as f:
                        yaml_rt.dump(conf, f)
                    ctr += 1
    return ctr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=str, required=True, choices=list(ENV_PRESETS.keys()))
    parser.add_argument("--out-dir", type=Path, default=Path("configs/cwtemp"))
    parser.add_argument("--config", type=Path, default=None, help="Override default config file")
    parser.add_argument("--planner", type=str, choices=["ng", "cem"], help="Filter to a single planner")
    parser.add_argument("--objective", type=str, choices=["L1", "L2"], help="Filter to a single objective")
    parser.add_argument("--variant", type=str, nargs="+", help="Filter to specific variants (e.g., --variant H1 H3 for droid, --variant reach place for robocasa)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preset = ENV_PRESETS[args.env].copy()
    if args.config:
        preset["default_conf"] = args.config
    if args.planner:
        preset["planners"] = [(a, n) for a, n in preset["planners"] if a == args.planner]
    if args.objective:
        preset["objectives"] = [(a, o) for a, o in preset["objectives"] if a == args.objective]
    if args.variant:
        preset["variants"] = [(a, v) for a, v in preset["variants"] if a in args.variant]

    num_configs = generate_configs(preset, args.out_dir)

    # Write batch config
    batch_config = [os.path.join(args.out_dir, f"{i}.yaml") for i in range(num_configs)]
    batch_path = os.path.join(args.out_dir, "batch.yaml")
    with open(batch_path, "w") as f:
        yaml.dump(batch_config, f, sort_keys=False)

    with open(preset["default_conf"], "r") as f:
        sample_conf = yaml_rt.load(f)

    print(f"Generated {num_configs} configs for {args.env}")
    print(
        f"python -m evals.main_distributed --fname {batch_path} --batch-launch "
        f"--array-parallelism 700 --account fair_amaia_cw_video --qos lowest --time 120 "
        f"--submitit_folder {sample_conf['folder']}/submitit-evals-batches/ --use_config_folder --copy_code"
    )


if __name__ == "__main__":
    main()
