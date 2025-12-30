import re


def normalize_eval_setup(eval_setup: str) -> str:
    """Normalize eval setup string to handle variations like alpha0.0 vs alpha0.

    Converts alpha values without decimal to include .0 suffix.
    E.g., 'alpha0' -> 'alpha0.0', 'alpha1' -> 'alpha1.0'
    """
    return re.sub(r"_alpha(\d+)(?!\.)", r"_alpha\1.0", eval_setup)


task_groups_mapping = {
    "droid": "DROID",
    "pt": "Push-T",
    "mz": "Maze",
    "wall": "Wall",
    "mw-reach": "MW-\nReach",
    "mw-reach-wall": "MW-\nReach-\nWall",
    "rcasa-reach": "Rc-R",
    "rcasa-pick": "Rc-P",
    "rcasa-place": "Rc-Pl",
    "rcasa-reach-pick": "Rc-RP",
    "rcasa-pick-place": "Rc-PP",
    "rcasa-reach-pick-place": "Rc-RPP",
}

# Hardcoded order for task groups in LaTeX tables
TASK_GROUP_ORDER = ["Maze", "Wall", "Push-T", "MW-\nReach", "MW-\nReach-\nWall", "Rc-R", "Rc-Pl", "DROID"]

best_eval_setup_per_task_group = {
    "Push-T": r"CEM $L_2$",
    "Maze": r"CEM rand $L_2$",
    "Wall": r"CEM rand $L_2$",
    # "MW-\nReach": r"CEM $L_2$",
    # "MW-\nReach-\nWall": r"CEM $L_2$",
    "MW-\nReach": r"NG $L_2$",
    "MW-\nReach-\nWall": r"NG $L_2$",
    "DROID": r"CEM H3 $L_2$ max0.1 ep64",
    # "DROID": r"CEM H3 $L_2$ max0.1",  # maybe comment out for the planner plot
    # "DROID": r"CEM H3 $L_2$ max0.1",  # maybe comment out for the planner plot
    "Rc-R": r"CEM $L_2$ ep32",
    # "Rc-R": r"CEM $L_2$",
    "Rc-P": r"CEM $L_2$",
    "Rc-Pl": r"CEM $L_2$ ep32",
    # "Rc-Pl": r"CEM $L_2$",
    # "Rc-R": r"NG $L_1$",
    # "Rc-Pl": r"NG $L_1$",
    "Rc-RP": r"CEM $L_2$",
    "Rc-PP": r"CEM $L_2$",
    "Rc-RPP": r"CEM $L_2$",
}

eval_setup_aliases_full_plan_step = {
    # source dataset
    "L1_noprop_cem_sourcedset_H6_nas6_ctxt2": r"CEM H6 $L_1$",
    "L2_noprop_cem_sourcedset_H6_nas6_ctxt2": r"CEM H6 $L_2$",
    "L1_noprop_ng_sourcedset_H6_nas6_ctxt2": r"NG H6 $L_1$",
    "L2_noprop_ng_sourcedset_H6_nas6_ctxt2": r"NG H6 $L_2$",
    # H6
    "L1_cem_sourcedset_H6_nas6_ctxt2": r"CEM H6 $L_1$",
    "L2_cem_sourcedset_H6_nas6_ctxt2": r"CEM H6 $L_2$",
    "L2_cem_sourcedset_H6_nas6_maxnorm01_ctxt2": r"CEM H6 $L_2$ max0.1",
    # "L2_cem_sourcedset_H6_nas6_maxnorm01_momentum015_ctxt2": r"CEM H6 $L_2$ max0.1 mom0.15",
    "L1_ng_sourcedset_H6_nas6_ctxt2": r"NG H6 $L_1$",
    "L2_ng_sourcedset_H6_nas6_ctxt2": r"NG H6 $L_2$",
    "L2_ng_sourcedset_H6_nas6_maxnorm01_ctxt2": r"NG H6 $L_2$ max0.1",
    # H3
    "L1_cem_sourcedset_H3_nas3_ctxt2": r"CEM H3 $L_1$",
    "L2_cem_sourcedset_H3_nas3_ctxt2": r"CEM H3 $L_2$",
    "L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2": r"CEM H3 $L_2$ max0.1",
    # "L2_cem_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt2": r"CEM H3 $L_2$ max0.1 mom0.15",
    "L1_ng_sourcedset_H3_nas3_ctxt2": r"NG H3 $L_1$",
    "L2_ng_sourcedset_H3_nas3_ctxt2": r"NG H3 $L_2$",
    "L2_ng_sourcedset_H3_nas3_maxnorm01_ctxt2": r"NG H3 $L_2$ max0.1",
    # "L2_ng_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt2": r"NG H3 $L_2$ max0.1 mom0.15",
    # H1
    "L1_cem_sourcedset_H1_nas1_ctxt2": r"CEM H1 $L_1$",
    "L2_cem_sourcedset_H1_nas1_ctxt2": r"CEM H1 $L_2$",
    "L2_cem_sourcedset_H1_nas1_maxnorm01_ctxt2": r"CEM H1 $L_2$ max0.1",
    # "L2_cem_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt2": r"CEM H1 $L_2$ max0.1 mom0.15",
    "L1_ng_sourcedset_H1_nas1_ctxt2": r"NG H1 $L_1$",
    "L2_ng_sourcedset_H1_nas1_ctxt2": r"NG H1 $L_2$",
    "L2_ng_sourcedset_H1_nas1_maxnorm01_ctxt2": r"NG H1 $L_2$ max0.1",
    # "L2_ng_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt2": r"NG H1 $L_2$ max0.1 mom0.15",
    # rand state
    "L1_noprop_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_1$",
    "L2_noprop_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_2$",
    "L1_noprop_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_1$",
    "L2_noprop_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_2$",
    #
    "L1_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_1$",
    "L2_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_2$",
    "L1_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_1$",
    "L2_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_2$",
    # source expert
    # nas6
    "L1_cem_sourcexp_H6_nas6_ctxt2": "L1 CEM nas=6",
    "L2_cem_sourcexp_H6_nas6_ctxt2": "L2 CEM nas=6",
    "L1_ng_sourcexp_H6_nas6_ctxt2": "L1 NG nas=6",
    "L2_ng_sourcexp_H6_nas6_ctxt2": "L2 NG nas=6",
    # H=15
    "L1_cem_sourcexp_H15_nas15_ctxt2": "L1 CEM H=15",
    "L2_cem_sourcexp_H15_nas15_ctxt2": "L2 CEM H=15",
    "L1_ng_sourcexp_H15_nas15_ctxt2": "L1 NG H=15",
    "L2_ng_sourcexp_H15_nas15_ctxt2": "L2 NG H=15",
}

eval_setup_aliases = {
    "wall": {
        # rand state
        "L1_noprop_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_1$",
        "L2_noprop_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_2$",
        # "L1_noprop_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_1$",
        # "L2_noprop_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_2$",
        #
        "L1_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_1$",
        "L2_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_2$",
        "L1_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_1$",
        "L2_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_2$",
        "L1_gd_sourcerandstate_H6_nas6_ctxt2": r"GD rand $L_1$",
        "L2_gd_sourcerandstate_H6_nas6_ctxt2": r"GD rand $L_2$",
        "L1_adam_sourcerandstate_H6_nas6_ctxt2": r"Adam rand $L_1$",
        "L2_adam_sourcerandstate_H6_nas6_ctxt2": r"Adam rand $L_2$",
        # cut at alpha
        "L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.0": r"CEM rand $L_2$ alpha0.0",
        "L2_cem_sourcerandstate_H6_nas6_ctxt2_r256_alpha0.0": r"CEM rand $L_2$ alpha0.0",
        "L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1": r"CEM rand $L_2$ alpha0.1",
        "L2_cem_sourcerandstate_H6_nas6_ctxt2_r256_alpha0.1": r"CEM rand $L_2$ alpha0.1",
        # H5
        "L1_cem_sourcerandstate_H5_nas5_ctxt2": r"CEM $L_1$ H=5",
        "L2_cem_sourcerandstate_H5_nas5_ctxt2": r"CEM $L_2$ H=5",
        "L1_ng_sourcerandstate_H5_nas5_ctxt2": r"NG $L_1$ H=5",
        "L2_ng_sourcerandstate_H5_nas5_ctxt2": r"NG $L_2$ H=5",
            # ctxt 3
            "L1_cem_sourcerandstate_H5_nas5_ctxt3": r"CEM $L_1$ H=5 ctx3",
            "L2_cem_sourcerandstate_H5_nas5_ctxt3": r"CEM $L_2$ H=5 ctx3",
            "L1_ng_sourcerandstate_H5_nas5_ctxt3": r"NG $L_1$ H=5 ctx3",
            "L2_ng_sourcerandstate_H5_nas5_ctxt3": r"NG $L_2$ H=5 ctx3",
    },
    "pt": {
        # source dataset
        "L1_noprop_cem_sourcedset_H6_nas6_ctxt2": r"CEM $L_1$",
        "L2_noprop_cem_sourcedset_H6_nas6_ctxt2": r"CEM $L_2$",
        # "L1_noprop_ng_sourcedset_H6_nas6_ctxt2": r"NG $L_1$",
        # "L2_noprop_ng_sourcedset_H6_nas6_ctxt2": r"NG $L_2$",
        #
        "L1_cem_sourcedset_H6_nas6_ctxt2": r"CEM $L_1$",
        "L2_cem_sourcedset_H6_nas6_ctxt2": r"CEM $L_2$",
        "L1_ng_sourcedset_H6_nas6_ctxt2": r"NG $L_1$",
        "L2_ng_sourcedset_H6_nas6_ctxt2": r"NG $L_2$",
        "L1_gd_sourcedset_H6_nas6_ctxt2": r"GD $L_1$",
        "L2_gd_sourcedset_H6_nas6_ctxt2": r"GD $L_2$",
        "L1_adam_sourcedset_H6_nas6_ctxt2": r"Adam $L_1$",
        "L2_adam_sourcedset_H6_nas6_ctxt2": r"Adam $L_2$",
        # H5
        "L1_cem_sourcedset_H5_nas5_ctxt2": r"CEM $L_1$ H=5",
        "L2_cem_sourcedset_H5_nas5_ctxt2": r"CEM $L_2$ H=5",
        "L1_ng_sourcedset_H5_nas5_ctxt2": r"NG $L_1$ H=5",
        "L2_ng_sourcedset_H5_nas5_ctxt2": r"NG $L_2$ H=5",
            # ctxt 3
            "L1_cem_sourcedset_H5_nas5_ctxt3": r"CEM $L_1$ H=5 ctx3",
            "L2_cem_sourcedset_H5_nas5_ctxt3": r"CEM $L_2$ H=5 ctx3",
            "L1_ng_sourcedset_H5_nas5_ctxt3": r"NG $L_1$ H=5 ctx3",
            "L2_ng_sourcedset_H5_nas5_ctxt3": r"NG $L_2$ H=5 ctx3",
        # cut at alpha
        "L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcedset_H6_nas6_ctxt2_r256_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1": r"CEM $L_2$ alpha0.1",
        "L2_cem_sourcedset_H6_nas6_ctxt2_r256_alpha0.1": r"CEM $L_2$ alpha0.1",
    },
    "mz": {
        # rand state
        "L1_noprop_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_1$",
        "L2_noprop_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_2$",
        # "L1_noprop_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_1$",
        # "L2_noprop_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_2$",
        #
        "L1_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_1$",
        "L2_cem_sourcerandstate_H6_nas6_ctxt2": r"CEM rand $L_2$",
        "L1_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_1$",
        "L2_ng_sourcerandstate_H6_nas6_ctxt2": r"NG rand $L_2$",
        "L1_gd_sourcerandstate_H6_nas6_ctxt2": r"GD rand $L_1$",
        "L2_gd_sourcerandstate_H6_nas6_ctxt2": r"GD rand $L_2$",
        "L1_adam_sourcerandstate_H6_nas6_ctxt2": r"Adam rand $L_1$",
        "L2_adam_sourcerandstate_H6_nas6_ctxt2": r"Adam rand $L_2$",
        # cut at alpha
        "L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.0": r"CEM rand $L_2$ alpha0.0",
        "L2_cem_sourcerandstate_H6_nas6_ctxt2_r256_alpha0.0": r"CEM rand $L_2$ alpha0.0",
        "L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1": r"CEM rand $L_2$ alpha0.1",
        "L2_cem_sourcerandstate_H6_nas6_ctxt2_r256_alpha0.1": r"CEM rand $L_2$ alpha0.1",
        # H5
        "L1_cem_sourcerandstate_H5_nas5_ctxt2": r"CEM $L_1$ H=5",
        "L2_cem_sourcerandstate_H5_nas5_ctxt2": r"CEM $L_2$ H=5",
        "L1_ng_sourcerandstate_H5_nas5_ctxt2": r"NG $L_1$ H=5",
        "L2_ng_sourcerandstate_H5_nas5_ctxt2": r"NG $L_2$ H=5",
            # ctxt 3
            "L1_cem_sourcerandstate_H5_nas5_ctxt3": r"CEM $L_1$ H=5 ctx3",
            "L2_cem_sourcerandstate_H5_nas5_ctxt3": r"CEM $L_2$ H=5 ctx3",
            "L1_ng_sourcerandstate_H5_nas5_ctxt3": r"NG $L_1$ H=5 ctx3",
            "L2_ng_sourcerandstate_H5_nas5_ctxt3": r"NG $L_2$ H=5 ctx3",
    },
    "droid": {
        # H6
        "L1_cem_sourcedset_H6_nas6_ctxt2": r"CEM H6 $L_1$",
        "L2_cem_sourcedset_H6_nas6_ctxt2": r"CEM H6 $L_2$",
        "L2_cem_sourcedset_H6_nas6_maxnorm01_ctxt2": r"CEM H6 $L_2$ max0.1",
        "L1_cem_sourcedset_H6_nas6_maxnorm01_ctxt2": r"CEM H6 $L_1$ max0.1",
        # "L2_cem_sourcedset_H6_nas6_maxnorm01_momentum015_ctxt2": r"CEM H6 $L_2$ max0.1 mom0.15",
        "L1_ng_sourcedset_H6_nas6_ctxt2": r"NG H6 $L_1$",
        "L2_ng_sourcedset_H6_nas6_ctxt2": r"NG H6 $L_2$",
        "L2_ng_sourcedset_H6_nas6_maxnorm01_ctxt2": r"NG H6 $L_2$ max0.1",
        "L1_ng_sourcedset_H6_nas6_maxnorm01_ctxt2": r"NG H6 $L_1$ max0.1",
        # H3
        "L1_cem_sourcedset_H3_nas3_ctxt2": r"CEM H3 $L_1$",
        "L2_cem_sourcedset_H3_nas3_ctxt2": r"CEM H3 $L_2$",
        "L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2": r"CEM H3 $L_2$ max0.1",
        "L1_cem_sourcedset_H3_nas3_maxnorm01_ctxt2": r"CEM H3 $L_1$ max0.1",
        # "L2_cem_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt2": r"CEM H3 $L_2$ max0.1 mom0.15",
        "L1_ng_sourcedset_H3_nas3_ctxt2": r"NG H3 $L_1$",
        "L2_ng_sourcedset_H3_nas3_ctxt2": r"NG H3 $L_2$",
        "L2_ng_sourcedset_H3_nas3_maxnorm01_ctxt2": r"NG H3 $L_2$ max0.1",
        "L1_ng_sourcedset_H3_nas3_maxnorm01_ctxt2": r"NG H3 $L_1$ max0.1",
        # "L2_ng_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt2": r"NG H3 $L_2$ max0.1 mom0.15",
        "L1_gd_sourcedset_H3_nas3_maxnorm01_ctxt2": r"GD H3 $L_1$ max0.1",
        "L2_gd_sourcedset_H3_nas3_maxnorm01_ctxt2": r"GD H3 $L_2$ max0.1",
        "L1_adam_sourcedset_H3_nas3_maxnorm01_ctxt2": r"Adam H3 $L_1$ max0.1",
        "L2_adam_sourcedset_H3_nas3_maxnorm01_ctxt2": r"Adam H3 $L_2$ max0.1",
        # H1
        "L1_cem_sourcedset_H1_nas1_ctxt2": r"CEM H1 $L_1$",
        "L2_cem_sourcedset_H1_nas1_ctxt2": r"CEM H1 $L_2$",
        "L2_cem_sourcedset_H1_nas1_maxnorm01_ctxt2": r"CEM H1 $L_2$ max0.1",
        "L1_cem_sourcedset_H1_nas1_maxnorm01_ctxt2": r"CEM H1 $L_1$ max0.1",
        # "L2_cem_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt2": r"CEM H1 $L_2$ max0.1 mom0.15",
        "L1_ng_sourcedset_H1_nas1_ctxt2": r"NG H1 $L_1$",
        "L2_ng_sourcedset_H1_nas1_ctxt2": r"NG H1 $L_2$",
        "L2_ng_sourcedset_H1_nas1_maxnorm01_ctxt2": r"NG H1 $L_2$ max0.1",
        "L1_ng_sourcedset_H1_nas1_maxnorm01_ctxt2": r"NG H1 $L_1$ max0.1",
        # "L2_ng_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt2": r"NG H1 $L_2$ max0.1 mom0.15",
        # cut at alpha
        "L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.0": r"CEM H6 $L_2$",
        "L2_cem_sourcedset_H6_nas6_ctxt2_r256_alpha0.0": r"CEM H6 $L_2$",
        "L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1": r"CEM H6 $L_2$",
        "L2_cem_sourcedset_H6_nas6_ctxt2_r256_alpha0.1": r"CEM H6 $L_2$",
        "L2_cem_sourcedset_H3_nas3_ctxt2_r224_alpha0.0": r"CEM H3 $L_2$",
        "L2_cem_sourcedset_H3_nas3_ctxt2_r256_alpha0.0": r"CEM H3 $L_2$",
        "L2_cem_sourcedset_H3_nas3_ctxt2_r224_alpha0.1": r"CEM H3 $L_2$",
        "L2_cem_sourcedset_H3_nas3_ctxt2_r256_alpha0.1": r"CEM H3 $L_2$",
        # cut at ep
        # H3
        "L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep64": r"CEM H3 $L_2$ max0.1 ep64",
        "L1_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep64": r"CEM H3 $L_1$ max0.1 ep64",
        "L2_ng_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep64": r"NG H3 $L_2$ max0.1 ep64",
        "L1_ng_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep64": r"NG H3 $L_1$ max0.1 ep64",
        "L1_gd_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep64": r"GD H3 $L_1$ max0.1 ep64",
        "L2_gd_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep64": r"GD H3 $L_2$ max0.1 ep64",
        #
        "L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0.0_ep64": r"CEM H3 $L_2$ max0.1 ep64",
        "L1_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0.0_ep64": r"CEM H3 $L_1$ max0.1 ep64",
        "L2_ng_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0.0_ep64": r"NG H3 $L_2$ max0.1 ep64",
        "L1_ng_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0.0_ep64": r"NG H3 $L_1$ max0.1 ep64",
        "L1_gd_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0.0_ep64": r"GD H3 $L_1$ max0.1 ep64",
        "L2_gd_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0.0_ep64": r"GD H3 $L_2$ max0.1 ep64",
        # ctxt8
        # H6
        "L2_cem_sourcedset_H6_nas6_maxnorm01_ctxt8": r"CEM H6 $L_2$ max0.1",
        "L1_cem_sourcedset_H6_nas6_maxnorm01_ctxt8": r"CEM H6 $L_1$ max0.1",
        # H3
        "L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt8": r"CEM H3 $L_2$ max0.1",
        "L1_cem_sourcedset_H3_nas3_maxnorm01_ctxt8": r"CEM H3 $L_1$ max0.1",
        # H1
        "L2_cem_sourcedset_H1_nas1_maxnorm01_ctxt8": r"CEM H1 $L_2$ max0.1",
        "L1_cem_sourcedset_H1_nas1_maxnorm01_ctxt8": r"CEM H1 $L_1$ max0.1",
    },
    "mw-reach": {
        # source expert
        "L1_noprop_cem_sourcexp_H6_nas3_ctxt2": r"CEM $L_1$",
        "L2_noprop_cem_sourcexp_H6_nas3_ctxt2": r"CEM $L_2$",
        "L1_noprop_ng_sourcexp_H6_nas3_ctxt2": r"NG $L_1$",
        "L2_noprop_ng_sourcexp_H6_nas3_ctxt2": r"NG $L_2$",
        #
        "L1_cem_sourcexp_H6_nas3_ctxt2": r"CEM $L_1$",
        "L2_cem_sourcexp_H6_nas3_ctxt2": r"CEM $L_2$",
        "L1_ng_sourcexp_H6_nas3_ctxt2": r"NG $L_1$",
        "L2_ng_sourcexp_H6_nas3_ctxt2": r"NG $L_2$",
        "L1_gd_sourcexp_H6_nas3_ctxt2": r"GD $L_1$",
        "L2_gd_sourcexp_H6_nas3_ctxt2": r"GD $L_2$",
        "L1_adam_sourcexp_H6_nas3_ctxt2": r"Adam $L_1$",
        "L2_adam_sourcexp_H6_nas3_ctxt2": r"Adam $L_2$",
        # cut at alpha
        "L2_cem_sourcexp_H6_nas3_ctxt1_r224_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r256_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r224_alpha0.1": r"CEM $L_2$ alpha0.1",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r256_alpha0.1": r"CEM $L_2$ alpha0.1",
    },
    "mw-reach-wall": {
        # source expert
        "L1_noprop_cem_sourcexp_H6_nas3_ctxt2": r"CEM $L_1$",
        "L2_noprop_cem_sourcexp_H6_nas3_ctxt2": r"CEM $L_2$",
        "L1_noprop_ng_sourcexp_H6_nas3_ctxt2": r"NG $L_1$",
        "L2_noprop_ng_sourcexp_H6_nas3_ctxt2": r"NG $L_2$",
        #
        "L1_cem_sourcexp_H6_nas3_ctxt2": r"CEM $L_1$",
        "L2_cem_sourcexp_H6_nas3_ctxt2": r"CEM $L_2$",
        "L1_ng_sourcexp_H6_nas3_ctxt2": r"NG $L_1$",
        "L2_ng_sourcexp_H6_nas3_ctxt2": r"NG $L_2$",
        "L1_gd_sourcexp_H6_nas3_ctxt2": r"GD $L_1$",
        "L2_gd_sourcexp_H6_nas3_ctxt2": r"GD $L_2$",
        "L1_adam_sourcexp_H6_nas3_ctxt2": r"Adam $L_1$",
        "L2_adam_sourcexp_H6_nas3_ctxt2": r"Adam $L_2$",
        # cut at alpha
        "L2_cem_sourcexp_H6_nas3_ctxt1_r224_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r256_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r224_alpha0.1": r"CEM $L_2$ alpha0.1",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r256_alpha0.1": r"CEM $L_2$ alpha0.1",
    },
    "rcasa-reach": {
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"CEM $L_2$",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"CEM $L_1$",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"NG $L_2$",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"NG $L_1$",
        "L2_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"GD $L_2$",
        "L1_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"GD $L_1$",
        "L2_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"Adam $L_2$",
        "L1_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"Adam $L_1$",
        # alpha 0
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0": r"CEM $L_1$ alpha0.0",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0": r"NG $L_2$ alpha0.0",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0": r"NG $L_1$ alpha0.0",
        # alpha 0.1
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.1": r"CEM $L_2$ alpha0.1",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.1": r"CEM $L_1$ alpha0.1",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.1": r"NG $L_2$ alpha0.1",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.1": r"NG $L_1$ alpha0.1",
        # cut at ep
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"CEM $L_2$ ep32",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"CEM $L_1$ ep32",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"NG $L_2$ ep32",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"NG $L_1$ ep32",
        "L1_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"GD $L_1$ ep32",
        "L2_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"GD $L_2$ ep32",
        #
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"CEM $L_2$ ep32",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"CEM $L_1$ ep32",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"NG $L_2$ ep32",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"NG $L_1$ ep32",
        "L1_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"GD $L_1$ ep32",
        "L2_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"GD $L_2$ ep32",
        # Adam
        "L2_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"Adam $L_2$ ep32",
        "L1_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"Adam $L_1$ ep32",
        "L2_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"Adam $L_2$ ep32",
        "L1_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"Adam $L_1$ ep32",
    },
    # # "rcasa-pick": {
    # #     'L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2': r"CEM $L_2$",
    # # },
    "rcasa-place": {
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"CEM $L_2$",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"CEM $L_1$",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"NG $L_2$",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"NG $L_1$",
        "L2_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"GD $L_2$",
        "L1_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"GD $L_1$",
        "L2_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"Adam $L_2$",
        "L1_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2": r"Adam $L_1$",
        # alpha 0
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0": r"CEM $L_1$ alpha0.0",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0": r"NG $L_2$ alpha0.0",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0": r"NG $L_1$ alpha0.0",
        # alpha 0.1
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.1": r"CEM $L_2$ alpha0.1",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.1": r"CEM $L_1$ alpha0.1",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.1": r"NG $L_2$ alpha0.1",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.1": r"NG $L_1$ alpha0.1",
        # cut at ep
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"CEM $L_2$ ep32",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"CEM $L_1$ ep32",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"NG $L_2$ ep32",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"NG $L_1$ ep32",
        "L1_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"GD $L_1$ ep32",
        "L2_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"GD $L_2$ ep32",
        #
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"CEM $L_2$ ep32",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"CEM $L_1$ ep32",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"NG $L_2$ ep32",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"NG $L_1$ ep32",
        "L1_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"GD $L_1$ ep32",
        "L2_gd_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"GD $L_2$ ep32",
        # Adam
        "L2_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"Adam $L_2$ ep32",
        "L1_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r256_alpha0.0_ep32": r"Adam $L_1$ ep32",
        "L2_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"Adam $L_2$ ep32",
        "L1_adam_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2_r224_alpha0.0_ep32": r"Adam $L_1$ ep32",
    },
    # "rcasa-reach-pick": {
    #     'L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2': r"CEM $L_2$",
    # },
    # "rcasa-pick-place": {
    #     'L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2': r"CEM $L_2$",
    # },
    # "rcasa-reach-pick-place": {
    #     'L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2': r"CEM $L_2$",
    # },
}


hist1_eval_setup_aliases = {
    "wall": {
        # rand state
        "L1_noprop_cem_sourcerandstate_H6_nas6_ctxt1": r"CEM rand $L_1$",
        "L2_noprop_cem_sourcerandstate_H6_nas6_ctxt1": r"CEM rand $L_2$",
        # "L1_noprop_ng_sourcerandstate_H6_nas6_ctxt1": r"NG rand $L_1$",
        # "L2_noprop_ng_sourcerandstate_H6_nas6_ctxt1": r"NG rand $L_2$",
        #
        "L1_cem_sourcerandstate_H6_nas6_ctxt1": r"CEM rand $L_1$",
        "L2_cem_sourcerandstate_H6_nas6_ctxt1": r"CEM rand $L_2$",
        "L1_ng_sourcerandstate_H6_nas6_ctxt1": r"NG rand $L_1$",
        "L2_ng_sourcerandstate_H6_nas6_ctxt1": r"NG rand $L_2$",
        # H5
        "L1_cem_sourcerandstate_H5_nas5_ctxt1": r"CEM $L_1$ H=5",
        "L2_cem_sourcerandstate_H5_nas5_ctxt1": r"CEM $L_2$ H=5",
        "L1_ng_sourcerandstate_H5_nas5_ctxt1": r"NG $L_1$ H=5",
        "L2_ng_sourcerandstate_H5_nas5_ctxt1": r"NG $L_2$ H=5",
            # ctxt 3
            "L1_cem_sourcerandstate_H5_nas5_ctxt1": r"CEM $L_1$ H=5 ctx3",
            "L2_cem_sourcerandstate_H5_nas5_ctxt1": r"CEM $L_2$ H=5 ctx3",
            "L1_ng_sourcerandstate_H5_nas5_ctxt1": r"NG $L_1$ H=5 ctx3",
            "L2_ng_sourcerandstate_H5_nas5_ctxt1": r"NG $L_2$ H=5 ctx3",
    },
    "pt": {
        # source dataset
        "L1_noprop_cem_sourcedset_H6_nas6_ctxt1": r"CEM $L_1$",
        "L2_noprop_cem_sourcedset_H6_nas6_ctxt1": r"CEM $L_2$",
        # "L1_noprop_ng_sourcedset_H6_nas6_ctxt1": r"NG $L_1$",
        # "L2_noprop_ng_sourcedset_H6_nas6_ctxt1": r"NG $L_2$",
        #
        "L1_cem_sourcedset_H6_nas6_ctxt1": r"CEM $L_1$",
        "L2_cem_sourcedset_H6_nas6_ctxt1": r"CEM $L_2$",
        "L1_ng_sourcedset_H6_nas6_ctxt1": r"NG $L_1$",
        "L2_ng_sourcedset_H6_nas6_ctxt1": r"NG $L_2$",
    },
    "mz": {
        # rand state
        "L1_noprop_cem_sourcerandstate_H6_nas6_ctxt1": r"CEM rand $L_1$",
        "L2_noprop_cem_sourcerandstate_H6_nas6_ctxt1": r"CEM rand $L_2$",
        # "L1_noprop_ng_sourcerandstate_H6_nas6_ctxt1": r"NG rand $L_1$",
        # "L2_noprop_ng_sourcerandstate_H6_nas6_ctxt1": r"NG rand $L_2$",
        #
        "L1_cem_sourcerandstate_H6_nas6_ctxt1": r"CEM rand $L_1$",
        "L2_cem_sourcerandstate_H6_nas6_ctxt1": r"CEM rand $L_2$",
        "L1_ng_sourcerandstate_H6_nas6_ctxt1": r"NG rand $L_1$",
        "L2_ng_sourcerandstate_H6_nas6_ctxt1": r"NG rand $L_2$",
    },
    "droid": {
        # H6
        "L1_cem_sourcedset_H6_nas6_ctxt1": r"CEM H6 $L_1$",
        "L2_cem_sourcedset_H6_nas6_ctxt1": r"CEM H6 $L_2$",
        "L2_cem_sourcedset_H6_nas6_maxnorm01_ctxt1": r"CEM H6 $L_2$ max0.1",
        # "L2_cem_sourcedset_H6_nas6_maxnorm01_momentum015_ctxt1": r"CEM H6 $L_2$ max0.1 mom0.15",
        "L1_ng_sourcedset_H6_nas6_ctxt1": r"NG H6 $L_1$",
        "L2_ng_sourcedset_H6_nas6_ctxt1": r"NG H6 $L_2$",
        "L2_ng_sourcedset_H6_nas6_maxnorm01_ctxt1": r"NG H6 $L_2$ max0.1",
        # H3
        "L1_cem_sourcedset_H3_nas3_ctxt1": r"CEM H3 $L_1$",
        "L2_cem_sourcedset_H3_nas3_ctxt1": r"CEM H3 $L_2$",
        "L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt1": r"CEM H3 $L_2$ max0.1",
        # "L2_cem_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt1": r"CEM H3 $L_2$ max0.1 mom0.15",
        "L1_ng_sourcedset_H3_nas3_ctxt1": r"NG H3 $L_1$",
        "L2_ng_sourcedset_H3_nas3_ctxt1": r"NG H3 $L_2$",
        "L2_ng_sourcedset_H3_nas3_maxnorm01_ctxt1": r"NG H3 $L_2$ max0.1",
        # "L2_ng_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt1": r"NG H3 $L_2$ max0.1 mom0.15",
        # H1
        "L1_cem_sourcedset_H1_nas1_ctxt1": r"CEM H1 $L_1$",
        "L2_cem_sourcedset_H1_nas1_ctxt1": r"CEM H1 $L_2$",
        "L2_cem_sourcedset_H1_nas1_maxnorm01_ctxt1": r"CEM H1 $L_2$ max0.1",
        # "L2_cem_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt1": r"CEM H1 $L_2$ max0.1 mom0.15",
        "L1_ng_sourcedset_H1_nas1_ctxt1": r"NG H1 $L_1$",
        "L2_ng_sourcedset_H1_nas1_ctxt1": r"NG H1 $L_2$",
        "L2_ng_sourcedset_H1_nas1_maxnorm01_ctxt1": r"NG H1 $L_2$ max0.1",
        # "L2_ng_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt1": r"NG H1 $L_2$ max0.1 mom0.15",
    },
    "mw-reach": {
        # source expert
        "L1_noprop_cem_sourcexp_H6_nas3_ctxt1": r"CEM $L_1$",
        "L2_noprop_cem_sourcexp_H6_nas3_ctxt1": r"CEM $L_2$",
        "L1_noprop_ng_sourcexp_H6_nas3_ctxt1": r"NG $L_1$",
        "L2_noprop_ng_sourcexp_H6_nas3_ctxt1": r"NG $L_2$",
        #
        "L1_cem_sourcexp_H6_nas3_ctxt1": r"CEM $L_1$",
        "L2_cem_sourcexp_H6_nas3_ctxt1": r"CEM $L_2$",
        "L1_ng_sourcexp_H6_nas3_ctxt1": r"NG $L_1$",
        "L2_ng_sourcexp_H6_nas3_ctxt1": r"NG $L_2$",
        # cut at alpha
        "L2_cem_sourcexp_H6_nas3_ctxt1_r224_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r256_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r224_alpha0.1": r"CEM $L_2$ alpha0.1",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r256_alpha0.1": r"CEM $L_2$ alpha0.1",
    },
    "mw-reach-wall": {
        # source expert
        "L1_noprop_cem_sourcexp_H6_nas3_ctxt1": r"CEM $L_1$",
        "L2_noprop_cem_sourcexp_H6_nas3_ctxt1": r"CEM $L_2$",
        "L1_noprop_ng_sourcexp_H6_nas3_ctxt1": r"NG $L_1$",
        "L2_noprop_ng_sourcexp_H6_nas3_ctxt1": r"NG $L_2$",
        #
        "L1_cem_sourcexp_H6_nas3_ctxt1": r"CEM $L_1$",
        "L2_cem_sourcexp_H6_nas3_ctxt1": r"CEM $L_2$",
        "L1_ng_sourcexp_H6_nas3_ctxt1": r"NG $L_1$",
        "L2_ng_sourcexp_H6_nas3_ctxt1": r"NG $L_2$",
        #
        "L2_cem_sourcexp_H6_nas3_ctxt1_r224_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r256_alpha0.0": r"CEM $L_2$ alpha0.0",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r224_alpha0.1": r"CEM $L_2$ alpha0.1",
        "L2_cem_sourcexp_H6_nas3_ctxt1_r256_alpha0.1": r"CEM $L_2$ alpha0.1",
    },
    "rcasa-reach": {
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1": r"CEM $L_2$",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1": r"CEM $L_1$",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1": r"NG $L_2$",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1": r"NG $L_1$",
    },
    # "rcasa-pick": {
    #     'L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1': r"CEM $L_2$",
    # },
    "rcasa-place": {
        "L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1": r"CEM $L_2$",
        "L1_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1": r"CEM $L_1$",
        "L2_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1": r"NG $L_2$",
        "L1_ng_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1": r"NG $L_1$",
    },
    # "rcasa-reach-pick": {
    #     'L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1': r"CEM $L_2$",
    # },
    # "rcasa-pick-place": {
    #     'L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1': r"CEM $L_2$",
    # },
    # "rcasa-reach-pick-place": {
    #     'L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt1': r"CEM $L_2$",
    # },
}

# Remove any group from the eval setup figure by commenting out the corresponding key
unif_eval_setup_aliases_across_tasks = {
    "Push-T": {
        "CEM $L_1$": r"CEM $L_1$",
        "CEM $L_2$": r"CEM $L_2$",
        "NG $L_1$": r"NG $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "GD $L_1$": r"GD $L_1$",
        "GD $L_2$": r"GD $L_2$",
        "Adam $L_1$": r"Adam $L_1$",
        "Adam $L_2$": r"Adam $L_2$",
    },
    "Wall": {
        "CEM rand $L_1$": r"CEM $L_1$",
        "CEM rand $L_2$": r"CEM $L_2$",
        "NG rand $L_1$": r"NG $L_1$",
        "NG rand $L_2$": r"NG $L_2$",
        "GD rand $L_1$": r"GD $L_1$",
        "GD rand $L_2$": r"GD $L_2$",
        "Adam rand $L_1$": r"Adam $L_1$",
        "Adam rand $L_2$": r"Adam $L_2$",
    },
    "Maze": {
        "CEM rand $L_1$": r"CEM $L_1$",
        "CEM rand $L_2$": r"CEM $L_2$",
        "NG rand $L_1$": r"NG $L_1$",
        "NG rand $L_2$": r"NG $L_2$",
        "GD rand $L_1$": r"GD $L_1$",
        "GD rand $L_2$": r"GD $L_2$",
        "Adam rand $L_1$": r"Adam $L_1$",
        "Adam rand $L_2$": r"Adam $L_2$",
    },
    "MW-\nReach-\nWall": {
        "CEM $L_1$": r"CEM $L_1$",
        "CEM $L_2$": r"CEM $L_2$",
        "NG $L_1$": r"NG $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "GD $L_1$": r"GD $L_1$",
        "GD $L_2$": r"GD $L_2$",
        "Adam $L_1$": r"Adam $L_1$",
        "Adam $L_2$": r"Adam $L_2$",
    },
    "MW-\nReach": {
        "CEM $L_1$": r"CEM $L_1$",
        "CEM $L_2$": r"CEM $L_2$",
        "NG $L_1$": r"NG $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "GD $L_1$": r"GD $L_1$",
        "GD $L_2$": r"GD $L_2$",
        "Adam $L_1$": r"Adam $L_1$",
        "Adam $L_2$": r"Adam $L_2$",
    },
    "DROID": {
        # "CEM H6 $L_1$": r"CEM $L_1$",
        # "CEM H6 $L_2$": r"CEM $L_2$",
        # "CEM H6 $L_2$ max0.1": "CEM-L2-max0.1",
        # "CEM H6 $L_1$ max0.1": "CEM-L1-max0.1",
        # # "NG H6 $L_1$": r"NG $L_1$",
        # # "NG H6 $L_2$": r"NG $L_2$",
        # "NG H6 $L_2$ max0.1": "NG-L2-max0.1",
        # "NG H6 $L_1$ max0.1": "NG-L1-max0.1",
        # "CEM H3 $L_1$": r"CEM $L_1$",
        # "CEM H3 $L_2$": r"CEM $L_2$",
        # ===
        "CEM H3 $L_2$ max0.1": r"CEM $L_2$",
        "CEM H3 $L_1$ max0.1": r"CEM $L_1$",
        "CEM H3 $L_2$ max0.1 ep64": r"CEM $L_2$",
        "CEM H3 $L_1$ max0.1 ep64": r"CEM $L_1$",
        # ===
        # "CEM H3 $L_2$ max0.1": "CEM-L2-max0.1",
        # "CEM H3 $L_1$ max0.1": "CEM-L1-max0.1",
        # "NG H3 $L_1$": r"NG $L_1$",
        # "NG H3 $L_2$": r"NG $L_2$",
        # ===
        "NG H3 $L_2$ max0.1": r"NG $L_2$",
        "NG H3 $L_1$ max0.1": r"NG $L_1$",
        "NG H3 $L_2$ max0.1 ep64": r"NG $L_2$",
        "NG H3 $L_1$ max0.1 ep64": r"NG $L_1$",
        # ===
        # "NG H3 $L_2$ max0.1": "NG-L2-max0.1",
        # "NG H3 $L_1$ max0.1": "NG-L1-max0.1",
        # "CEM H1 $L_1$": r"CEM $L_1$",
        # "CEM H1 $L_2$": r"CEM $L_2$",
        # "CEM H1 $L_2$ max0.1": "CEM-L2-max0.1",
        # "CEM H1 $L_1$ max0.1": "CEM-L1-max0.1",
        # # "NG H1 $L_1$": r"NG $L_1$",
        # # "NG H1 $L_2$": r"NG $L_2$",
        # "NG H1 $L_2$ max0.1": "NG-L2-max0.1",
        # "NG H1 $L_1$ max0.1": "NG-L1-max0.1",
        "GD H3 $L_2$ max0.1": r"GD $L_2$",
        "GD H3 $L_1$ max0.1": r"GD $L_1$",
        # "GD H3 $L_2$ max0.1 ep64": r"GD $L_2$",
        # "GD H3 $L_1$ max0.1 ep64": r"GD $L_1$",
        "Adam H3 $L_2$ max0.1": r"Adam $L_2$",
        "Adam H3 $L_1$ max0.1": r"Adam $L_1$",
    },
    "Rc-R": {
        "CEM $L_2$": r"CEM $L_2$",
        "CEM $L_1$": r"CEM $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "NG $L_1$": r"NG $L_1$",
        "GD $L_2$": r"GD $L_2$",
        "GD $L_1$": r"GD $L_1$",
        "Adam $L_2$": r"Adam $L_2$",
        "Adam $L_1$": r"Adam $L_1$",
        # ===
        "CEM $L_2$ ep32": r"CEM $L_2$",
        "CEM $L_1$ ep32": r"CEM $L_1$",
        "NG $L_2$ ep32": r"NG $L_2$",
        "NG $L_1$ ep32": r"NG $L_1$",
        "GD $L_2$ ep32": r"GD $L_2$",
        "GD $L_1$ ep32": r"GD $L_1$",
        "Adam $L_2$ ep32": r"Adam $L_2$",
        "Adam $L_1$ ep32": r"Adam $L_1$",
    },
    "Rc-Pl": {
        "CEM $L_2$ ep32": r"CEM $L_2$",
        "CEM $L_1$ ep32": r"CEM $L_1$",
        "NG $L_2$ ep32": r"NG $L_2$",
        "NG $L_1$ ep32": r"NG $L_1$",
        "GD $L_2$ ep32": r"GD $L_2$",
        "GD $L_1$ ep32": r"GD $L_1$",
        "Adam $L_2$ ep32": r"Adam $L_2$",
        "Adam $L_1$ ep32": r"Adam $L_1$",
        # ===
        "CEM $L_2$": r"CEM $L_2$",
        "CEM $L_1$": r"CEM $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "NG $L_1$": r"NG $L_1$",
        "GD $L_2$": r"GD $L_2$",
        "GD $L_1$": r"GD $L_1$",
        "Adam $L_2$": r"Adam $L_2$",
        "Adam $L_1$": r"Adam $L_1$",
    },
    "Rc-P": {
        "CEM $L_2$": r"CEM $L_2$",
        "CEM $L_1$": r"CEM $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "NG $L_1$": r"NG $L_1$",
    },
    "Rc-RP": {
        "CEM $L_2$": r"CEM $L_2$",
        "CEM $L_1$": r"CEM $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "NG $L_1$": r"NG $L_1$",
    },
    "Rc-PP": {
        "CEM $L_2$": r"CEM $L_2$",
        "CEM $L_1$": r"CEM $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "NG $L_1$": r"NG $L_1$",
    },
    "Rc-RPP": {
        "CEM $L_2$": r"CEM $L_2$",
        "CEM $L_1$": r"CEM $L_1$",
        "NG $L_2$": r"NG $L_2$",
        "NG $L_1$": r"NG $L_1$",
    },
}

exclude_eval_folders = {
    "droid": [
        "droid_L2_cem_sourcedset_H6_nas6_ctxt2_gH6_r256_alpha0.0_ep32_decode",
        "droid_L2_cem_sourcedset_H1_nas1_ctxt2_gH6_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H1_nas1_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H1_nas1_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H3_nas3_ctxt2_gH6_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt2_gH6_r256_alpha0.0_ep16_decode",
        "droid_L2_ng_sourcedset_H1_nas1_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_ng_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_ng_sourcedset_H3_nas3_maxnorm01_ctxt2_gH6_r256_alpha0.0_ep16_decode",
        "droid_L2_ng_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt2_gH6_r256_alpha0.0_ep16_decode",
    ]
}

# Define task-specific cut_eval_setup strategies
task_cut_eval_setup_mapping = {
    "droid": "ep",
    "pt": "ctxt",
    "mz": "ctxt",
    "wall": "ctxt",
    "mw-reach": "ctxt",
    "mw-reach-wall": "ctxt",
    "rcasa-reach": "ep",
    "rcasa-pick": "ctxt",
    "rcasa-place": "ep",
    "rcasa-reach-pick": "ctxt",
    "rcasa-pick-place": "ctxt",
    "rcasa-reach-pick-place": "ctxt",
}

last_n_epochs = {
    "droid": 100,
    "pt": 10,
    "mz": 10,
    "wall": 10,
    "mw-reach": 30,
    "mw-reach-wall": 30,
    # "mw-reach": 10,
    # "mw-reach-wall": 10,
    "rcasa-reach": 100,
    "rcasa-pick": 100,
    "rcasa-place": 100,
    "rcasa-reach-pick": 100,
    "rcasa-pick-place": 100,
    "rcasa-reach-pick-place": 100,
}
start_from_epoch = {
    "droid": 215,
    "pt": 40,
    "mz": 40,
    "wall": 40,
    "mw-reach": 20,
    "mw-reach-wall": 20,
    # "mw-reach": 40,
    # "mw-reach-wall": 40,
    "rcasa-reach": 215,
    "rcasa-pick": 215,
    "rcasa-place": 215,
    "rcasa-reach-pick": 215,
    "rcasa-pick-place": 215,
    "rcasa-reach-pick-place": 215,
}
