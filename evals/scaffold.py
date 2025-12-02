# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib

from src.utils.logging import get_logger

logger = get_logger("Eval runner scaffold")


def main(eval_name, args_eval, resume_preempt=False):
    logger.info(f"Running evaluation: {eval_name}")
    if eval_name.startswith("app."):
        import_path = f"{eval_name}.eval"
    else:
        import_path = f"evals.{eval_name}.eval"
    return importlib.import_module(import_path).main(args_eval=args_eval, resume_preempt=resume_preempt)
