import json
import os
import socket
import uuid
from datetime import datetime

import yaml

from src.utils.logging import git_information


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:6]
    return f"{ts}_{short_hash}"


def resolve_run_dir(base_folder: str, run_id: str) -> str:
    return os.path.join(base_folder, "runs", run_id)


def save_resolved_config(run_dir: str, config: dict, filename: str = "config_resolved.yaml"):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_run_metadata(run_dir: str, run_id: str):
    os.makedirs(run_dir, exist_ok=True)
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "username": os.environ.get("USER", "unknown"),
        "git_info": git_information(),
    }
    path = os.path.join(run_dir, "run_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def create_latest_symlink(base_folder: str, run_id: str):
    link_path = os.path.join(base_folder, "latest")
    target = os.path.join("runs", run_id)
    try:
        if os.path.islink(link_path):
            os.unlink(link_path)
        os.symlink(target, link_path)
    except OSError:
        pass
