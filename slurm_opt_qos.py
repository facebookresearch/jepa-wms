import subprocess
import re
import time
import argparse

# CONFIGURATION
DEV_QOS = "dev"
LOWEST_QOS = "lowest"
CHECK_INTERVAL = 600  # 10 minutes in seconds
MAX_DEV_GPUS = 16      # Set this to the actual number of GPUs available in dev QOS


def parse_gpu_from_tres(tres_str):
    """Extract GPU count from TRES string like 'gres/gpu:8' or 'gres/gpu=16'"""
    # Try colon format first (from squeue): gres/gpu:8
    match = re.search(r"gres/gpu:(\d+)", tres_str)
    if match:
        return int(match.group(1))
    # Try equals format (from scontrol): gres/gpu=16
    match = re.search(r"gres/gpu=(\d+)", tres_str)
    if match:
        return int(match.group(1))
    return 8  # Default to 1 node (8 GPUs) if not found


def get_all_jobs_info():
    """
    Single squeue call to get all jobs for the user with GPU info.
    Returns: (dev_gpus_used, pending_lowest_tasks)
    where pending_lowest_tasks is a list of (task_id, gpu_count) tuples.
    """
    # Format: job_id, state, qos, tres_per_node, num_nodes
    # %i = job ID, %t = state, %q = QOS, %b = TRES per node, %D = NumNodes
    cmd = [
        "squeue", "--me",
        "--format", "%i|%t|%q|%b|%D",
        "--noheader"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    dev_gpus_used = 0
    pending_lowest_tasks = []

    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) < 5:
            continue

        job_id = parts[0].strip()
        state = parts[1].strip()
        qos = parts[2].strip()
        tres = parts[3].strip()
        num_nodes_str = parts[4].strip()

        gpu_per_node = parse_gpu_from_tres(tres)
        try:
            num_nodes = int(num_nodes_str)
        except ValueError:
            num_nodes = 1

        total_gpus = gpu_per_node * num_nodes

        # Count GPUs used in dev QOS (running or pending, they count against quota)
        if qos == DEV_QOS:
            # For dev jobs, count actual GPUs (expand arrays if needed)
            expanded = expand_job_array(job_id, total_gpus)
            dev_gpus_used += sum(gpu for _, gpu in expanded)

        # Collect pending jobs in lowest QOS
        if qos == LOWEST_QOS and state == "PD":
            # Expand job arrays if needed
            expanded = expand_job_array(job_id, total_gpus)
            pending_lowest_tasks.extend(expanded)

    return dev_gpus_used, pending_lowest_tasks


def expand_job_array(job_id, gpu_count):
    """
    Expand job array notation, e.g. 4319422_[0-7%8] -> [(4319422_0, gpu), ...]
    Returns list of (task_id, gpu_count) tuples.
    """
    match = re.match(r"(\d+)_\[(\d+)-(\d+)(?:%\d+)?\]", job_id)
    if match:
        base_id = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        return [(f"{base_id}_{i}", gpu_count) for i in range(start, end + 1)]
    else:
        return [(job_id, gpu_count)]


def select_tasks_to_move(pending_tasks, available_gpus):
    """
    Select tasks to move to dev QOS.
    pending_tasks: list of (task_id, gpu_count) tuples
    Prioritizes smaller jobs to maximize usage of limited dev QOS.
    """
    # Sort by GPU requirement ascending (smaller jobs first)
    sorted_tasks = sorted(pending_tasks, key=lambda x: x[1])

    tasks_to_move = []
    gpus_used = 0
    for task_id, gpus_needed in sorted_tasks:
        if gpus_needed > available_gpus:
            # Skip jobs that are too large for dev QOS entirely
            continue
        if gpus_used + gpus_needed <= available_gpus:
            tasks_to_move.append(task_id)
            gpus_used += gpus_needed
    return tasks_to_move

def update_task_qos(task_id, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would move job {task_id} from {LOWEST_QOS} to {DEV_QOS}")
    else:
        cmd = ["scontrol", "update", f"jobid={task_id}", f"qos={DEV_QOS}"]
        subprocess.run(cmd)
        print(f"[ACTION] Moved job {task_id} from {LOWEST_QOS} to {DEV_QOS}")

def main(dry_run=False):
    while True:
        dev_gpus_used, pending_lowest_tasks = get_all_jobs_info()
        available_gpus = MAX_DEV_GPUS - dev_gpus_used
        print(f"[INFO] GPUs used in dev: {dev_gpus_used}, available: {available_gpus}")

        if available_gpus > 0 and pending_lowest_tasks:
            tasks_to_move = select_tasks_to_move(pending_lowest_tasks, available_gpus)
            for task_id in tasks_to_move:
                update_task_qos(task_id, dry_run=dry_run)
            print(f"[INFO] {'Would move' if dry_run else 'Moved'} {len(tasks_to_move)} jobs to dev QOS.")
        else:
            print("[INFO] No available GPUs in dev QOS or no pending jobs in lowest.")

        if dry_run:
            break  # Only run once in dry run mode
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Slurm job QOS usage between dev and lowest using GPU count.")
    parser.add_argument('--dry-run', action='store_true', help='Print actions without making changes')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
