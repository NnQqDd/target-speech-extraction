import wandb


def remove_empty_runs(entity: str, project: str, least: int = 1, skip_active: bool = True) -> int:
    """
    Delete runs in `entity/project` that have fewer than `least` history rows and are not active.
    Returns number of n_deleted runs.
    """
    api = wandb.Api()
    n_deleted = 0
    active_states = {"running", "starting", "created", "queued", "resuming", "scheduled"}

    for run in api.runs(f"{entity}/{project}"):
        try:
            state = getattr(run, "state", None)
            if state is None:
                info = getattr(run, "info", None)
                if isinstance(info, dict):
                    state = info.get("state")

            if skip_active and state and str(state).lower() in active_states:
                continue

            rows = 0
            try:
                hist = run.history()
                rows = len(hist) if hist is not None else 0
            except Exception:
                try:
                    summary = dict(run.summary)
                    rows = sum(1 for v in summary.values() if v is not None)
                except Exception:
                    rows = 0

            if rows < least:
                try:
                    run.delete()
                    n_deleted += 1
                except Exception:
                    pass
        except Exception:
            pass

    return n_deleted


def get_most_recent_run_id(entity: str, project: str) -> str:
    """
    Return the run_id of the most recently finished run in this project,
    after pruning empty runs.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    for run in runs:
        # skip runs that are still live or have no steps
        if len(run.history()) > 0:
            return run.id, run.name
    return None


def clone_run_history(old_run_id: str, new_run: wandb.sdk.wandb_run.Run):
    raise NotImplementedError("Please be patient!")

