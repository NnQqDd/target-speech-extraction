import os
import re
import importlib
import wandb


def load_class(full_name: str): # or use eval to convert to class
    module_name, class_name = full_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


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


