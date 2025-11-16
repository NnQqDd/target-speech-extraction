import os
import re
import importlib
import wandb


def load_class(full_name: str): # or use eval to convert to class
    module_name, class_name = full_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_recent_checkpoint(ckpt_dir):
    """
    Recursively search for the newest checkpoint matching epoch_{epoch}.pth.
    Returns (ckpt_path, epoch). If none found, returns (None, 0).
    """
    pattern = re.compile(r'epoch_(\d+)\.pth$')
    newest_path, newest_time, newest_epoch = None, None, 0

    for root, _, files in os.walk(ckpt_dir):
        for fname in files:
            m = pattern.match(fname)
            if m:
                fpath = os.path.join(root, fname)
                ctime = os.path.getctime(fpath)
                if newest_time is None or ctime > newest_time:
                    newest_path = fpath
                    newest_time = ctime
                    newest_epoch = int(m.group(1))

    return newest_path, newest_epoch if newest_path else (None, 0)


def print_num_params(model, show_per_module=False):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,} ({total/1e6:.3f} M)")
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.3f} M)")
    if show_per_module:
        print("\nPer-module parameter counts:")
        for name, module in model.named_children():
            p = sum(p.numel() for p in module.parameters())
            t = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:20s} | total: {p:12,} | trainable: {t:12,}")


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


