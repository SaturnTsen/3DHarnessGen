"""Own the W&B run lifecycle; Transformers supplies the metric callback."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Literal
from typing import TYPE_CHECKING

from harness_training.config import StrictConfig, TrainingConfig

if TYPE_CHECKING:
    from harness_training.training.elastic import DataProgress


def record_resume_transition(config: TrainingConfig, progress: DataProgress, world_size: int) -> None:
    """Record topology changes on the existing run without adding fake training steps."""
    if config.tracking.mode == "disabled" or int(os.environ.get("RANK", "0")) != 0:
        return
    import wandb

    if wandb.run is None:
        raise ValueError("Resume transition requires the original W&B run")
    wandb.run.config.update({"post_training": config.model_dump(
        mode="json", exclude={"method": {"backend": {"hub_token"}}})}, allow_val_change=True)
    wandb.run.summary.update({
        "resume/source_world_size": progress.world_size,
        "resume/world_size": world_size,
        "resume/global_step": progress.global_step,
        "resume/epoch": progress.epoch,
        "resume/next_sample": progress.next_sample,
        "resume/global_batch_size": progress.global_batch_size,
    })


class WandbRunState(StrictConfig):
    id: str
    project: str
    entity: str | None
    mode: Literal["offline", "online"]
    url: str | None


@contextmanager
def tracking_session(config: TrainingConfig, resume_from: Path | None, *,
                     is_main_process: bool) -> Iterator[None]:
    tracking = config.tracking
    if tracking.mode == "disabled" or not is_main_process:
        yield
        return

    import wandb
    from wandb.util import generate_id

    if wandb.run is not None:
        raise ValueError("Another W&B run is active; finish it before starting this training run")
    output = config.experiment.output_dir
    state_path = output / "wandb_run.json"
    if resume_from is not None:
        state = WandbRunState.model_validate_json(state_path.read_text())
        if (state.project, state.entity, state.mode) != (tracking.project, tracking.entity, tracking.mode):
            raise ValueError("W&B project, entity or mode changed since the checkpoint")
        if tracking.run_id is not None and tracking.run_id != state.id:
            raise ValueError("W&B run_id does not match the saved training run")
        run_id = state.id
    else:
        if state_path.exists():
            raise ValueError("W&B run state already exists; choose another output_dir or resume a checkpoint")
        run_id = tracking.run_id if tracking.run_id is not None else generate_id()
    output.mkdir(parents=True, exist_ok=True)
    environment = {"WANDB_LOG_MODEL": tracking.log_model, "WANDB_WATCH": tracking.watch}
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        # Offline restarts create local segments with the same ID. The SDK's
        # server-side resume policies apply only when logging online.
        resume = (tracking.checkpoint_resume if resume_from is not None else tracking.new_run_resume
                  ) if tracking.mode == "online" else None
        with wandb.init(
            project=tracking.project, entity=tracking.entity, name=tracking.name,
            group=tracking.group, tags=tracking.tags, job_type=tracking.job_type,
            id=run_id, resume=resume, mode=tracking.mode, dir=str(output),
            settings=wandb.Settings(**tracking.settings.model_dump()),
            config={"post_training": config.model_dump(
                mode="json", exclude={"method": {"backend": {"hub_token"}}})},
        ) as run:
            state = WandbRunState(id=run.id, project=tracking.project, entity=tracking.entity,
                                  mode=tracking.mode, url=run.url)
            state_path.write_text(state.model_dump_json(indent=2) + "\n")
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
