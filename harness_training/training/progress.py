"""Count only updates executed in this session when estimating resumed training time."""

from transformers import ProgressCallback
from transformers.utils import logging


class ResumeProgressCallback(ProgressCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        if state.is_world_process_zero:
            self.training_bar = logging.tqdm(
                total=state.max_steps, initial=state.global_step, dynamic_ncols=True,
            )
