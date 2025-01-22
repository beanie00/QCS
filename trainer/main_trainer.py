import time
import wandb

from utils import create_eval_function, create_dataloader, save_model
from trainer.seq_trainer import SequenceTrainer
from trainer.act_trainer import ActTrainer

class Maintrainer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.variant = experiment.variant
        self.iter = 0

    def train(self, eval_envs, log_to_wb, max_return):
        self._initialize_trainer_and_eval_functions(eval_envs, max_return)

        while self.iter < self.variant["max_iters"]:
            self.experiment.common_dataloader_params.update({
                "trajectories": self.experiment.offline_trajs,
                "num_iters": self.variant["num_updates_per_iter"]
            })
            dataloader = create_dataloader(**self.experiment.common_dataloader_params)

            train_outputs = self.trainer.train_iteration(dataloader=dataloader)
            eval_outputs, eval_reward = self.experiment.evaluate(self.eval_fns)

            outputs = self._prepare_outputs(train_outputs, eval_outputs)

            if log_to_wb:
                wandb.log(outputs)

            self.iter += 1

            save_model(
                self.experiment,
                path_prefix=self.experiment.logger.log_path
            )

    def _initialize_trainer_and_eval_functions(self, eval_envs, max_return):
        if "no-condition" in self.variant["base_arch"] or self.variant["conditioning"] == "subgoal":
            target_rtg = [max_return]
        else:
            if "antmaze" in self.variant["env"]:
                target_rtg = [max_return, 100*max_return]
            else:
                target_rtg = [max_return, 2*max_return]
        self.eval_fns = [create_eval_function(self.experiment, eval_envs, tar, "OFFLINE") for tar in target_rtg]

        if "mlp" in self.variant["base_arch"]:
            self.trainer = ActTrainer(**self.experiment.trainer_params)
        elif "dt" in self.variant["base_arch"] or "dc" in self.variant["base_arch"]:
            self.trainer = SequenceTrainer(**self.experiment.trainer_params)

    def _prepare_outputs(self, train_outputs, eval_outputs):
        outputs = {"time/total": time.time() - self.experiment.start_time}
        outputs.update(train_outputs)
        outputs.update(eval_outputs)
        return outputs