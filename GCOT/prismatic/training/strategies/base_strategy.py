"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import (
    PaddedCollatorForActionPrediction,
    PaddedCollatorForLanguageModeling,
    ValPaddedCollatorForActionPrediction,
)
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.solver import Solver
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, random_split
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    @overwatch.rank_zero_only
    def save_val_scores(
        self,
        run_dir,
        epoch,
        eval_state_acc_ls,
        eval_action_acc_ls,
        eval_L1_loss_ls,
        eval_relative_L1_loss_ls,
        gt_ls,
        pred_ls,
        gt_policies_ls,
        pred_policies_ls,
    ):
        save_dir = os.path.join(run_dir, "validation_results")
        os.makedirs(save_dir, exist_ok=True)

        data = []
        for i in range(len(gt_ls)):
            json_obj = {
                "Ground Truth": gt_ls[i],
                "Prediction": pred_ls[i],
                "Ground Truth Policy": gt_policies_ls[i],
                "Prediction Policy": pred_policies_ls[i],
                "Action l1 distance": eval_action_acc_ls[i],
                "State Accuracy": eval_state_acc_ls[i],
                "L1 Loss": eval_L1_loss_ls[i],
                "relative L1 Loss": eval_relative_L1_loss_ls[i],
            }
            data.append(json_obj)

        # Create the filename
        filename = f"epoch_{epoch}_{overwatch.rank()}.json"

        # Full path for the file
        file_path = os.path.join(save_dir, filename)

        # Save the data as JSON
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Validation Results saved to {file_path}")

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Changing Precision for testing
            # print("EDIT: Changed Model Precision to bfloat16!!!")
            # self.vlm.to(torch.bfloat16)
            # for param in self.vlm.parameters():
            #     print("model precision", param.dtype)
            #     break
            # input("Check if this is correct")

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # for param in self.vlm.parameters():
            #     print(param.dtype)

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                # start_fwd = time.time()
                # print('mixed_precision_dtype', self.mixed_precision_dtype)
                # print('enable_mixed_precision_training', self.enable_mixed_precision_training)
                # input() # GPU Usage #  org:31599 ### bfloat16: 31737
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!

                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss
                    # print()
                    # print("Output", output.device)
                    # print("Output", output.logits.device, output.logits.shape, output.logits.dtype)
                    # print("Output", output)
                    # input() # GPU Usage org: 43461 ###bfloat16: 31765
                # print("FWD Pass", time.time()-start_fwd)

                # start_bckd = time.time()
                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()
                # input("backward") # GPU Usage #  org:75183 ### bfloat16: 48771
                # print("Backward Pass", time.time()-start_bckd)

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                # start_metric = time.time()

                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx
                # print("action_gt",action_gt.device,action_gt.shape )
                # print("action_preds",action_preds.device, action_preds.shape)
                # input()

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                # print(correct_preds[0])
                # input()
                action_accuracy = correct_preds.sum().float() / mask.sum().float()
                # print("action_accuracy", action_accuracy.device, action_accuracy.shape)

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                # print("continuous_actions_pred", continuous_actions_pred.device)

                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                # print("continuous_actions_gt", continuous_actions_gt.device)
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                # print("action_l1_loss", action_l1_loss.device)

                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )
                # print("Metric Computation", time.time()-start_metric)

                # === Gradient Step ===

                # print(action_l1_loss.device, action_accuracy.device, continuous_actions_pred.device, continuous_actions_gt.device)
                # input()
                # start_opt = time.time()

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                # input("metric") # GPU Usage #  org:75183 ###bfloat16: 48771
                self.clip_grad_norm()
                # input("clip") # GPU Usage #  org:75183 ### bfloat16: 48771

                # Optimizer & LR Scheduler Step
                self.optimizer.step()  ###76205
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                # print("Optimiser step", time.time()-start_opt)

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)
                # input()

    # === Discrete VLA Training ===

    def run_discrete_vla_training(
        self,
        vla_dataset: IterableDataset,
        val_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        val_collator: ValPaddedCollatorForActionPrediction,
        base_tokenizer,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        seed: int = 7,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        # assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        sampler = DistributedSampler(
            vla_dataset,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=None,
            collate_fn=val_collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        solver = Solver(action_tokenizer, verbose=False)

        best_relative_l1 = float("inf")

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()

                # Zero Gradients (just in case)
                self.optimizer.zero_grad()

                # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
                #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
                #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
                for i, batch in enumerate(dataloader):
                    # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                    #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!

                    with torch.autocast(
                        "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                    ):
                        # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                        )
                        loss = output.loss

                    # Commit Loss =>> Backward!
                    metrics.commit(loss=loss)
                    # metrics.commit(loss=loss, update_step_time=True)
                    loss.backward()

                    # === Compute Action Accuracy & L1 Loss ===

                    if i % 100 == 0:
                        action_preds_tokens = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                        action_preds = base_tokenizer.batch_decode(action_preds_tokens, skip_special_tokens=True)

                        action_gt_tokens = batch["labels"][:, 1:]
                        padding_token_id = base_tokenizer.pad_token_id if base_tokenizer.pad_token_id is not None else 0
                        action_gt_tokens = torch.where(action_gt_tokens == -100, padding_token_id, action_gt_tokens)
                        action_gt = base_tokenizer.batch_decode(action_gt_tokens, skip_special_tokens=True)

                        # Compute Accuracy
                        state_acc, action_acc, L1_loss, relative_L1_loss, _, _ = solver.evaluate_batch(
                            action_gt, action_preds, verbose=overwatch.is_rank_zero()
                        )

                        # Commit Metrics
                        metrics.commit(
                            state_accuracy=torch.tensor(state_acc, dtype=torch.float16),
                            action_accuracy=torch.tensor(action_acc, dtype=torch.float16),
                            l1_loss=torch.tensor(L1_loss, dtype=torch.float16),
                            relative_l1_loss=torch.tensor(relative_L1_loss, dtype=torch.float16),
                            update_step_time=True,
                        )

                    # === Gradient Step ===

                    # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                    self.clip_grad_norm()

                    # Optimizer & LR Scheduler Step
                    self.optimizer.step()  ###76205
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Compute epoch value using number of completed gradient steps
                    # epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                    # Push Metrics
                    metrics.commit(
                        global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0]
                    )
                    status = metrics.push()

                    # Check for Save Interval or Max Steps & Save Checkpoint
                    if terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps):
                        self.save_checkpoint(
                            metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                        )
                        dist.barrier()

                        if terminate:
                            return

                    progress.update()
                    progress.set_description(status)


                self.vlm.eval()
                (
                    eval_state_acc_ls,
                    eval_action_acc_ls,
                    eval_L1_loss_ls,
                    eval_relative_L1_loss_ls,
                    gt_ls,
                    pred_ls,
                    pred_policies_ls,
                    gt_policies_ls,
                ) = ([], [], [], [], [], [], [], [])

                for batch in tqdm(val_dataloader, desc="Validation", disable=not overwatch.is_rank_zero()):
                    with torch.autocast(
                        "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                    ):
                        generated = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                        init_len = generated.size(1)
                        action_gt = batch["labels"][0]
                        max_length = init_len + len(base_tokenizer(action_gt).input_ids) + 20
                        past_key_values = None
                        while True:
                            with torch.no_grad():
                                output: CausalLMOutputWithPast = self.vlm(
                                    input_ids=generated,
                                    attention_mask=attention_mask,
                                    pixel_values=batch["pixel_values"],
                                    labels=None,
                                    # past_key_values = past_key_values
                                )

                                next_token_logits = output.logits[:, -1]
                                next_token = next_token_logits.argmax(dim=1).unsqueeze(-1)
                                generated = torch.cat([generated, next_token.cpu()], dim=1)

                                # Update the attention mask
                                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=1)

                            if next_token.item() == base_tokenizer.eos_token_id or generated.size(1) >= max_length:
                                break

                    # === Compute Action Accuracy & L1 Loss ===

                    action_preds_tokens = generated[:, init_len:-1]  # .argmax(dim=2)
                    action_preds = base_tokenizer.batch_decode(action_preds_tokens, skip_special_tokens=True)

                    # Compute Accuracy

                    state_acc, action_acc, L1_loss, relative_L1_loss, pred_policies, gt_policies = solver.evaluate_batch(
                        batch["labels"], action_preds, verbose=overwatch.is_rank_zero()
                    )

                    pred_ls.extend(action_preds)
                    gt_ls.extend(batch["labels"])
                    pred_policies_ls.extend(pred_policies)
                    gt_policies_ls.extend(gt_policies)
                    eval_state_acc_ls.extend(state_acc)
                    eval_action_acc_ls.extend(action_acc)
                    eval_L1_loss_ls.extend(L1_loss)
                    eval_relative_L1_loss_ls.extend(relative_L1_loss)

                # Commit Metrics
                metrics.commit(
                    state_accuracy_val=torch.tensor(eval_state_acc_ls, dtype=torch.float16).cpu(),
                    action_accuracy_val=torch.tensor(eval_action_acc_ls, dtype=torch.float16).cpu(),
                    l1_loss_val=torch.tensor(eval_L1_loss_ls, dtype=torch.float16).cpu(),
                    relative_l1_loss_val=torch.tensor(eval_relative_L1_loss_ls, dtype=torch.float16).cpu(),
                )
                status = metrics.push()

                self.save_val_scores(
                    metrics.run_dir,
                    epoch,
                    eval_state_acc_ls,
                    eval_action_acc_ls,
                    eval_L1_loss_ls,
                    eval_relative_L1_loss_ls,
                    gt_ls,
                    pred_ls,
                    gt_policies_ls,
                    pred_policies_ls,
                )
                dist.barrier()

                curr_val_relative_l1 = torch.tensor(eval_relative_L1_loss_ls, dtype=torch.float16).cpu().mean().item()

                if curr_val_relative_l1 <= best_relative_l1:
                    # if (curr_val_acc >= best_val_acc):
                    best_relative_l1 = curr_val_relative_l1
                    self.save_checkpoint(
                        metrics.run_dir,
                        metrics.global_step,
                        epoch,
                        loss.item(),
                        only_trainable=not save_full_model,
                        best_checkpoint=True,
                    )
                    dist.barrier()
