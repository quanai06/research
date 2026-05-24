import numpy as np
import torch
from tqdm.auto import tqdm


def train_rec_epoch(
    *,
    prompt_encoder,
    rec_model,
    train_dataloader,
    accelerator,
    optimizer,
    lr_scheduler,
    args,
    progress_bar,
    completed_steps,
    run=None,
    use_rec_prefix=False,
):
    train_loss = []
    prompt_encoder.train()
    for step, batch in enumerate(train_dataloader):
        rec_output = rec_model.forward_rec(batch, use_rec_prefix=use_rec_prefix)
        loss = rec_model.loss_for_backward(
            rec_output,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cl_loss_weight=args.cl_loss_weight,
        )
        accelerator.backward(loss)
        train_loss.append(float(loss))

        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            if args.max_grad_norm is not None:
                accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            completed_steps += 1
            if run:
                run.log({"loss": np.mean(train_loss) * args.gradient_accumulation_steps})

        if completed_steps >= args.max_train_steps:
            break

    return np.mean(train_loss) * args.gradient_accumulation_steps, completed_steps


def evaluate_rec_epoch(
    *,
    prompt_encoder,
    rec_model,
    dataloader,
    evaluator,
    accelerator,
    report_prefix,
    epoch,
    item_ids=None,
    use_rec_prefix=False,
):
    losses = []
    prompt_encoder.eval()
    for batch in tqdm(dataloader):
        with torch.no_grad():
            rec_output = rec_model.forward_rec(batch, use_rec_prefix=use_rec_prefix)
        losses.append(float(rec_output.rec_loss))
        ranks = rec_model.topk_ranks(rec_output.rec_logits, k=50, item_ids=item_ids)
        labels = batch["context"]["rec_labels"]
        evaluator.evaluate(ranks, labels)

    report = accelerator.gather(evaluator.report())
    for key, value in report.items():
        report[key] = value.sum().item()

    prefixed_report = {}
    for key, value in report.items():
        if key != "count":
            prefixed_report[f"{report_prefix}/{key}"] = value / report["count"]
    prefixed_report[f"{report_prefix}/loss"] = np.mean(losses)
    prefixed_report["epoch"] = epoch
    evaluator.reset_metric()
    return prefixed_report
