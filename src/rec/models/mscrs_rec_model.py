from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class RecForwardOutput:
    rec_loss: torch.Tensor
    rec_logits: torch.Tensor
    cl_loss: torch.Tensor
    outputs: object


class MSCRSRecModel(nn.Module):
    """Recommendation facade around text encoder, prompt encoder, and CRS GPT2."""

    def __init__(self, model: nn.Module, text_encoder: nn.Module, prompt_encoder: nn.Module):
        super().__init__()
        self.model = model
        self.text_encoder = text_encoder
        self.prompt_encoder = prompt_encoder

    def freeze_backbone(self):
        self.model.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def forward_rec(self, batch, use_rec_prefix: bool = False) -> RecForwardOutput:
        with torch.no_grad():
            token_embeds = self.text_encoder(**batch["prompt"]).last_hidden_state

        prompt_result = self.prompt_encoder(
            entity_ids=batch["entity"],
            token_embeds=token_embeds,
            output_entity=True,
            use_rec_prefix=use_rec_prefix,
            return_entity_embeds=True,
        )
        if len(prompt_result) == 3:
            prompt_embeds, cl_loss, entity_embeds = prompt_result
        else:
            prompt_embeds, cl_loss = prompt_result
            entity_embeds = self.prompt_encoder.get_entity_embeds()

        context = dict(batch["context"])
        context["prompt_embeds"] = prompt_embeds
        context["entity_embeds"] = entity_embeds

        outputs = self.model(**context, rec=True)
        if cl_loss is None:
            cl_loss = outputs.rec_loss.new_zeros(())

        return RecForwardOutput(
            rec_loss=outputs.rec_loss,
            rec_logits=outputs.rec_logits,
            cl_loss=cl_loss,
            outputs=outputs,
        )

    @staticmethod
    def loss_for_backward(
        output: RecForwardOutput,
        gradient_accumulation_steps: int,
        cl_loss_weight: float,
    ) -> torch.Tensor:
        loss = output.rec_loss / gradient_accumulation_steps
        if cl_loss_weight:
            loss = loss + output.cl_loss * cl_loss_weight
        return loss

    @staticmethod
    def topk_ranks(logits: torch.Tensor, k: int = 50, item_ids: Optional[list[int]] = None):
        if item_ids is None:
            return torch.topk(logits, k=k, dim=-1).indices.tolist()

        item_logits = logits[:, item_ids]
        ranks = torch.topk(item_logits, k=k, dim=-1).indices.tolist()
        return [[item_ids[rank] for rank in batch_rank] for batch_rank in ranks]
