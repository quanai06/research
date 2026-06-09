import torch
from torch import nn


class RetrievalPromptBuilder(nn.Module):
    """Build DialoGPT inputs from retrieved examples and current context embeddings."""

    def __init__(self, n_examples: int, prompt_max_length: int, hidden_size: int):
        super().__init__()
        self.n_examples = n_examples
        self.prompt_max_length = prompt_max_length
        self.hidden_size = hidden_size

    def forward(
        self,
        *,
        token_embeds: torch.Tensor,
        batch_size: int,
        entity_embeds: torch.Tensor,
        cross_attn: nn.Module,
        word_embeddings: torch.Tensor,
        context_input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        mapping: bool,
    ):
        prompt_embeds = self._flatten_retrieved_prompts(token_embeds, batch_size)

        mapped_entity_embeds = entity_embeds
        if mapping:
            prompt_embeds = self._map_to_word_embedding_space(
                prompt_embeds=prompt_embeds,
                cross_attn=cross_attn,
                word_embeddings=word_embeddings,
            )
            mapped_entity_embeds = self._expand_entity_summary(entity_embeds, prompt_embeds.shape[1])

        prompt_attention_mask = torch.ones(
            (prompt_embeds.shape[0], prompt_embeds.shape[1]),
            device=prompt_embeds.device,
            dtype=attention_mask.dtype,
        )
        inputs_embeds = torch.cat([prompt_embeds, context_input_embeddings], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        assert inputs_embeds.shape[1] == attention_mask.shape[1]
        return inputs_embeds, attention_mask, None, mapped_entity_embeds

    def _flatten_retrieved_prompts(self, token_embeds: torch.Tensor, batch_size: int) -> torch.Tensor:
        prompt_embeds = token_embeds[:, : self.prompt_max_length, :]
        return prompt_embeds.contiguous().view(
            batch_size,
            self.n_examples * self.prompt_max_length,
            self.hidden_size,
        )

    def _map_to_word_embedding_space(
        self,
        *,
        prompt_embeds: torch.Tensor,
        cross_attn: nn.Module,
        word_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        affinity_scores = cross_attn(prompt_embeds) @ word_embeddings.T
        affinity_scores = affinity_scores / self.hidden_size
        return torch.softmax(affinity_scores, dim=-1) @ word_embeddings

    @staticmethod
    def _expand_entity_summary(entity_embeds: torch.Tensor, prompt_len: int) -> torch.Tensor:
        entity_mean = torch.mean(entity_embeds, dim=1)
        return entity_mean.unsqueeze(1).expand(-1, prompt_len, -1)
