import torch
from torch import nn


class MSCRSConvModel(nn.Module):
    """Conversation facade around text encoder, retrieval prompt encoder, and DialoGPT."""

    def __init__(
        self,
        model: nn.Module,
        text_encoder: nn.Module,
        prompt_encoder: nn.Module,
        n_examples: int,
        prompt_max_length: int,
        mapping: bool = False,
    ):
        super().__init__()
        self.model = model
        self.text_encoder = text_encoder
        self.prompt_encoder = prompt_encoder
        self.n_examples = n_examples
        self.prompt_max_length = prompt_max_length
        self.mapping = mapping

    def build_augmented_inputs(self, batch):
        with torch.no_grad():
            token_embeds = self.text_encoder(**batch["prompt"]).last_hidden_state

        inputs_embeds, attention_mask, _, _ = self.prompt_encoder(
            entity_ids=batch["entity"],
            token_embeds=token_embeds,
            output_entity=False,
            use_conv_prefix=True,
            mapping=self.mapping,
            word_embeddings=self.model.get_input_embeddings().weight,
            context_input_embeddings=self.model.get_input_embeddings()(batch["context"]["input_ids"]),
            attention_mask=batch["context"]["attention_mask"],
        )
        return inputs_embeds, attention_mask

    def pad_labels_for_prompt(self, labels, attention_mask):
        prompt_len = self.n_examples * self.prompt_max_length
        pad_resp = -100 * torch.ones(
            (attention_mask.shape[0], prompt_len),
            device=attention_mask.device,
            dtype=torch.long,
        )
        return torch.cat([pad_resp, labels], dim=1)

    def loss_from_batch(self, batch):
        inputs_embeds, attention_mask = self.build_augmented_inputs(batch)
        labels = self.pad_labels_for_prompt(batch["resp"], attention_mask)
        loss = self.model(
            inputs_embeds=inputs_embeds,
            input_ids=None,
            labels=labels,
            return_dict=True,
        )["loss"]
        return loss

    def generate_from_batch(self, batch, generation_model, max_new_tokens: int, no_repeat_ngram_size: int = 3):
        inputs_embeds, _ = self.build_augmented_inputs(batch)
        return generation_model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
