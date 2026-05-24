# MSCRS Refactor Review

Ngày cập nhật: 2026-05-24

## Mục tiêu

Refactor code MSCRS theo hướng gọn hơn, dễ thêm/bớt module hơn, nhưng vẫn giữ logic theo paper và giữ behavior mặc định của code cũ.

Ưu tiên:

- Giữ nguyên loss, ranking và output mặc định.
- Hỗ trợ cả `inspired` và `redial`.
- Tách các khối dùng chung ra module rõ trách nhiệm.
- Giảm duplication giữa train scripts.
- Cho phép tắt/bật graph modality bằng flag thay vì comment code.

## Thay đổi đã làm

### 1. Thêm cấu hình bật/tắt modality

File mới:

- `src/common/modality_config.py`

Nội dung:

- `ModalityConfig` gom các switch:
  - `use_kg_graph`
  - `use_collaborative_graph`
  - `use_text_graph`
  - `use_image_graph`
  - `add_item_semantic_to_entities`
- `add_modality_args(parser)` thêm CLI flags dùng chung.
- `modality_config_from_args(...)` convert args thành config truyền vào prompt encoder.

CLI flags mới:

```bash
--disable_kg_graph
--disable_collaborative_graph
--disable_text_graph
--disable_image_graph
--disable_item_semantic_add
```

Không truyền flag thì vẫn chạy full MSCRS như cũ.

### 2. Tách graph encoder và graph fusion

Files mới:

- `src/common/graph_encoders.py`
- `src/common/graph_fusion.py`

Nội dung:

- `LightGCNConv`: layer LightGCN dùng chung cho collaborative/text/image graph.
- `GraphFusionSettings`: khai báo khác biệt fuse giữa Inspired, Redial và Conv.
- `MultiModalEntityFusion`: gom logic fuse:
  1. KG/R-GCN.
  2. Text semantic graph.
  3. Image semantic graph.
  4. Collaborative semantic graph.
  5. Add item semantic embedding vào global entity table.
  6. Project sang hidden size của GPT/DialoGPT.

Ghi chú:

- Parameter trainable vẫn nằm trong prompt encoder để giảm rủi ro lệch checkpoint.
- Module fusion chỉ gom thuật toán forward, không tự tạo parameter mới.

### 3. Tách semantic graph loaders

File mới:

- `src/common/semantic_graphs.py`

Nội dung:

- `CollaborativeSemanticGraph`: load `edge_index_c.pt`.
- `TextSemanticGraph`: build top-k graph từ `id_embeddings_text.json`.
- `ImageSemanticGraph`: build top-k graph từ `id_embeddings_image.json`.

Ý nghĩa:

- Train/infer không còn import graph builder từ các file `dataset_dbpedia*.py`.
- Text/image feature extraction vẫn là offline/precomputed, dùng theo data sẵn có của repo.

### 4. Tách KG resource loader dùng chung

File mới:

- `src/common/kg_resources.py`

Nội dung:

- `KGProcessingConfig`: cấu hình cách build KG.
- `DBpedia`: loader chung cho entity/relation/item ids và triples.
- `inspired_kg(...)`: cấu hình KG cho Inspired.
- `redial_kg(...)`: cấu hình KG cho Redial.

Wrappers giữ theo dataset/task:

- `src/rec/data/inspired_kg_resources.py`
- `src/rec/data/redial_kg_resources.py`
- `src/conv/data/kg_resources.py`

Lý do vẫn giữ wrapper:

- Import path theo dataset rõ hơn.
- Redial có xử lý thêm self-loop và relation filtering khác Inspired.

### 5. Đổi tên dataset files theo trách nhiệm

Files hiện tại:

- `src/rec/data/inspired_pretrain_dataset.py`
- `src/rec/data/inspired_rec_dataset.py`
- `src/rec/data/redial_pretrain_dataset.py`
- `src/rec/data/redial_rec_dataset.py`
- `src/conv/data/retrieval_prompt_dataset.py`

Files cũ đã xóa sau khi đổi import:

- `src/rec/data/dataset_pre_copy.py`
- `src/rec/data/dataset_rec_copy.py`
- `src/rec/data/dataset_dbpedia.py`
- `src/rec/data/dataset_dbpedia_inspired.py`
- `src/conv/data/dataset_conv_retrieval_prompt.py`
- `src/conv/data/dataset_dbpedia.py`

### 6. Đổi tên prompt encoder cho rõ nghĩa

Recommendation:

- `src/rec/models/rec_prompt_encoder.py`
- `InspiredRecommendationPromptEncoder`
- `RedialRecommendationPromptEncoder`

Conversation:

- `src/conv/models/retrieval_prompt_encoder.py`
- `RetrievalConversationPromptEncoder`
- `InspiredConversationPromptEncoder`

Đã bỏ tên cũ `MMPrompt`, `MMPrompt_inspired` và `KGPrompt` trong path chính.

### 7. Thêm recommendation facade

File mới:

- `src/rec/models/mscrs_rec_model.py`

Nội dung:

- `RecForwardOutput`: gom `rec_loss`, `rec_logits`, `cl_loss`, raw `outputs`.
- `MSCRSRecModel.forward_rec(...)`: gom luồng RoBERTa -> prompt encoder -> fused entity table -> GPT2 rec.
- `MSCRSRecModel.loss_for_backward(...)`: giữ logic loss cũ.
- `MSCRSRecModel.topk_ranks(...)`: gom ranking full entity hoặc chỉ `item_ids`.

Tối ưu computation:

- Prompt encoder có `return_entity_embeds=True`.
- Recommendation path không còn tính fused entity table hai lần trong cùng batch.

### 8. Tách train/eval runner cho recommendation

File mới:

- `src/rec/train/rec_runner.py`

Nội dung:

- `train_rec_epoch(...)`: train loop dùng chung cho pretrain/fine-tune, Inspired/Redial.
- `evaluate_rec_epoch(...)`: eval loop dùng chung, có tùy chọn rank theo `item_ids`.

Files đã wire:

- `src/rec/train/train_pre_inspired.py`
- `src/rec/train/train_rec_inspired.py`
- `src/rec/train/train_pre_redial.py`
- `src/rec/train/train_rec_redial.py`

Khác biệt loss được giữ:

- Inspired pretrain/rec và Redial rec: `rec_loss + 0.0001 * loss_cl`.
- Redial pretrain: giữ logic cũ, `cl_loss_weight=0.0`.

### 9. Thêm conversation facade

File mới:

- `src/conv/models/mscrs_conv_model.py`

Nội dung:

- `MSCRSConvModel.build_augmented_inputs(...)`: gom RoBERTa retrieved prompt -> prompt encoder -> concat với context embeddings.
- `MSCRSConvModel.loss_from_batch(...)`: teacher-forcing loss.
- `MSCRSConvModel.generate_from_batch(...)`: generation path.

Ý nghĩa:

- `train_conv.py` và `infer_conv.py` dùng cùng một path tạo augmented inputs.
- Conversation branch rõ hơn: retrieval prompt data -> RoBERTa -> graph prompt encoder -> DialoGPT.

### 10. Tách retrieval prompt builder

File mới:

- `src/conv/models/retrieval_prompt_builder.py`

Nội dung:

- Cắt `prompt_max_length` token của retrieved examples.
- Reshape thành `(batch, n_examples * prompt_max_length, hidden)`.
- Optional mapping sang GPT word embedding space.
- Concat retrieved prompt embeddings với current context embeddings.
- Build attention mask mới.

Ý nghĩa:

- `RetrievalConversationPromptEncoder.forward(...)` không còn chứa trực tiếp block retrieval/mapping dài và khó đọc.

### 11. Refactor conversation train/infer

Files sửa:

- `src/conv/train_conv.py`
- `src/conv/infer_conv.py`

Thay đổi:

- `train_conv.py` không còn tự nối `inputs_embeds`, labels và attention mask trong loop.
- `infer_conv.py` align với `train_conv.py`, dùng `AutoModelForCausalLM`, retrieval dataset và conv facade.
- Thêm `--gen_model_checkpoint` cho inference nếu muốn load checkpoint generator.

### 12. Đổi tên config conversation

File hiện tại:

- `src/conv/retrieval_prompt_config.py`

File cũ đã xóa:

- `src/conv/config_copy.py`

## Kiểm tra đã chạy

Đã chạy:

```bash
python -m compileall -q src
git diff --check
rg -n "model_prompt|dataset_pre_inspired|dataset_rec\\b|KGPrompt|MMPrompt\\b|MMPrompt_inspired|dataset_rec.py|dataset_pre_copy|dataset_dbpedia|config_copy|prompt_builder" src
AST static import resolver cho mọi import dạng src.*
```

Kết quả:

- Pass compile, không có lỗi cú pháp.
- Pass whitespace check.
- Không còn import nội bộ tới các file/class cũ trong `src`.
- Tất cả import dạng `src.*` trỏ tới module/package còn tồn tại.

Ghi chú: khi kiểm tra tên file cũ `prompt_builder`, cần tránh match nhầm file mới `retrieval_prompt_builder.py`. Pattern kiểm tra chính xác hơn là:

```bash
rg -n "models\\.prompt_builder|src/conv/models/prompt_builder\\.py|model_prompt|dataset_pre_inspired|dataset_rec\\b|KGPrompt|MMPrompt\\b|MMPrompt_inspired|dataset_pre_copy|dataset_dbpedia|config_copy" src
```

Chưa chạy được runtime forward/training trong shell này vì Python environment hiện tại thiếu `torch`:

```text
ModuleNotFoundError: No module named 'torch'
```

## Phạm vi còn cần để ý

1. Chưa smoke test full train/infer trên GPU với checkpoint/data thật.
2. Conversation train loop đã có facade, nhưng chưa tách runner riêng như `src/rec/train/rec_runner.py`.
3. Correlation semantic mapping của conv vẫn là data/offline retrieval trong `data/conv/.../extra_data.py`, chưa thành module runtime sát paper.
4. Prompt encoder vẫn sở hữu nhiều parameter graph để giữ checkpoint compatibility; nếu muốn sạch hơn nữa có thể tách thành submodule có state dict migration rõ ràng.
5. Dataset collator vẫn còn logic tokenize/pad riêng theo task; có thể tách tiếp utility chung nếu cần giảm duplication thêm.

## Trạng thái hiện tại so với mục tiêu mentor nói

Đã đi theo hướng:

- Giữ logic paper.
- Không copy y nguyên code KPICRS.
- Viết lại luồng theo module để dễ nắm:
  - graph resources
  - semantic graphs
  - graph encoders
  - graph fusion
  - prompt encoders
  - task facade
  - train/eval runner cho rec
- Muốn tắt module nào thì dùng flag CLI, không cần comment code.

Phần cần test tiếp là behavior runtime, không phải mapping/import nữa.

## Rà soát repo sau refactor

Ngày rà soát: 2026-05-24

Đã kiểm tra lại các điểm dễ lệch logic MSCRS:

- KG loader:
  - Inspired vẫn dùng KG hai chiều, không self-loop/filter như cũ.
  - Redial vẫn dùng self-loop relation `185` và filter relation count `> 1000` như code cũ.
- Semantic graph top-k:
  - Inspired recommendation dùng `top_k=10`.
  - Redial recommendation dùng `top_k=20`.
  - Conversation dùng `top_k=20`.
- Recommendation loss:
  - Inspired pretrain/rec và Redial rec giữ `rec_loss / grad_acc + 0.0001 * loss_cl`.
  - Redial pretrain giữ `cl_loss_weight=0.0`.
- Recommendation ranking:
  - Pretrain rank trên toàn bộ entity logits.
  - Fine-tune rank trong `kg["item_ids"]` như cũ.
- Conversation:
  - `train_conv.py` và `infer_conv.py` dùng cùng `MSCRSConvModel`.
  - Retrieval prompt vẫn là RoBERTa retrieved examples -> optional mapping -> concat trước current context embeddings.

Đã chạy lại:

```bash
python -m compileall -q src
git diff --check
AST static import resolver cho mọi import dạng src.*
rg kiểm tra import/tên cũ trong src
```

Kết quả:

- Pass compile.
- Pass whitespace check.
- Không còn import lỗi tới các module đã đổi tên.
- Không còn tham chiếu trong `src` tới `model_prompt`, `MMPrompt`, `KGPrompt`, `dataset_dbpedia`, `dataset_pre_copy`, `dataset_rec_copy`, `config_copy`.

Giới hạn hiện tại:

- Chưa thể khẳng định 100% runtime vì môi trường Python hiện tại chưa có `torch`.
- Cần chạy smoke test thật trên môi trường có dependency/model/data để xác nhận forward, checkpoint load và metric không lệch.
