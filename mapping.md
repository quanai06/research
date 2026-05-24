# Mapping từ Paper MSCRS sang Code

Ghi chú này ánh xạ các khối trong hình kiến trúc MSCRS sang code hiện tại sau refactor.

Paper tham chiếu: `MSCRS: Multi-modal Semantic Graph Prompt Learning Framework for Conversational Recommender Systems`, arXiv `2504.10921`, đặc biệt Sections 4.1-4.4.

## 1. Tổng Quan Luồng

Paper có thể đọc thành bốn giai đoạn chính:

1. Mã hóa dữ liệu đầu vào:
   conversation history, mentioned entities, textual item features, image item features.
2. Multi-modal semantic graph modeling:
   KG graph, collaborative semantic graph, textual semantic graph, image semantic graph, sau đó fuse thành `E_final`.
3. Recommendation prompt:
   kết hợp RoBERTa conversation representation, mentioned entities và `E_final`, đưa vào DialoGPT để matching item/entity.
4. Conversation prompt:
   dùng retrieved/correlation conversation examples, RoBERTa, fused entity representation và DialoGPT để sinh response.

Code hiện tại đã tách các khối chính thành module rõ hơn. Một số phần trong paper là offline/precomputed data, không chạy trong training runtime.

## 2. File Map Mức Cao

| Khối trong paper | Code hiện tại | Ghi chú |
|---|---|---|
| Conversation History cho rec | `src/rec/data/inspired_pretrain_dataset.py`, `src/rec/data/inspired_rec_dataset.py`, `src/rec/data/redial_pretrain_dataset.py`, `src/rec/data/redial_rec_dataset.py` | Tạo `context`, `prompt`, `entity`, `rec`. |
| Conversation History cho conv | `src/conv/data/retrieval_prompt_dataset.py` | Tạo current context, response target và retrieved prompt examples. |
| Mentioned Entities | trường `entity` trong JSONL data | Entity extraction đã được làm sẵn offline. Runtime chỉ pad và truyền ID. |
| Knowledge Graph resources | `src/common/kg_resources.py` và wrappers trong `src/rec/data/*_kg_resources.py`, `src/conv/data/kg_resources.py` | Load DBpedia triples, entity/relation ids, item ids. |
| Collaborative Semantic Graph | `CollaborativeSemanticGraph` trong `src/common/semantic_graphs.py` | Load `edge_index_c.pt`. |
| Textual Semantic Graph | `TextSemanticGraph` trong `src/common/semantic_graphs.py` | Build top-k cosine graph từ `id_embeddings_text.json`. |
| Image Semantic Graph | `ImageSemanticGraph` trong `src/common/semantic_graphs.py` | Build top-k cosine graph từ `id_embeddings_image.json`. |
| LightGCN layer | `LightGCNConv` trong `src/common/graph_encoders.py` | Dùng cho collaborative/text/image graph. |
| Graph fusion | `MultiModalEntityFusion` trong `src/common/graph_fusion.py` | Fuse KG, collaborative, text, image thành entity table cuối. |
| Rec prompt encoder | `InspiredRecommendationPromptEncoder`, `RedialRecommendationPromptEncoder` trong `src/rec/models/rec_prompt_encoder.py` | Tạo past-key-value prompt cho recommendation. |
| Rec task facade | `MSCRSRecModel` trong `src/rec/models/mscrs_rec_model.py` | Gom RoBERTa -> prompt encoder -> GPT2 rec -> loss/ranking. |
| Rec train/eval loop | `src/rec/train/rec_runner.py` | Dùng chung cho Inspired/Redial, pretrain/fine-tune. |
| Conversation prompt encoder | `RetrievalConversationPromptEncoder`, `InspiredConversationPromptEncoder` trong `src/conv/models/retrieval_prompt_encoder.py` | Encode retrieved examples và entity graph prompt. |
| Retrieval prompt builder | `RetrievalPromptBuilder` trong `src/conv/models/retrieval_prompt_builder.py` | Ghép retrieved prompt embeddings với current context embeddings. |
| Conv task facade | `MSCRSConvModel` trong `src/conv/models/mscrs_conv_model.py` | Gom train loss và generation path cho conversation. |
| Conv train/infer | `src/conv/train_conv.py`, `src/conv/infer_conv.py` | Dùng cùng conv facade. |

## 3. Stage A - Dữ Liệu Đầu Vào

### A1. Conversation History

Recommendation:

- Inspired pretrain: `src/rec/data/inspired_pretrain_dataset.py`
- Inspired fine-tune: `src/rec/data/inspired_rec_dataset.py`
- Redial pretrain: `src/rec/data/redial_pretrain_dataset.py`
- Redial fine-tune: `src/rec/data/redial_rec_dataset.py`

Các dataset này tạo:

- `context`: token IDs cho GPT/DialoGPT.
- `prompt`: token IDs cho RoBERTa.
- `entity`: mentioned entity IDs.
- `rec`: recommendation label/entity label.

Conversation:

- `src/conv/data/retrieval_prompt_dataset.py`

Dataset này tạo:

- `context`: current conversation.
- `resp`: target response.
- `entity`: entity IDs của current và retrieved examples.
- `prompt`: retrieved examples tokenize bằng RoBERTa.
- `retrieved_example_gen_ids`: retrieved examples tokenize bằng DialoGPT.

### A2. Mentioned Entities

Trong paper, mentioned entities được lấy từ conversation history.

Trong repo, phần extraction này đã được làm offline trong data JSONL. Runtime chỉ pad và batch:

- `src/rec/utils.py`
- `src/conv/utils.py`
- collators trong các dataset files.

### A3. Textual Item Features

Paper mô tả dùng GPT-4o tạo textual description rồi encode bằng RoBERTa.

Trong repo:

- Runtime không sinh lại textual descriptions.
- Code đọc `data/common/*/id_embeddings_text.json`.
- Loader/build graph nằm ở `TextSemanticGraph` trong `src/common/semantic_graphs.py`.

### A4. Image Item Features

Paper mô tả thu thập ảnh item rồi encode bằng ViT.

Trong repo:

- Runtime không scrape ảnh hoặc chạy ViT.
- Code đọc `data/common/*/id_embeddings_image.json`.
- Loader/build graph nằm ở `ImageSemanticGraph` trong `src/common/semantic_graphs.py`.

## 4. Stage B - Multi-modal Semantic Graph Modeling

### B1. Knowledge Graph + R-GCN

Paper:

- `Knowledge Graph -> R-GCN -> G_kg Emb`

Code:

- KG loader chung: `src/common/kg_resources.py`
- Inspired wrapper: `src/rec/data/inspired_kg_resources.py`
- Redial wrapper: `src/rec/data/redial_kg_resources.py`
- Conv wrapper: `src/conv/data/kg_resources.py`
- R-GCN parameters vẫn nằm trong prompt encoders:
  - `src/rec/models/rec_prompt_encoder.py`
  - `src/conv/models/retrieval_prompt_encoder.py`

Lý do R-GCN parameter vẫn ở prompt encoder:

- Giữ state dict/checkpoint ít rủi ro hơn.
- `MultiModalEntityFusion` chỉ nhận module/parameter từ prompt encoder để chạy thuật toán fuse.

### B2. Collaborative Semantic Graph + LightGCN

Paper:

- `Collaborative Semantic Graph -> Light-GCN -> G_c Emb`

Code:

- Data loader: `CollaborativeSemanticGraph` trong `src/common/semantic_graphs.py`
- Graph file: `data/common/*/edge_index_c.pt`
- LightGCN layer: `LightGCNConv` trong `src/common/graph_encoders.py`
- Fusion path: `MultiModalEntityFusion` trong `src/common/graph_fusion.py`

### B3. Textual Semantic Graph + LightGCN

Paper:

- `Textual Semantic Graph -> Light-GCN -> G_t Emb`

Code:

- `TextSemanticGraph` trong `src/common/semantic_graphs.py`
- Đọc `id_embeddings_text.json`
- Tính cosine similarity và giữ top-k neighbors.
- LightGCN chạy trong `MultiModalEntityFusion`.

### B4. Image Semantic Graph + LightGCN

Paper:

- `Image Semantic Graph -> Light-GCN -> G_v Emb`

Code:

- `ImageSemanticGraph` trong `src/common/semantic_graphs.py`
- Đọc `id_embeddings_image.json`
- Tính cosine similarity và giữ top-k neighbors.
- LightGCN chạy trong `MultiModalEntityFusion`.

### B5. Fuse thành `E_final`

Paper:

- Fuse `G_kg`, `G_c`, `G_t`, `G_v` thành final entity representation.

Code:

- `src/common/graph_fusion.py`
- `MultiModalEntityFusion`
- `GraphFusionSettings`

Luồng chính:

1. Encode KG bằng R-GCN.
2. Encode text item graph bằng LightGCN.
3. Encode image item graph bằng LightGCN.
4. Encode collaborative graph bằng LightGCN.
5. Fuse text/image item embeddings.
6. Add semantic item embeddings trở lại global entity table nếu config bật.
7. Project entity table sang hidden size của GPT/DialoGPT.

Tắt/bật module:

- `src/common/modality_config.py`
- CLI flags:
  - `--disable_kg_graph`
  - `--disable_collaborative_graph`
  - `--disable_text_graph`
  - `--disable_image_graph`
  - `--disable_item_semantic_add`

## 5. Stage C - Recommendation Branch

### C1. Recommendation Pretrain

Entrypoints:

- `src/rec/train/train_pre_inspired.py`
- `src/rec/train/train_pre_redial.py`

Data:

- Inspired: `src/rec/data/inspired_pretrain_dataset.py`
- Redial: `src/rec/data/redial_pretrain_dataset.py`

Model modules:

- `src/rec/models/rec_prompt_encoder.py`
- `src/rec/models/mscrs_rec_model.py`
- `src/rec/models/model_gpt2.py`

Runtime path:

1. Batch đưa `prompt` vào RoBERTa.
2. `InspiredRecommendationPromptEncoder` hoặc `RedialRecommendationPromptEncoder` tạo prompt embeddings.
3. Prompt encoder trả luôn fused `entity_embeds` khi `return_entity_embeds=True`.
4. `MSCRSRecModel.forward_rec(...)` gọi `PromptGPT2forCRS(... rec=True)`.
5. `MSCRSRecModel.loss_for_backward(...)` giữ đúng loss cũ.

Train/eval loop:

- `src/rec/train/rec_runner.py`
- `train_rec_epoch(...)`
- `evaluate_rec_epoch(...)`

### C2. Recommendation Fine-tune

Entrypoints:

- `src/rec/train/train_rec_inspired.py`
- `src/rec/train/train_rec_redial.py`

Khác pretrain:

- Load prompt encoder checkpoint pretrain.
- Dùng `use_rec_prefix=True`.
- Khi eval/test fine-tune thì rank trong `kg["item_ids"]`.

Matching:

- `PromptGPT2forCRS.forward(... rec=True)` lấy hidden state cuối.
- Tính `rec_logits = hidden_state @ entity_embeds.T`.
- Cross entropy với label `rec`.

## 6. Stage D - Conversation Branch

### D1. Correlation Semantic Mapping

Paper:

- Dùng toàn bộ conversation histories để tìm/fuse các semantic correlated conversations.

Code hiện tại:

- Offline/preprocessing: `data/conv/inspired/extra_data.py`
- Runtime data fields:
  - `mm_contexts`
  - `mm_resps`
  - `retrieved_mm_context_entity`
  - `retrieved_mm_response_entity`

Ghi chú:

- Implementation hiện tại đơn giản hơn hình paper.
- Nó dùng retrieval/cosine theo entity bag và lưu sẵn vào JSONL.
- Đây là phần còn có thể tách thành `src/conv/data/correlation_mapping.py` nếu muốn sạch hơn.

### D2. Conversation Prompt Construction

Entrypoints:

- `src/conv/train_conv.py`
- `src/conv/infer_conv.py`

Data:

- `src/conv/data/retrieval_prompt_dataset.py`

Model modules:

- `src/conv/models/retrieval_prompt_encoder.py`
- `src/conv/models/retrieval_prompt_builder.py`
- `src/conv/models/mscrs_conv_model.py`

Runtime path:

1. Dataset lấy retrieved examples từ `mm_contexts/mm_resps`.
2. RoBERTa encode `batch["prompt"]`.
3. `RetrievalConversationPromptEncoder.forward(...)` lấy entity graph representation.
4. `RetrievalPromptBuilder` reshape retrieved embeddings.
5. Nếu `mapping=True`, map retrieved prompt embedding sang GPT word embedding space.
6. Concat retrieved prompt embeddings trước current context embeddings.
7. `MSCRSConvModel` đưa `inputs_embeds` vào DialoGPT.

### D3. Conversation Generation

Training:

- `MSCRSConvModel.loss_from_batch(...)`
- `src/conv/train_conv.py`

Inference:

- `MSCRSConvModel.generate_from_batch(...)`
- `src/conv/infer_conv.py`

Hiện tại `train_conv.py` và `infer_conv.py` đã dùng cùng model path, không còn lệch kiểu cũ.

## 7. Cấu Trúc Module Hiện Tại

```text
src/common/
  kg_resources.py        # KG/entity/relation/item resource loader
  semantic_graphs.py     # collaborative/text/image semantic graph loaders
  graph_encoders.py      # LightGCN layer
  graph_fusion.py        # MultiModalEntityFusion -> E_final
  modality_config.py     # bật/tắt KG/collab/text/image/item semantic add

src/rec/data/
  inspired_kg_resources.py
  redial_kg_resources.py
  inspired_pretrain_dataset.py
  inspired_rec_dataset.py
  redial_pretrain_dataset.py
  redial_rec_dataset.py

src/rec/models/
  rec_prompt_encoder.py
  mscrs_rec_model.py
  model_gpt2.py

src/rec/train/
  train_pre_inspired.py
  train_rec_inspired.py
  train_pre_redial.py
  train_rec_redial.py
  rec_runner.py

src/conv/data/
  retrieval_prompt_dataset.py
  kg_resources.py

src/conv/models/
  retrieval_prompt_encoder.py
  retrieval_prompt_builder.py
  mscrs_conv_model.py
  model_gpt2.py

src/conv/
  retrieval_prompt_config.py
  train_conv.py
  infer_conv.py
```

## 8. Các Điểm Khác Paper Cần Nhớ

1. Text feature extraction và image feature extraction là offline/precomputed.
   - Training chỉ đọc `id_embeddings_text.json` và `id_embeddings_image.json`.

2. Collaborative graph construction là offline/precomputed.
   - Training chỉ đọc `edge_index_c.pt`.

3. Conversation correlation mapping vẫn là retrieval data được build offline.
   - Paper mô tả semantic mapping graph hơn.
   - Repo hiện dùng retrieved examples trong JSONL.

4. Recommendation và conversation dùng prompt style khác nhau.
   - Rec dùng past-key-value prompt.
   - Conv dùng concat `inputs_embeds`.

5. Prompt encoder vẫn sở hữu nhiều parameter graph.
   - Đây là lựa chọn có chủ đích để tránh làm vỡ checkpoint.
   - Nếu muốn sạch tuyệt đối hơn nữa, cần kế hoạch migrate state dict.
