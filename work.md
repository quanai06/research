# MSCRS Server Setup And Compatibility Work Log

Mục tiêu: sửa compatibility cho Python 3.12.7 và bộ thư viện mới trong `requirements.txt`, không thay đổi logic thí nghiệm/training.

## Server GPU Setup

Các bước sau khi thuê được GPU server và đã copy repo vào server.

### 1. Kiểm tra GPU/CUDA/conda

```bash
nvidia-smi
nvcc --version
conda --version
```

### 2. Tạo môi trường Python 3.12.7

```bash
conda create -n mscrs python=3.12.7 -y
conda activate mscrs
python --version
```

Kỳ vọng:

```text
Python 3.12.7
```

### 3. Cài dependencies

Chạy từ root repo, ví dụ `/research`:

```bash
cd /path/to/research
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Với `torch==2.4.1+cu121`, cài thêm PyG CUDA 12.1 extensions:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

### 4. Kiểm tra PyTorch thấy GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Kỳ vọng ví dụ:

```text
2.4.1+cu121
True
12.1
NVIDIA GeForce RTX 3090
```

### 5. Chạy train trong tmux để tắt máy vẫn chạy

Tạo phiên tmux:

```bash
tmux new -s mscrs
```

Trong tmux:

```bash
conda activate mscrs
cd /path/to/research
```

Chạy pretrain:

```bash
python rec/src/train_pre_inspired.py > log/train_pre_inspired.out 2>&1
```

Detach khỏi tmux nhưng process vẫn chạy:

```text
Ctrl+b
d
```

Vào lại tmux:

```bash
tmux attach -t mscrs
```

Theo dõi log:

```bash
tail -f log/train_pre_inspired.out
```

### 6. Chạy đủ pipeline

Chạy theo thứ tự:

```bash
python rec/src/train_pre_inspired.py > log/train_pre_inspired.out 2>&1
python rec/src/train_rec_inspired.py > log/train_rec_inspired.out 2>&1
python conv/src/train_conv.py > log/train_conv.out 2>&1
```

Nếu `conv` bị CUDA OOM trên RTX 3090 24GB, chạy lại với batch nhỏ hơn:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python conv/src/train_conv.py \
  --fp16 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  > log/train_conv.out 2>&1
```

### 7. Output cần tải về

Nếu chỉ cần metrics:

```text
log/
```

Nếu cần checkpoints:

```text
pretrained/pre-trained-inspired/
pretrained/inspired-model/
prompt-conv-inspired/
```

# Compatibility Changes

## 1. Accelerate mixed precision

Files:
- `rec/src/train_pre_inspired.py`
- `rec/src/train_rec_inspired.py`
- `rec/src/train_pre_redial.py`
- `rec/src/train_rec_redial.py`
- `conv/src/train_conv.py`
- `conv/src/infer_conv.py`

Code cũ:

```python
accelerator = Accelerator()
```

hoặc:

```python
accelerator = Accelerator(device_placement=False)
```

hoặc:

```python
accelerator = Accelerator(device_placement=False, fp16=args.fp16)
```

Code mới:

```python
accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
```

hoặc:

```python
accelerator = Accelerator(device_placement=False, mixed_precision="fp16" if args.fp16 else "no")
```

Note: Không đổi logic model/dataset/loss. Thay đổi này chỉ cập nhật API mới của `accelerate`; flag `--fp16` giờ được truyền đúng vào `Accelerator`.

## 2. Replace `accelerator.use_fp16`

Files:
- `rec/src/train_pre_inspired.py`
- `rec/src/train_pre_redial.py`
- `conv/src/train_conv.py`
- `conv/src/infer_conv.py`

Code cũ:

```python
use_amp=accelerator.use_fp16
```

Code mới:

```python
use_amp=accelerator.mixed_precision == "fp16"
```

Note: Không đổi logic padding/collator. Đây là đổi API tương thích vì `use_fp16` là interface cũ.

## 3. Guard W&B calls in conv training

File:
- `conv/src/train_conv.py`

Code cũ:

```python
init_wandb_run(...)
wandb_logging(eval_dict=valid_report, step = epoch)
wandb_logging(eval_dict=test_report, step = epoch)
wandb.finish()
```

Code mới:

```python
if args.use_wandb:
    init_wandb_run(...)

if args.use_wandb:
    wandb_logging(eval_dict=valid_report, step = epoch)

if args.use_wandb:
    wandb_logging(eval_dict=test_report, step = epoch)

if args.use_wandb:
    wandb.finish()
```

Note: Không đổi training logic. Thay đổi này chỉ ngăn lỗi login/init W&B khi không bật `--use_wandb`.

## 4. Remove unused fragile Transformers imports

File:
- `conv/src/train_conv.py`

Code cũ:

```python
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModel, AutoTokenizer, RobertaForMaskedLM
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
```

Code mới:

```python
from transformers import AutoModelForCausalLM
```

Note: Không đổi logic. Các symbol bị bỏ đều không được dùng trong file này; `AdamW`, `AutoModel`, `AutoTokenizer` vẫn đã được import ở dòng import chính phía trên.

## 5. `ModelOutput` import compatibility

Files:
- `rec/src/model_gpt2.py`
- `conv/src/model_gpt2.py`

Code cũ:

```python
from transformers.file_utils import ModelOutput
```

Code mới:

```python
try:
    from transformers.utils import ModelOutput
except ImportError:
    from transformers.file_utils import ModelOutput
```

Note: Không đổi logic model. Đây là fallback import để chạy được với transformers mới hơn nhưng vẫn tương thích bản cũ.

## 6. Generation cache argument compatibility

Files:
- `rec/src/model_gpt2.py`
- `conv/src/model_gpt2.py`

Code cũ:

```python
def prepare_inputs_for_generation(
    self, input_ids, past=None, prompt_embeds=None, **kwargs
):
    token_type_ids = kwargs.get("token_type_ids", None)
```

Code mới:

```python
def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, prompt_embeds=None, **kwargs
):
    past = past_key_values if past_key_values is not None else kwargs.get("past", None)
    token_type_ids = kwargs.get("token_type_ids", None)
```

Note: Không đổi công thức generation. Thay đổi này cho phép Hugging Face generation API mới truyền cache bằng tên `past_key_values`, đồng thời vẫn giữ fallback `past` cũ.

## 7. Not changed: optimizer

Files liên quan:
- `rec/src/train_pre_inspired.py`
- `rec/src/train_rec_inspired.py`
- `rec/src/train_pre_redial.py`
- `rec/src/train_rec_redial.py`
- `conv/src/train_conv.py`

Code giữ nguyên:

```python
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
```

Note: Tôi chưa đổi sang `torch.optim.AdamW` vì đây là optimizer của thí nghiệm. Dù `transformers.AdamW` là API cũ/deprecated, việc đổi optimizer implementation/default epsilon có thể làm kết quả lệch. `requirements.txt` đang pin `transformers==4.36.2` để import này vẫn chạy.
