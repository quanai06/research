import argparse
import os
import sys
import time

import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from src.common.modality_config import add_modality_args, modality_config_from_args
from src.common.semantic_graphs import CollaborativeSemanticGraph, ImageSemanticGraph, TextSemanticGraph
from src.conv.retrieval_prompt_config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from src.conv.data.retrieval_prompt_dataset import CRSConvDataCollator, CRSConvDataset
from src.conv.data.kg_resources import DBpedia
from src.conv.eval.evaluate_conv import ConvEvaluator
from src.conv.models.retrieval_prompt_encoder import RetrievalConversationPromptEncoder
from src.conv.models.mscrs_conv_model import MSCRSConvModel
from src.conv.utils import load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, default='inspired', help="A file containing all data.")
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_max_length', type=int, default=200, help="max length of both encoder and decoder input.")
    parser.add_argument('--resp_max_length', type=int, default=80, help="max length of decoder input.")
    parser.add_argument("--entity_max_length", type=int, default=64, help="max entity length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=50)
    parser.add_argument("--tokenizer", type=str, default='microsoft/DialoGPT-small')
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    parser.add_argument("--text_tokenizer", type=str, default='roberta-base')
    # model
    parser.add_argument("--model", type=str, default='microsoft/DialoGPT-small')
    parser.add_argument("--gen_model_checkpoint", type=str, help="Optional conv gen_model checkpoint directory saved by train_conv.py.")
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--text_encoder", type=str, default='roberta-base')
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--n_prefix_conv", type=int, default=110)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN")
    parser.add_argument("--n_examples", type=int, default=3, help="number of retrieved demonstrations")
    parser.add_argument('--mapping', action='store_true', help='if we use semantic mapping')
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    add_modality_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    co = CollaborativeSemanticGraph(
        dataset=args.dataset,
        split='train',
        debug=args.debug,
        all_items=kg['item_ids'],
        entity_max_length=args.entity_max_length,
        n_entity=kg['num_entities'],
    ).get_entity_co_info()
    text_simi = TextSemanticGraph(pad_entity_id=kg['pad_entity_id'], dataset=args.dataset, top_k=20).get_entity_ts_info()
    image_simi = ImageSemanticGraph(pad_entity_id=kg['pad_entity_id'], dataset=args.dataset, top_k=20).get_entity_is_info()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gen_model_checkpoint is not None:
        model = load(model, args.gen_model_checkpoint)
    model = model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    prompt_encoder = RetrievalConversationPromptEncoder(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        edge_index_c=co['edge_index_c'], edge_index_i_s=image_simi['edge_index_i_s'],
        edge_index_t_s=text_simi['edge_index_t_s'], idx_to_id=text_simi['idx_to_id'],
        n_prefix_rec=args.n_prefix_conv,
        prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples,
        modality_config=modality_config_from_args(args, add_item_semantic_to_entities=True),
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    # data
    dataset = CRSConvDataset(
        args.dataset, args.split, tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples
    )
    data_collator_generator = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, gen=True, use_amp=accelerator.mixed_precision == "fp16", debug=args.debug,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer,
        n_examples=args.n_examples,
        prompt_max_length=args.prompt_max_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )
    model, prompt_encoder, dataloader = accelerator.prepare(model, prompt_encoder, dataloader)
    conv_model = MSCRSConvModel(
        model=model,
        text_encoder=text_encoder,
        prompt_encoder=prompt_encoder,
        n_examples=args.n_examples,
        prompt_max_length=args.prompt_max_length,
        mapping=args.mapping,
    )
    gen_dir = os.path.join('save', args.dataset)
    os.makedirs(gen_dir, exist_ok=True)
    model_name = args.prompt_encoder.split('/')[-2] if args.prompt_encoder is not None else args.model.replace('/', '_')
    gen_file_path = os.path.join(gen_dir, f'{model_name}_{args.split}.jsonl')
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
    conv_model.eval()

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            gen_seqs = conv_model.generate_from_batch(
                batch,
                generation_model=accelerator.unwrap_model(model),
                max_new_tokens=args.max_gen_len,
            )
            gen_resp_ids = []
            for gen_seq, length in zip(gen_seqs, batch['context_len']):
                gen_seq = [int(token_id) for token_id in gen_seq if int(token_id) != tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq)
            evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)

    # metric
    accelerator.wait_for_everyone()
    report = evaluator.report()
    test_report = {}
    for k, v in report.items():
        test_report[f'{args.split}/{k}'] = v
    logger.info(test_report)
    if run:
        run.log(test_report)
