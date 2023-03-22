import collections
import transformers

from transformers import get_linear_schedule_with_warmup

from apex.optimizers import FusedAdam
from torch.optim import AdamW
import torch.distributed as dist

import torch
import os
import sys

sys.path.append(os.getcwd())
from model.deberta_v2.modeling_deberta_v2 import DebertaV2ForMaskedLM
from model.bert.bert_modeling import BertForMaskedLM, BertModel
import torch.nn as nn

from collections import OrderedDict

__all__ = [
    "get_model",
    "get_optimizer",
    "get_lr_scheduler",
    "get_dataloader_for_pretraining",
]


def get_new_state_dict(state_dict, start_index=13):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[start_index:]
        new_state_dict[name] = v
    return new_state_dict


class LMModel(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()

        self.checkpoint = args.checkpoint_activations
        self.config = config
        self.model = model
        if self.checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Only return lm_logits
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )


def get_model(args, mlm_model_type, load_pretrain_model, model_config, logger, dtr=False):

    if mlm_model_type == "bert":
        config = transformers.BertConfig.from_json_file(model_config)
        # setattr(config, "output_attentions", True)
        model = BertForMaskedLM(config, dtr)
    elif mlm_model_type == "deberta_v2":
        config = transformers.DebertaV2Config.from_json_file(model_config)
        model = DebertaV2ForMaskedLM(config)
    else:
        raise Exception("Invalid mlm!")

    if len(load_pretrain_model) > 0:
        assert os.path.exists(load_pretrain_model)
        # load_checkpoint(args.load_pretrain_model, model, strict=False)
        m_state_dict = torch.load(
            load_pretrain_model,
            map_location=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        new_state_dict = collections.OrderedDict()
        for k, v in m_state_dict.items():
            # if "module." in k:
            #     k = k.replace("module.", "")
            # if "task_pooler" in k:
            #     continue
            # if "classifier" in k:
            #     continue
            # if "gamma" in k:
            #     k = k.replace("gamma", "weight")
            # elif "beta" in k:
            #     k = k.replace("beta", "bias")
            if k.startswith("module."):
                k = k.replace("module.", "")
            new_state_dict[k] = v
        # new_state_dict = get_new_state_dict(m_state_dict)
        model.load_state_dict(
            new_state_dict, strict=True
        )  # must insure that every process have identical parameters !!!!!!!
        logger.info("load model success")

    numel = sum([p.numel() for p in model.parameters()])
    if args.checkpoint_activations:
        logger.info("use gradient checkpointing!")
        model.gradient_checkpointing_enable()

    return config, model, numel


def get_optimizer(model, lr):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta", "LayerNorm"]

    # configure the weight decay for bert models
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.1,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=lr, betas=[0.9, 0.999])
    return optimizer


def get_lr_scheduler(optimizer, total_steps, warmup_steps=4000, last_epoch=-1):
    # warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        last_epoch=last_epoch,
    )
    # lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    return lr_scheduler


def save_ckpt(model, optimizer, lr_scheduler, path, epoch, shard, global_step):
    model_path = path + "_pytorch_model.bin"
    optimizer_lr_path = path + ".op_lrs"
    checkpoint = {}
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["shard"] = shard
    checkpoint["global_step"] = global_step
    model_state = model.state_dict()  # each process must run model.state_dict()
    if dist.get_rank() == 0:
        torch.save(checkpoint, optimizer_lr_path)
        torch.save(model_state, model_path)
