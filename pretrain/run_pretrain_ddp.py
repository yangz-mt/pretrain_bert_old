import math
import torch
print(f"torch.__version__ = {torch.__version__}")
from arguments import parse_args
from pretrain_utils import get_model, get_optimizer, get_lr_scheduler, save_ckpt
from utils.exp_util import get_tflops, get_mem_info, throughput_calculator, log_args
from utils.global_vars import set_global_variables, get_timers, get_tensorboard_writer
from utils.logger import Logger
from evaluation_ddp import evaluate

from nvidia_bert_dataset_provider import NvidiaBertDatasetProvider
from tqdm import tqdm
import os
import time
from functools import partial
from itertools import chain


from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from dist_utils import reduce_value, distributed_concat_with_all_gather


def main():

    args = parse_args()
    # init the distributed backend
    print("local_rank: ", args.local_rank)
    if args.local_rank == -1:
        raise ValueError("should use local")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # torch.distributed.init_process_group(backend="nccl")

    launch_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    logger = Logger(
        os.path.join(args.log_path, launch_time),
        local_rank=args.local_rank,
        cuda=torch.cuda.is_available(),
        debug=args.vscode_debug,
    )

    log_args(logger, args)
    args.tokenizer = tokenizer
    args.logger = logger
    set_global_variables(launch_time, args.tensorboard_path)

    # world_size = torch.distributed.get_world_size()
    world_size = 1
    # build model, optimizer and criterion
    config, model, model_numel = get_model(
        args,
        mlm_model_type=args.mlm_model_type,
        load_pretrain_model=args.load_pretrain_model,
        model_config=args.bert_config,
        logger=logger,
        dtr=args.dtr
    )
    logger.info(f"Model numel: {model_numel}")


    # ddp
    model.to(device)
    # if args.local_rank != -1:
    #     model = DDP(
    #         model,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         find_unused_parameters=False,
    #     )

    # if torch.distributed.get_rank() == 0:
    os.mkdir(os.path.join(args.ckpt_path, launch_time))

    get_tflops_func = partial(
        get_tflops,
        model_numel,
        args.train_micro_batch_size_per_gpu,
        args.max_seq_length,
    )
    steps_per_epoch = (
        144003367
        // world_size
        // args.train_micro_batch_size_per_gpu
        // args.gradient_accumulation_steps
        // args.refresh_bucket_size
    )  # len(dataloader)
    total_steps = steps_per_epoch * args.epoch
    # total_steps = 1000000  # follow minilm paper

    # build optimizer and lr_scheduler
    start_epoch = 0
    start_shard = 0
    global_step = 0
    if args.resume_train:
        assert os.path.exists(args.load_optimizer_lr)
        o_l_state_dict = torch.load(args.load_optimizer_lr, map_location="cpu")
        o_l_state_dict["lr_scheduler"]["last_epoch"] = (
            o_l_state_dict["lr_scheduler"]["last_epoch"] - 1
        )
        optimizer = get_optimizer(model, lr=args.lr)
        optimizer.load_state_dict(o_l_state_dict["optimizer"])
        lr_scheduler = get_lr_scheduler(
            optimizer,
            total_steps=total_steps,
            last_epoch=o_l_state_dict["lr_scheduler"]["last_epoch"],
        )  # o_l_state_dict['lr_scheduler']['last_epoch']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(f"cuda:{torch.cuda.current_device()}")
        # if you want delete the above three code, have to move the model to gpu, because in optimizer.step()
        lr_scheduler.load_state_dict(o_l_state_dict["lr_scheduler"])

        start_epoch = o_l_state_dict["epoch"]
        start_shard = o_l_state_dict["shard"] + 1
        # global_step = o_l_state_dict['global_step'] + 1
        logger.info(
            f"resume from epoch {start_epoch} shard {start_shard} step {lr_scheduler.last_epoch} lr {lr_scheduler.get_last_lr()[0]}"
        )
    else:
        optimizer = get_optimizer(model, lr=args.lr)
        lr_scheduler = get_lr_scheduler(
            optimizer, total_steps=total_steps, last_epoch=-1
        )

    # optimizer = gpc.config.optimizer.pop('type')(
    # model.parameters(), **gpc.config.optimizer)
    # optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)

    # build dataloader
    pretrain_dataset_provider = NvidiaBertDatasetProvider(args)

    # engine, _, _, lr_scheduelr = colossalai.initialize(
    #     model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=lr_scheduler
    # )

    logger.info(get_mem_info(prefix="After init model, "))

    best_loss = None
    train_loss = 0
    eval_loss = 0
    local_step = 0

    train_rel_loss = 0
    train_attn_loss = 0
    train_mlm_loss = 0
    eval_mlm_loss = 0
    eval_kd_loss = 0

    timers = get_timers()
    timers("interval_time").start()
    timers("epoch_time").start()
    timers("shard_time").start()


    model.eval()
    # if args.dtr:
    #     # import pdb;pdb.set_trace()
    #     torch.set_memory_budget(8 * 1024 * 1024 * 1024)  # 8 GiB
    for epoch in range(start_epoch, args.epoch):

        for shard in range(start_shard, len(os.listdir(args.data_path_prefix))):

            dataset_iterator, total_length = pretrain_dataset_provider.get_shard(shard)
            # dataset_iterator.sampler.set_epoch(epoch)
            # pretrain_dataset_provider.prefetch_shard(shard + 1) # may cause cpu memory overload
            # if torch.distributed.get_rank() == 0:
            iterator_data = tqdm(
                enumerate(dataset_iterator),
                total=(
                    total_length
                    // args.train_micro_batch_size_per_gpu
                    // world_size
                ),
                colour="cyan",
                smoothing=1,
            )
            # else:
            #     iterator_data = enumerate(dataset_iterator)

            model.train()
            if args.dtr:
                model._apply(lambda t: t.detach().checkpoint())
            for step, batch_data in iterator_data:
                # batch_data = pretrain_dataset_provider.get_batch(batch_index)
                input_ids = batch_data[0].cuda(f"cuda:{torch.cuda.current_device()}")
                attention_mask = batch_data[1].cuda(
                    f"cuda:{torch.cuda.current_device()}"
                )
                token_type_ids = batch_data[2].cuda(
                    f"cuda:{torch.cuda.current_device()}"
                )
                mlm_label = batch_data[3].cuda(f"cuda:{torch.cuda.current_device()}")
                # nsp_label = batch_data[5].cuda()
                if args.dtr:
                    # input_ids.detach().checkpoint()
                    attention_mask.detach().checkpoint()
                    # token_type_ids.detach().checkpoint()
                    mlm_label.detach().checkpoint()
                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=mlm_label,
                )
                loss = outputs.loss

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # backward
                loss.backward()

                # if args.dtr:
                #     loss.decheckpoint()

                pretrain_dataset_provider.prefetch_batch()

                local_step += 1

                # cur_all_loss = reduce_value(loss)

                # train_loss += cur_all_loss
                train_loss = loss

                if local_step % args.gradient_accumulation_steps == 0:
                    # clip gradient
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (
                        global_step % args.log_interval == 0
                        and global_step != 0
                        # and torch.distributed.get_rank() == 0
                    ):
                        elapsed_time = timers("interval_time").elapsed(reset=False)
                        elapsed_time_per_iteration = elapsed_time / global_step
                        (
                            samples_per_sec,
                            tflops,
                            approx_parameters_in_billions,
                        ) = throughput_calculator(
                            model_numel,
                            args,
                            config,
                            elapsed_time,
                            global_step,
                            world_size,
                        )

                        cur_loss = train_loss / args.log_interval
                        current_lr = lr_scheduler.get_last_lr()[0]
                        ppl = math.exp(cur_loss)
                        log_str = (
                            f"| epoch: {epoch} | shard: {shard} | step: {global_step} | lr {current_lr:.7f} | elapsed_time: {elapsed_time / 60 :.3f} minutes "
                            + f"| mins/batch: {elapsed_time_per_iteration :.3f} seconds | loss: {cur_loss:.7f} | ppl:{ppl:.7f}  | TFLOPS: {get_tflops_func(elapsed_time_per_iteration):.3f} or {tflops:.3f}"
                        )  # TODO(补充有效日志)
                        logger.info(log_str, print_=False)

                        if args.wandb:
                            tensorboard_log = get_tensorboard_writer()
                            tensorboard_log.log_train(
                                {
                                    "lr": current_lr,
                                    "loss": cur_loss,
                                    "ppl": ppl,
                                    "mins_batch": elapsed_time_per_iteration,
                                },
                                # "ppl": math.exp(train_mlm_loss / args.log_interval),
                                global_step,
                            )
                        train_loss = 0
            logger.info(
                f'epoch {epoch} shard {shard} has cost {timers("shard_time").elapsed() / 60 :.3f} mins'
            )
            logger.info("*" * 100)
            cur_eval_loss = evaluate(
                model, args, logger, global_step
            )
            eval_loss += cur_eval_loss
            # eval_mlm_loss += cur_eval_mlm_loss
            # eval_kd_loss += cur_eval_kd_loss
            save_ckpt(
                model,
                optimizer,
                lr_scheduler,
                os.path.join(
                    args.ckpt_path,
                    launch_time,
                    f"epoch-{epoch}_shard-{shard}_" + launch_time,
                ),
                epoch,
                shard,
                global_step,
            )

        eval_loss /= len(os.listdir(args.data_path_prefix))
        logger.info(
            f'epoch {epoch} | shard_length {len(os.listdir(args.data_path_prefix))} | elapsed_time: {timers("epoch_time").elapsed() / 60 :.3f} mins'
            + f"eval_loss: {eval_loss}"
        )
        logger.info("-" * 100)
        if args.wandb:# and torch.distributed.get_rank() == 0:
            tensorboard_log = get_tensorboard_writer()
            tensorboard_log.log_eval(
                {
                    "all_eval_shard_loss": eval_loss,
                },
                epoch,
            )
        start_shard = 0
        eval_loss = 0

    pretrain_dataset_provider.release_shard()

    logger.info("Congratulation, training has finished!!!")


if __name__ == "__main__":
    main()
