import torch
import torch.distributed as dist

from typing import Any, Dict, Iterator, List, Optional, Union


def distributed_concat_with_all_gather(
    tensor: Any, num_total_examples: Optional[int] = None
) -> Any:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(
                distributed_concat_with_all_gather(t, num_total_examples)
                for t in tensor
            )
        output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensors, tensor)
        if len(tensor.size()) == 0:
            concat = torch.stack(output_tensors, dim=0)
        else:
            concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def reduce_value(value, ReduceOp=dist.ReduceOp.SUM):
    value_tensor = value.clone()
    dist.all_reduce(value_tensor, ReduceOp, async_op=False)
    world_size = dist.get_world_size()
    value = value_tensor.item() / world_size
    del value_tensor
    return value
