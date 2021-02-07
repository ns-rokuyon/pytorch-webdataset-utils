import torch
import webdataset as wds
import warnings
from typing import List


class DistributedShardSelector:
    """Shard selector of WebDataset in DDP
    """
    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size

    def __call__(self, urls: List[str]) -> List[str]:
        assert not isinstance(urls, str)

        rank, world_size = self.rank, self.world_size
        worker_info = torch.utils.data.get_worker_info()

        n_all_urls = len(urls)

        # Split given urls based on distributed process rank
        urls = urls[rank::world_size]
        n_urls_per_rank = len(urls)

        if worker_info is None:
            num_workers = 1
        else:
            wid = worker_info.id
            num_workers = worker_info.num_workers
            if wid == 0 and n_urls_per_rank < num_workers:
                warnings.warn(f'num_workers {num_workers} > '
                              f'num_shards per rank {n_urls_per_rank}')
            # Worker based splitting
            urls = urls[wid::num_workers]

        n_urls_per_worker = len(urls)

        return urls
