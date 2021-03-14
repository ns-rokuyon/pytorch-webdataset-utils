import torch
import random
import webdataset as wds
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Set


@dataclass
class DistributedShardInfo:
    unavailable_urls: Set[str]
    use_size_in_cluster: int
    use_size_in_dataloader: int
    n_urls_per_rank: int
    n_urls_per_worker: int



class DistributedShardSelector:
    """Shard selector of WebDataset in DDP

    Parameters
    ----------
    rank : int
        Rank ID in distributed training

    world_size : int
        Cluster size of distributed training

    shuffle : bool
        If true, first, given url list will be shuffled 

    callback : Optional[Callable[[DistributedShardInfo], None]]
        Callback function to get splitted shard results
    """
    def __init__(
        self,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        callback: Optional[Callable[[DistributedShardInfo], None]] = None
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.callback = callback

    def __call__(self, urls: List[str]) -> List[str]:
        assert not isinstance(urls, str)

        rank, world_size = self.rank, self.world_size
        worker_info = torch.utils.data.get_worker_info()

        urls = urls.copy()
        n_all_urls = len(urls)
        if self.shuffle:
            random.shuffle(urls)

        unavailable_urls = set()

        # Normalize number of urls to distribute uniformly for each rank
        use_size_in_cluster = n_all_urls - (n_all_urls % world_size)
        unavailable_urls.add(
            set(urls[use_size_in_cluster:])
        )
        urls = urls[:use_size_in_cluster]

        # Split given urls based on distributed process rank
        urls = urls[rank::world_size]
        n_urls_per_rank = len(urls)

        if worker_info is None:
            num_workers = 1
            use_size_in_dataloader = n_urls_per_rank
        else:
            wid = worker_info.id
            num_workers = worker_info.num_workers
            if wid == 0 and n_urls_per_rank < num_workers:
                warnings.warn(f'num_workers {num_workers} > '
                              f'num_shards per rank {n_urls_per_rank}')

            # Normalize number of urls to distribute uniformly
            # for each dataloader's worker
            use_size_in_dataloader = n_urls_per_rank - (
                n_urls_per_rank % num_workers
            )
            urls = urls[:use_size_in_dataloader]
            unavailable_urls.add(
                set(urls[use_size_in_dataloader:])
            )

            # Worker based splitting
            urls = urls[wid::num_workers]

        n_urls_per_worker = len(urls)

        if self.callback:
            self.callback(
                DistributedShardInfo(
                    unavailable_urls=unavailable_urls,
                    use_size_in_cluster=use_size_in_cluster,
                    use_size_in_dataloader=use_size_in_dataloader,
                    n_urls_per_rank=n_urls_per_rank,
                    n_urls_per_worker=n_urls_per_worker
                )
            )

        return urls


DistributedShardSplitter = DistributedShardSelector
