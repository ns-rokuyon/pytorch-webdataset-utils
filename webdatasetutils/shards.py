import json
import webdataset as wds
from typing import Optional, Union, Callable
from logging import Logger
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from .writer import save_shards_metadata
from .util import get_image_extension, NumpyJSONEncoder


def make(
    dataset: Dataset,
    output_dir: Union[Path, str],
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None,
    tarfilename_pattern: str = '%06d.tar',
    progress_bar: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    **kwargs
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise RuntimeError(f'{output_dir} already exists')
    output_dir.mkdir(parents=True)

    collate_fn = collate_fn or (lambda x: x)
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        collate_fn=collate_fn)
    if progress_bar is not None:
        loader = progress_bar(loader)
    
    num_source_data = len(dataset)
    sample_count = 0
    sample_structures = set()
    labels = set()

    tarfile_pattern = str(output_dir / tarfilename_pattern)
    with wds.ShardWriter(tarfile_pattern, **kwargs) as sink:
        for i, batch in enumerate(loader):
            # Get a sample-dict that dataset constructs
            sample = batch[0]

            if '__error__' in sample:
                err = sample['__error__']
                if logger:
                    logger.warning(
                        f'Dataset failed to construct sample '
                        f'due to following error: {err}'
                    )
                continue

            if '__key__' not in sample:
                sample['__key__'] = f'{i:012}'

            # Write the sample to the sharded tar archives
            sink.write(sample)

            # Stats for metadata
            sample_count += 1
            sample_structures.add(tuple(sample.keys()))
            if 'cls' in sample:
                labels.add(sample['cls'])
    
    save_shards_metadata(
        output_dir,
        num_data=sample_count,
        num_errors=num_source_data - sample_count,
        num_classes=len(labels),
        structures=list(sample_structures),
        shuffled=shuffle,
    )
    return output_dir


class _TarRecordDatasetWrapper:
    """Base adapter class between torch.utils.data.Dataset and 
       interface of wds.ShardWriter

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object which has the signature below
        >>> def __getitem__(self, i: int) -> Union[bytes, Tuple[bytes, ...]]
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict:
        try:
            sample = self.get_record(i)
        except Exception as e:
            return {'__error__': e}
        return sample

    def get_record(self, i: int) -> dict:
        raise NotImplementedError


class ImageOnlyTarRecord(_TarRecordDatasetWrapper):
    def get_record(self, i: int) -> dict:
        img = self.dataset[i]
        ext = get_image_extension(img)
        if not ext:
            raise RuntimeError(f'Invalid image type: {self.dataset}[{i}]')
        return {ext: img}


class ImageLabelPairTarRecord(_TarRecordDatasetWrapper):
    def get_record(self, i: int) -> dict:
        img, label = self.dataset[i]
        ext = get_image_extension(img)
        if not ext:
            raise RuntimeError(f'Invalid image type: {self.dataset}[{i}]')
        return {
            ext: img,
            'cls': label
        }


class ImageMetadataPairTarRecord(_TarRecordDatasetWrapper):
    def get_record(self, i: int,
                   label_key: str = 'cls') -> dict:
        img, meta = self.dataset[i]
        assert isinstance(meta, dict)
        ext = get_image_extension(img)
        if not ext:
            raise RuntimeError(f'Invalid image type: {self.dataset}[{i}]')
        sample = {
            ext: img,
            'json': json.dumps(meta, ensure_ascii=False, cls=NumpyJSONEncoder)
        }
        if label_key in meta:
            sample['cls'] = meta[label_key]
        return sample
