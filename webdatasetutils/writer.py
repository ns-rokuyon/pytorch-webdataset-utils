import json
from pathlib import Path


def save_shards_metadata(shard_dir: Path, **kwargs) -> Path:
    """Save metadata file into tar-shards directory
    """
    if not shard_dir.exists() or not shard_dir.is_dir():
        raise ValueError(f'Invalid shard_dir: {shard_dir}')
    metadata_file = shard_dir / 'metadata.json'
    if metadata_file.exists():
        raise RuntimeError(f'{metadata_file} already exists')
    with open(metadata_file, 'w') as fp:
        fp.write(json.dumps(kwargs, ensure_ascii=False, indent=4))
    return metadata_file