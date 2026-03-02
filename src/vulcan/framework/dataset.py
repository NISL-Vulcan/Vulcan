import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch.utils.data import DataLoader as TorchDataLoader
from torch import distributed as dist

from importlib import import_module
from typing import List, Dict, Tuple, Any

from vulcan.framework.preprocess import get_preprocess


_DATASET_LOADERS: Dict[str, Tuple[str, str]] = {
    # Graph datasets
    "ReGVD": ("vulcan.framework.datasets.regvd", "ReGVD"),
    "Devign_Partial": ("vulcan.framework.datasets.devign_partial", "Devign_Partial"),
    "CodeXGLUE": ("vulcan.framework.datasets.CodeXGLUE", "CodeXGLUE"),
    "DWK_Dataset": ("vulcan.framework.datasets.XFGDataset_build", "DWK_Dataset"),
    "IVDDataset": ("vulcan.framework.datasets.IVDetect.IVDetectDataset", "IVDetectDataset"),
    "LineVul": ("vulcan.framework.datasets.linevul", "LineVul"),
    "vdet_data": ("vulcan.framework.datasets.vdet_data", "vdet_data"),
    # Sequence/vector datasets
    "VDdata": ("vulcan.framework.datasets.vddata", "VDdata"),
    "test_source": ("vulcan.framework.datasets.test_source", "test_source"),
    # AST decomposition datasets
    "TrVD": ("vulcan.framework.datasets.trvd_dataset", "TrVDDataset"),
}


def get_dataset(config: Dict[str, Any], split: str):
    dataset_cfg = config["DATASET"]
    train_cfg, eval_cfg = config["TRAIN"], config["EVAL"]

    dataset_name = dataset_cfg["NAME"]
    dataset_param = dataset_cfg.get("PARAMS") or {}

    if dataset_name not in _DATASET_LOADERS:
        raise ValueError(f"The dataset name {dataset_name} does not exist. Available: {sorted(_DATASET_LOADERS.keys())}")

    # preprocess dataset
    preprocess_cfg = dataset_cfg["PREPROCESS"]
    preprocess_format = None
    if preprocess_cfg["ENABLE"]:
        preprocess_compose = preprocess_cfg["COMPOSE"]
        print("preprocess compose:")
        print(preprocess_compose)

        if split == "train":
            preprocess_format = get_preprocess(train_cfg["INPUT_SIZE"], preprocess_compose)
        elif split == "val":
            preprocess_format = get_preprocess(eval_cfg["INPUT_SIZE"], preprocess_compose)
        else:
            print(split)
            print("Invalid split specified")

    module_name, cls_name = _DATASET_LOADERS[dataset_name]
    module = import_module(module_name)
    cls = getattr(module, cls_name)

    dataset = cls(
        split=split,
        root=dataset_cfg["ROOT"],
        preprocess_format=preprocess_format,
        **dataset_param,
    )
    return dataset
'''
from vulcan.framework.datasets.XFGDataset_build import XFGSample,XFGBatch
def graph_collate_fn(samples: List[XFGSample]) -> XFGBatch:
    
    """
    Collate function for DataLoader.
    Args:
        samples (List[XFGSample]): List of XFGSamples
    Returns:
        XFGBatch: An object representing a batch of graphs and labels.
    """
    data = XFGBatch(XFGs=samples)
    return data.graphs, data.labels#XFGBatch(XFGs=samples)


'''
def graph_collate_fn(batch):
    try:
        from torch_geometric.data import Data
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Current config uses the geometric dataloader, but torch_geometric is not installed. "
            "Please install PyG (torch-geometric/torch-scatter/torch-sparse/torch-cluster) first."
        ) from e

    # batch is a list of items returned by dataset.__getitem__
    # e.g. [(input_x1, label1), (input_x2, label2), ...]
    #print('print graph collate batch')
    #print(batch[0])
    #(Data(edge_index=[2, 172], x=[205, 101], y=[1]), tensor(1))
    # Split inputs and labels
    try:
        input_xs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        #print(labels,type(labels))
    except:
        print('Error in graph_collate_fn')
        for item in batch:
            print(item)
    # We cannot stack input_xs directly because edge_index sizes differ.
    # Keep them as a list.
    data_list = []
    # try:
    for input_x in input_xs:
        edge_index, *remaining_data = input_x
        # print(edge_index,type(edge_index))
        # print(remaining_data, type(remaining_data))
        # Check whether edge_index is torch.Tensor; convert if needed.
        #if not isinstance(edge_index, torch.Tensor):
        #    edge_index = torch.tensor(edge_index, dtype=torch.long)
        # Build a dict where edge_index and remaining payload are separated.
        data_dict = {"edge_index": edge_index[1], "my_data": remaining_data}

        # Convert dict to Data object.
        data_object = Data(**data_dict)
        data_list.append(data_object)
        #data_list.append(Data(my_data=data, edge_index=edge_index))
    # except:
    #     print(input_xs)

    # labels is a simple tensor list; stack directly.
    labels = torch.stack(labels, dim=0)
    labels = labels.squeeze()

    return data_list, labels


def trvd_collate_fn(batch):
    """Collate function for TrVD dataset.

    Each sample is ``(subtree_list, label)`` where *subtree_list* is a
    variable-length list of nested index lists.  We simply group them
    and stack labels.
    """
    inputs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    return inputs, labels


def get_dataloader(config, split, dataset, batch_size, num_workers=1, **kw):
    if 'dataloader' in config and config['dataloader'] == 'geometric':
        if split == 'train':
            return TorchDataLoader(dataset, collate_fn=graph_collate_fn, batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=False, sampler=kw['sampler'])
        return TorchDataLoader(dataset, collate_fn=graph_collate_fn, batch_size=1, num_workers=1, pin_memory=True)

    if 'dataloader' in config and config['dataloader'] == 'trvd':
        if split == 'train':
            return TorchDataLoader(dataset, collate_fn=trvd_collate_fn, batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=False, sampler=kw['sampler'])
        return TorchDataLoader(dataset, collate_fn=trvd_collate_fn, batch_size=1, num_workers=1, pin_memory=True)

    #sequence dataset
    if split == 'train':
        return TorchDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=False, sampler=kw['sampler'])
    return TorchDataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)
