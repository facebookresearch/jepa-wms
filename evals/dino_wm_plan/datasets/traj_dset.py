import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Sequence, List
from torch.utils.data import Dataset, Subset
from torch import default_generator, randperm
from einops import rearrange
import abc

# https://github.com/JaidedAI/EasyOCR/issues/1243
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class TrajDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

class TrajSubset(Subset, TrajDataset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: TrajDataset, indices: Sequence[int]):
        super(TrajSubset, self).__init__(dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def __getattr__(self, name):
        try:
            return super(TrajSubset, self).__getattr__(name)
        except AttributeError:
            return getattr(self.dataset, name)

    def __getstate__(self):
        return self.dataset, self.indices

    def __setstate__(self, state):
        self.dataset, self.indices = state

class TrajSlicerDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        num_frames: int,
        frameskip: int = 1,
        process_actions: str = "concat",
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.slices = []
        for i in range(len(self.dataset)): 
            T = self.dataset.get_seq_length(i)
            if T - num_frames < 0:
                print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
            else:
                self.slices += [
                    (i, start, start + num_frames * self.frameskip)
                    for start in range(T - num_frames * frameskip + 1)
                ]  # slice indices follow convention [start, end)
        # randomly permute the slices
        self.slices = np.random.permutation(self.slices)
        
        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim


    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, state, _ = self.dataset[i]
        for k, v in obs.items():
            obs[k] = v[start:end:self.frameskip]
        state = state[start:end:self.frameskip]
        act = act[start:end]
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)  # concat actions
        return tuple([obs, act, state])


def random_split_traj(
    dataset: TrajDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
    traj_subset: bool = True,
) -> List[TrajSubset]:
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    print(
        [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
    )
    if traj_subset:
        return [
            TrajSubset(dataset, indices[offset - length : offset])
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
    # else:
    #     def create_sliced_dataset(original_dataset, indices):
    #         new_dataset = type(original_dataset)(
    #             data_path= original_dataset.data_path.parent if type(original_dataset).__name__ == "DeformDataset" else original_dataset.data_path,
    #             n_rollout=len(indices),
    #             transform=original_dataset.transform,
    #             normalize_action=original_dataset.normalize_action,
    #             action_scale=1.0,
    #             object_name=original_dataset.data_path.parts[-1],
    #         )
    #         # Slice all tensor attributes
    #         for attr_name in dir(original_dataset):
    #             attr_value = getattr(original_dataset, attr_name)
    #             if isinstance(attr_value, torch.Tensor) and attr_value.size(0) == len(original_dataset):
    #                 setattr(new_dataset, attr_name, attr_value[indices])
    #         return new_dataset

    #     split_indices = [
    #         indices[offset - length : offset]
    #         for offset, length in zip(_accumulate(lengths), lengths)
    #     ]
    #     train_indices, val_indices = split_indices
    #     train_dataset = create_sliced_dataset(dataset, train_indices)
    #     val_dataset = create_sliced_dataset(dataset, val_indices)
    #     return [train_dataset, val_dataset]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42, traj_subset=True):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed), traj_subset=traj_subset,
    )
    return train_set, val_set


def get_train_val_sliced(
    traj_dataset: TrajDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
    traj_subset: bool = True,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
        traj_subset=traj_subset,
    )
    train_slices = TrajSlicerDataset(train, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val, num_frames, frameskip)
    return train, val, train_slices, val_slices