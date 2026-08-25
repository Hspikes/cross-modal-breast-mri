import torch
from torch.utils.data import Sampler


class trainSampler(Sampler):
    def __init__(self, indices, num_sampler):
        self.indices = indices
        self.repeat = num_sampler
        self.g = torch.Generator()

    def __iter__(self):
        for _ in range(self.repeat):
            yield self.indices[
                torch.randint(len(self.indices), (1,), generator=self.g).item()
            ]

    def __len__(self):
        return self.repeat
