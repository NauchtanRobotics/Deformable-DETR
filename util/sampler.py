import torch

class SkipableRandomSampler(torch.utils.data.RandomSampler):

    def __init__(self, data_source, skip=0):
        self.data_source = data_source
        self.skip = skip

    def __iter__(self):
        it = super().__iter__()
        for _ in range(self.skip):
            next(it)

        return it

    def __len__(self):
        return len(self.data_source) - self.skip
