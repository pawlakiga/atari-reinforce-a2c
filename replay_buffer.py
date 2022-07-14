import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Class to represent replay buffer and sample from it
    """

    def __init__(self, max_length) -> None:
        self.buffer = deque(maxlen=max_length)

    def add_experience(self, experience : tuple):
        self.buffer.append(experience)

    def get_random_experience_batch(self, batch_size):
        """
        function to sample a random experience batch from buffer
        Arguments:
        @ batch_size - number of desired samples in batch
        """
        sample = []
        for i in range(batch_size):
            sample.append(random.choice(self.buffer))
        return [np.array([experience[field_index] for experience in sample]) for field_index in range(len(sample[0]))]

    def clear(self):
        self.buffer.clear()

    @property
    def len(self):
        return len(self.buffer)

