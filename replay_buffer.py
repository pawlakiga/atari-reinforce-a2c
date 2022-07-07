import random
from collections import deque
import numpy as np

class ReplayBuffer:

    def __init__(self, max_length) -> None:
        self.buffer = deque(maxlen=max_length)

    def add_experience(self, experience : tuple):
        self.buffer.append(experience)

    def get_random_experience_batch(self, batch_size):
        sample = []
        for i in range(batch_size):
            sample.append(random.choice(self.buffer))
        return [np.array([experience[field_index] for experience in sample]) for field_index in range(len(sample[0]))]

    def get_ordered_experience_batch(self, batch_size):
        sample = []
        for i in range(batch_size):
            sample.append(self.buffer[i])
        return [np.array([experience[field_index] for experience in sample]) for field_index in range(len(sample[0]))]

    def clear(self):
        self.buffer.clear()

    # def get_single_experience(self):
    #     experience = self.buffer.popleft()
    #     return [np.array([experience[field_index] for field_index in range(len(experience))])]

    @property
    def len(self):
        return len(self.buffer)

