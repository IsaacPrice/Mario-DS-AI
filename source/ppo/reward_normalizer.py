import numpy as np


class RewardNormalizer:
    def __init__(self, epsilon: float = 1e-8):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon
        self.epsilon: float = epsilon

    def update(self, rewards: np.ndarray) -> None:
        batch_mean: float = np.mean(rewards)
        batch_var: float = np.var(rewards)
        batch_count: int = len(rewards)

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int) -> None:
        delta: float = batch_mean - self.mean
        total_count: float = self.count + batch_count

        new_mean: float = self.mean + delta * batch_count / total_count
        m_a: float = self.var * self.count
        m_b: float = batch_var * batch_count
        M2: float = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var: float = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        return (rewards - self.mean) / (np.sqrt(self.var) + self.epsilon)

    def reset(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = self.epsilon
