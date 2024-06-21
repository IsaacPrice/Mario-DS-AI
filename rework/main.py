from pytorch_lightning import Trainer
from deepQlearning import DeepQLearning


algo = DeepQLearning(
    lr=0.001,
    a_last_episode=1000,
    b_last_episode=1000,
    sigma=.5,
    batch_size=64,
)

trainer = Trainer(
    max_epochs=1000, 
)

trainer.fit(algo)