from torch.utils.tensorboard import SummaryWriter
import random

tensorboard = SummaryWriter(
    log_dir='./log'
)

loss = 3.13
for step in range(1000):
    loss += random.random()-0.7
    tensorboard.add_scalar(tag='train/loss',
                        scalar_value=loss,
                        global_step=step)