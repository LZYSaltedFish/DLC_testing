from torch.utils.tensorboard import SummaryWriter
import random
import fire
from tqdm import tqdm

def main(
    log_dir='/root/data/tmp_dlc/log'
):
    tensorboard = SummaryWriter(
        log_dir=log_dir
    )

    loss = 3.13
    for step in tqdm(range(1000)):
        loss += random.random()-0.7
        tensorboard.add_scalar(tag='train/loss',
                            scalar_value=loss,
                            global_step=step)
    
if __name__ == '__main__':
    fire.Fire(main)