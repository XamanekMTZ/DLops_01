from src.dm import *
from src.module import *
from pathlib import Path
import pytorch_lightning as pl

path = Path('dataset')
dm = MNISTDataModule( path, batch_size = 25 )
module = MNISTModule()
trainer = pl.Trainer(
    max_epochs = 10, 
    logger = None, 
    enable_checkpointing = False, 
    # overfit_batches = 1,
)

trainer.fit ( module, dm )