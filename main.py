# from src.dm import *
# from src.module import *
from src import *
from pathlib import Path
import pytorch_lightning as pl
import yaml
import sys
import importlib
from pytorch_lightning.loggers import TensorBoardLogger

# path = Path('dataset')
# dm = MNISTDataModule( path, batch_size = 25 )
# module = MNISTModule()
# trainer = pl.Trainer(
#     max_epochs = 10, 
#     logger = None, 
#     enable_checkpointing = False, 
#     # overfit_batches = 1,
# )

# trainer.fit ( module, dm )

# config = {
#     'datamodule': {
#         'path': Path('dataset'), 
#         'batch_size': 25, 
#     }, 
#     'trainer': {
#         'max_epochs': 10, 
#         'logger': None, 
#         'enable_checkpointing': False, 
#         'overfit_batches': 0, 
#     },
# }

# dm = MNISTDataModule( **config['datamodule'] )
# module = MNISTModule()
# trainer = pl.Trainer( **config['trainer'] )
# trainer.fit( module, dm )

config = {
    'datamodule': {
        'path': Path('dataset'), 
        'batch_size': 25, 
    },
    'trainer': {
        'max_epochs': 10, 
        'enable_checkpointing': False,
        'overfit_batches': 0,
    },
    'logger': None, 
    'callbacks': None,
}

def train(config):
    dm = MNISTDataModule( **config['datamodule'] )
    module = MNISTModule( config )
    # configurar el logger
    if config[ 'logger' ] is not None:
        if config[ 'logger' ] == 'WandbLogger':
            config[ 'trainer' ][ 'logger' ] = getattr( pl.loggers, config[ 'logger' ] )( **config[ 'logger_params' ], config = config )
        else:
            config[ 'trainer' ][ 'logger' ] = getattr( pl.loggers, config[ 'logger' ] )( **config[ 'logger_params' ] )

    # Configurar callbacks
    if config[ 'callbacks' ] is not None:
        callbacks = []
        for callback in config[ 'callbacks' ]:
            if callback[ 'name' ] == 'WandBCallback':
                dm.setup()
                callback[ 'params' ][ 'dl' ] = dm.val_dataloader()
            elif callback[ 'name' ] == 'ModelCheckpoint':
                callback[ 'params' ][ 'filename' ] = f'{callback[ "params" ][ "filename" ]}-{{val_loss:.5f}}-{{epoch}}'
            cb = getattr( importlib.import_module( callback[ 'lib' ] ), callback[ 'name' ] )( **callback[ 'params' ] )
            callbacks.append( cb )
            config[ 'trainer' ][ 'callbacks' ] = callbacks


    trainer = pl.Trainer( **config['trainer'] )
    trainer.fit( module, dm )
    trainer.save_checkpoint( 'checkpoints/final.ckpt' )

if __name__ == '__main__':
    if len( sys.argv ) > 1:
        config_file = sys.argv[1]
        if config_file:
            with open( config_file, 'r' ) as stream:
                loaded_config = yaml.safe_load( stream )
            deep_update( config, loaded_config )
    print( config )
    train( config )