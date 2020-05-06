import argparse
import logging
import os,sys
from typing import Type
import random 
from tqdm import tqdm

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from gcnna.data.datasets import ShapenetDataset
from gcnna.models.base_nn import GraphConvClf
from gcnna.config import Config
from gcnna.utils.torch_utils import train_val_split, save_checkpoint, accuracy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import warnings
warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser("Run training for a particular phase.")
parser.add_argument(
    "--config-yml", required=True, help="Path to a config file for specified phase."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the results directory.",
)


logger: logging.Logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    # --------------------------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # --------------------------------------------------------------------------------------------
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config_yml, _A.config_override)

    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Create serialization directory and save config in it.
    os.makedirs(_C.CKP.experiment_path, exist_ok=True)
    _C.dump(os.path.join(_C.CKP.experiment_path, "config.yml"))
    
    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    _C.DEVICE = device

    
    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER & CRITERION
    # --------------------------------------------------------------------------------------------
    ## Datasets
    trn_objs, val_objs = train_val_split(config=_C)
    collate_fn = ShapenetDataset.collate_fn
    
    if _C.OVERFIT:
        trn_objs, val_objs = trn_objs[:10], val_objs[:10]
        _C.OPTIM.EPOCH = 10
    
    trn_dataset = ShapenetDataset(_C, trn_objs)
    trn_dataloader = DataLoader(trn_dataset, 
                            batch_size=_C.OPTIM.BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fn, 
                            num_workers=_C.OPTIM.WORKERS)
    
    val_dataset = ShapenetDataset(_C, val_objs)
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=_C.OPTIM.VAL_BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fn, 
                            num_workers=_C.OPTIM.WORKERS)
    
    print("Training Samples: "+str(len(trn_dataloader)))
    print("Validation Samples: "+str(len(val_dataloader)))
    

    model = GraphConvClf(_C).cuda()
#     model.load_state_dict(torch.load('results/exp_03_16_11_22_19_10classes/model@epoch3.pkl')['state_dict'])

#     optimizer = optim.SGD(
#         model.parameters(),
#         lr=_C.OPTIM.LR,
#         momentum=_C.OPTIM.MOMENTUM,
#         weight_decay=_C.OPTIM.WEIGHT_DECAY,
#     )
    optimizer = optim.Adam(
        model.parameters(),
        lr=_C.OPTIM.LR,
    )
    iters = len(trn_dataloader)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = int(iters/4), T_mult=1, eta_min=1e-6, last_epoch=-1)
    
    ## Tensorboard
    tb = SummaryWriter(os.path.join('tensorboard/',(_C.EXPERIMENT_NAME))) 
    
    criterion = nn.CrossEntropyLoss()
    args  = {}
    args['EXPERIMENT_NAME'] =  _C.EXPERIMENT_NAME
    args['full_experiment_name'] = _C.CKP.full_experiment_name
    args['experiment_path'] = _C.CKP.experiment_path
    args['best_loss'] = _C.CKP.best_loss
    args['best_acc'] = _C.CKP.best_acc
    # --------------------------------------------------------------------------------------------
    #   TRAINING LOOP
    # --------------------------------------------------------------------------------------------
    total_step = len(trn_dataloader)
    print('\n ***************** Training *****************')
    for epoch in tqdm(range(_C.OPTIM.EPOCH)):
        # --------------------------------------------------------------------------------------------
        #   TRAINING 
        # --------------------------------------------------------------------------------------------
        running_loss = 0.0
        print('Epoch: '+str(epoch))
        model.train()
        
        for i, data in enumerate(tqdm(trn_dataloader), 0):
            if data[0] == None and data[1] == None:
                continue
            label = data[0].cuda()
            mesh = data[1].cuda()
            # Step the scheduler
            scheduler.step(epoch + i / iters)
            tb.add_scalar(_C.EXPERIMENT_NAME+'/Learning_Rate', scheduler.get_lr()[0], i*(epoch+1))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            try:
                outputs = model(mesh)
            except:
#                 print('failed for :'+str(data[2]))
                tb.add_text(_C.EXPERIMENT_NAME+'/skipped_train_objects', str(data[2]), i*(epoch+1))
                continue
            #print(outputs, label)
            if outputs.size()[0] == label.size()[0]:
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                #lr_scheduler.step()
                # print statistics
                running_loss += loss.item()
            else:
                print('Shape Mismatch')
                print(outputs.size(), label.size())
                print(mesh.verts_packed_to_mesh_idx().unique(return_counts=True)[1])
        running_loss /= len(trn_dataloader)
        print('\n\tTraining Loss: '+ str(running_loss))
        tb.add_scalar(_C.EXPERIMENT_NAME+'/train_loss', running_loss, epoch)
        
        # ----------------------------------------------------------------------------------------
        #   VALIDATION
        # ----------------------------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        print("\n\n\tEvaluating..")
        for i, data in enumerate(tqdm(val_dataloader), 0):
            if data[0] == None and data[1] == None:
                continue
            label = data[0].cuda()
            mesh = data[1].cuda()
            with torch.no_grad():
                try:
                    batch_prediction = model(mesh)
                except:
#                     print('failed for :'+str(data[2]))
                    tb.add_text(_C.EXPERIMENT_NAME+'/skipped_val_objects', str(data[2]), i*(epoch+1))
                    continue
                
                if batch_prediction.size()[0] == label.size()[0]:
                    loss = criterion(batch_prediction, label)
                    acc = accuracy(batch_prediction, label)
                    val_loss += loss.item()
                    val_acc += np.sum(acc)
                else:
                    print('Shape Mismatch')
                    print(batch_prediction.size(), label.size())
                    print(mesh.verts_packed_to_mesh_idx().unique(return_counts=True)[1])
        # Average out the loss
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        print('\n\tValidation Loss: '+str(val_loss))
        print('\tValidation Acc: '+str(val_acc.item()))
        tb.add_scalar(_C.EXPERIMENT_NAME+'/val_loss', val_loss, epoch)
        tb.add_scalar(_C.EXPERIMENT_NAME+'/val_acc', val_acc.item(), epoch)
        # Final save of the model
        args = save_checkpoint(model    = model,
                             optimizer  = optimizer,
                             curr_epoch = epoch,
                             curr_loss  = val_loss,
                             curr_step  = (total_step * epoch),
                             args       = args,
                             curr_acc   = val_acc.item(),
                             trn_loss   = running_loss,
                             filename   = ('model@epoch%d.pkl' %(epoch)))
        torch.save(args['best_model'], os.path.join(args['experiment_path'], 'best_model.pth'))
          
        print('---------------------------------------------------------------------------------------\n')
    print('Finished Training')
    print('Best Accuracy on validation',args['best_acc'])
    print('Best Loss on validation',args['best_loss'])
    tb.close() 