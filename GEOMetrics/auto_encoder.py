from utils import *
from layers import * 
from models import *
from voxel  import voxel2obj
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
import pickle
import random, torch, argparse
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=40, help='Random seed.')
parser.add_argument('--model', type=str, default='gcn',
                    help='gcn or zgcn')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--exp_id', type=str, default='gcn_run_norm',
                    help='The experiment name')

args = parser.parse_args()
overfit = False
batch_size = 32
latent_length = 50
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Running : ', args.model)
# data settings
objects =['table', 'chair']#['bench','sofa','chair','lamp','table']
path = '/scratch/jiadeng_root/jiadeng/shared_data/datasets/GEOMetrics/shapenet/'
train_paths = []
valid_paths = []

print('Experiment : ',args.exp_id)

for p in glob(path+'/*'):
    if p.split('/')[-1] in objects:
        # print(p.split('/')[-1])
        cls_pths = glob(p+'/*')
        # ratio = int(len(cls_pths)*.7)
        random.shuffle(cls_pths)
        train_paths += cls_pths[:4000]
        valid_paths += cls_pths[4000:4500]
# print('\t training: ', len(train_paths))
# print('\t validation: ', len(valid_paths))


if overfit:
    train_paths = train_paths[:10]
    valid_paths = valid_paths[:10]
    args.epochs = 2


checkpoint_dir = "checkpoint/" +  args.exp_id +'/'
save_dir =  "plots/" +  args.exp_id +'/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Loading State
try:
    with open(checkpoint_dir+"/state.npy", 'rb') as handle:
        state = pickle.load(handle)
    print('Previous state found | Epoch: '+str(state['epoch'])+' Best: '+str(state['best']))
except:
    state ={}
    state['epoch'] = 0
    state['best'] = 1000


# ZGCN
if args.model == 'zgcn':
    train_data = Voxel_loader(train_paths, set_type = 'train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn = train_data.collate)

    valid_data = Voxel_loader(valid_paths, set_type = 'valid')
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn = valid_data.collate)

    # load models
    encoder_mesh = MeshEncoder(latent_length)
    decoder = Decoder(latent_length)

# GCN
elif args.model == 'gcn':
    train_data = GCN_Loader(train_paths, set_type = 'train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn = train_data.collate)

    valid_data = GCN_Loader(valid_paths, set_type = 'valid')
    valid_loader = DataLoader(valid_data, batch_size=batch_size*2, shuffle=False, collate_fn = valid_data.collate)

    # load models
    encoder_mesh = MeshEncoderGCN_elu(latent_length)
    decoder = Decoder(latent_length)

try:
    encoder_mesh.load_state_dict(torch.load(checkpoint_dir + 'encoder'))
    decoder.load_state_dict(torch.load(checkpoint_dir + 'decoder'))
    print('Loading from previous checkpoint..')
except:
    print('Starting from scratch')



# pytorch it up 
decoder.cuda(), encoder_mesh.cuda()
params = list(decoder.parameters()) + list(encoder_mesh.parameters())   
optimizer = optim.Adam(params,lr=args.lr)

tb = SummaryWriter(os.path.join('../tensorboard/',args.exp_id))

def normalize_verts(verts):
    # X
    if (verts[:,0].max() - verts[:,0].min()) != 0:
        verts[:,0] = ((verts[:,0] - verts[:,0].min())/(verts[:,0].max() - verts[:,0].min()))  - 0.5
    else:
        verts[:,0] = 0.1

    # Y
    if (verts[:,1].max() - verts[:,1].min()) != 0:
        verts[:,1] = ((verts[:,1] - verts[:,1].min())/(verts[:,1].max() - verts[:,1].min())) - 0.5
    else:
        verts[:,1] = 0.1

    # Z
    if (verts[:,2].max() - verts[:,2].min()) != 0:
        verts[:,2] = ((verts[:,2] - verts[:,2].min())/(verts[:,2].max() - verts[:,2].min())) - 0.5
    else:
        verts[:,2] = 0.1
    return verts

class Engine() :
    def __init__(self): 
        self.best = state['best']
        self.epoch = 0 
        self.train_losses = []
        self.valid_losses = []

    def forward(self, batch):
        voxel_gt = batch['voxels'].cuda()
        optimizer.zero_grad()    
        
        latent = None
        # pass through encoder
        take = []
        i = 0
        # FOR GCN
        if args.model == 'gcn':
            for verts, edges, idx in zip(batch['verts'], batch['edges'], batch['id']):
                verts = normalize_verts(verts.cuda())
                edges = edges.cuda()
                try:
                    if latent is None: 
                        latent = encoder_mesh(verts, edges).unsqueeze(0)
                    else: 
                        latent = torch.cat((latent,encoder_mesh(verts, edges).unsqueeze(0))) 
                    take.append(i)
                except:
                    print('failed for: ',  idx)
                i+=1
        # FOR ZGCN
        if args.model == 'zgcn':
            for verts, adj, idx in zip(batch['verts'], batch['adjs'], batch['id']):
                verts = normalize_verts(verts.cuda())
                adj = adj.cuda()
                try:
                    if latent is None:
                        latent = encoder_mesh(verts, adj).unsqueeze(0)
                    else:
                        latent = torch.cat((latent,encoder_mesh(verts,adj).unsqueeze(0)))
                    take.append(i)
                except:
                    print('failed for: ',  idx)
                i+=1
                
        voxel_gt = voxel_gt[take, ...]
        voxel_pred = decoder(latent)
        
        return voxel_gt, voxel_pred

    def train(self):
        decoder.train(),  encoder_mesh.train()
        total_loss = 0
        iteration = 0
        for batch in tqdm(train_loader): 
            #Run forward pass
            voxel_gt, voxel_pred = self.forward(batch)

            # calculate loss and optimize with respect to it 
            loss = torch.mean((voxel_pred- voxel_gt)**2 )
            loss.backward()
            optimizer.step()
            track_loss = loss.item()
            total_loss += track_loss

            # print info occasionally 
            if iteration % 200 ==0: 
                message = f'Train Loss: Epoch: {self.epoch}, loss: {track_loss}, best: {self.best}'
                tqdm.write(message)
            iteration += 1 

        self.train_losses.append(total_loss / float(iteration))

   


    def validate(self): 
        decoder.eval(), encoder_mesh.eval()
        total_loss = 0
        iteration = 0 
        for batch in tqdm(valid_loader): 
            voxel_gt, voxel_pred = self.forward(batch)

            # calculate loss and optimize with respect to it 
            loss = torch.mean((voxel_pred- voxel_gt)**2 )
            track_loss = loss.item()
            total_loss += track_loss

            # print info occasionally 
            if iteration % 20 ==0 : 
                message = f'Valid Loss: Epoch: {self.epoch}, new: {total_loss / float(iteration + 1 )}, cur: {self.best}'
                tqdm.write(message)
            iteration += 1 

        self.valid_losses.append(total_loss / float(iteration))
      
          
    def save(self): 
        if self.valid_losses[-1] <= self.best:
            self.best = self.valid_losses[-1] 
            torch.save(decoder.state_dict(), checkpoint_dir + 'decoder')
            torch.save(encoder_mesh.state_dict(), checkpoint_dir + 'encoder')

        self.state['best'] = self.best
        self.state['epoch'] = self.epoch
        with open(checkpoint_dir+"/state.npy", 'wb') as handle:
            pickle.dump(self.state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(encoder_mesh.state_dict(), checkpoint_dir + 'encoder_of')
        torch.save(decoder.state_dict(), checkpoint_dir + 'decoder_of')
    
    def run_tb(self, epoch):
        tb.add_scalar(args.exp_id+'/train', self.train_losses[-1], epoch)
        tb.add_scalar(args.exp_id+'/valid', self.valid_losses[-1], epoch)


trainer = Engine()
trainer.state = state


for epoch in range(state['epoch'], args.epochs):
    print('Epoch: ', epoch)
    trainer.epoch = epoch
    trainer.train()
    trainer.validate()
    trainer.save()
    trainer.run_tb(epoch)
    if overfit and epoch == 2:
        break

# print ('Saving latent codes of all models')
# encoder_mesh.load_state_dict(torch.load(checkpoint_dir + 'encoder'))
# encoder_mesh.eval()
# if not os.path.exists('data/latent/'):
#     os.makedirs('data/latent/')
# for batch in tqdm(train_loader):
#     for v, a, n  in zip(batch['verts'], batch['adjs'], batch['names']):
        
#         latent = encoder_mesh(v.cuda(), a.cuda() )
#         np.save('data/latent/' + n + '_latent', latent.data.cpu().numpy())
