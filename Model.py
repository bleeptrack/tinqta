
import os.path as osp
from functools import reduce
import math
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, LinearLR
from torch.nn import AvgPool1d, MaxPool1d
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_node
from torch_geometric.nn import GAE, VGAE, GCNConv, global_mean_pool, global_max_pool, Linear, TransformerConv,  SAGEConv, GCN
from DrawData import GraphDataset
from config import config
from torch_geometric.loader import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import subgraph


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        num_features = 2

        self.conv1 =  GCNConv(num_features, 2 * num_features)
        self.conv2 =  GCNConv(2 * num_features, 4 * num_features)
        self.lin = Linear(-1, config['latent_size'])

        self.mu = Linear(-1, config['latent_size'])
        self.logvar = Linear(-1, config['latent_size'])


    def pool_pairs(self, x):

        pool = MaxPool1d(2, stride=2)
        x = pool(x.permute(1,0)).permute(1,0)

        edge_index = torch.cat( ( torch.arange(0,x.size()[0]-1).unsqueeze(1), torch.arange(1,x.size()[0]).unsqueeze(1) ), 1)
        edge_index2 = torch.cat( ( torch.arange(1,x.size()[0]).unsqueeze(1), torch.arange(0,x.size()[0]-1).unsqueeze(1) ), 1)

        edge_index = torch.cat( (edge_index, edge_index2), 0).t().contiguous()

        #print(edge_index, edge_index.size())

        return x, edge_index

    def forward(self, x, edge_index):
        #encode
        x = self.conv1(x, edge_index).relu()
        x, edge_index = self.pool_pairs(x)

        x = self.conv2(x, edge_index).relu()
        x, edge_index = self.pool_pairs(x)

        #reshape to linear
        x = x.flatten()

        return self.mu(x), self.logvar(x)

#    def forward(self, x, edge_index):
#        #encode
#        x = self.conv1(x, edge_index).relu()
#        x, edge_index = self.pool_pairs(x)
#
#        x = self.conv2(x, edge_index).relu()
#        x, edge_index = self.pool_pairs(x)
#
#        #reshape to linear
#        x = x.flatten()
#
#        return self.lin(x)

class GCNDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.lin1 = Linear(-1, int(config['nrPoints']*2 /4) )
        self.lin2 = Linear(-1, int(config['nrPoints']*2 /2) )
        self.lin3 = Linear(-1, int(config['nrPoints']*2) )

        #mehr decoding layer machen das auf jeden fall besser!
        #aber halt auch echt nicht beliebig


    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x)

        #reshape
        return x.unflatten(0, ( config['nrPoints'], 2 ) )




class PatternEncoder(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super().__init__()

        extended_lat_size = int( config['latent_size'] + 4 ) # +4 fuer posx, posy, scale, rot

        self.conv = GCNConv( extended_lat_size , extended_lat_size * 2 )
        self.conv2 = GCNConv( extended_lat_size * 2 , extended_lat_size )
        self.lin1 = Linear(-1, extended_lat_size )
        self.lin2 = Linear(-1, extended_lat_size*8 )
        self.lin22 = Linear(-1, extended_lat_size*8 )
        self.lin23 = Linear(-1, extended_lat_size*8 )
        self.lin3 = Linear(-1, extended_lat_size )

    def dropout_node_min(self, edge_index, batch_vector=None, p = 0.5, num_nodes = None, min_node=3, training = True):


        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if not training or p == 0.0:
            node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
            edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
            return edge_index, edge_mask, node_mask


        if batch_vector is None:
            batch_vector = torch.zeros(num_nodes)

        count = torch.bincount(batch_vector)
        probs = []
        for i in range(count.size(0)):
            prob = torch.cat( (torch.rand( count[i]-min_node, device=edge_index.device), torch.ones(min_node)), 0)
            r=torch.randperm(prob.size(0))
            prob=prob[r]
            probs.append(prob)

        prob = torch.cat(probs, 0)
        node_mask = prob > p
        edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                            num_nodes=num_nodes,
                                            return_edge_mask=True)
        return edge_index, edge_mask, node_mask


    def forward(self, x, edge_index, batch_vector=None):
        #encode
        #x = self.conv1(x, edge_index).relu()

        #dropout
        #edge_index, edge_mask, node_mask = self.dropout_node_min(edge_index, batch_vector, p=config['node_dropout'], training=self.training)

        x = self.conv(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        #second part of dropout
        #x = x[node_mask]
        #if batch_vector is not None:
        #    batch_vector = batch_vector[node_mask]

        x = global_mean_pool(x, batch_vector)
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin22(x).relu()
        x = self.lin23(x).relu()
        x = self.lin3(x)
        x = torch.squeeze(x)
        x = x.flatten()  #to come back to the bached format from pygeometric
        return x




#class GCNEncoder(torch.nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.conv1 = GCNConv(in_channels, 2 * out_channels)
#        self.conv2 = GCNConv(2 * out_channels, out_channels)
#
#    def forward(self, x, edge_index):
#        x = self.conv1(x, edge_index).relu()
#        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def train(train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


class LineTrainer():

    def __init__(self, name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = T.Compose([
        # T.NormalizeFeatures(),
            T.ToDevice(device),
        # T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                   split_labels=True, add_negative_train_samples=False),
        ])

        self.model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'lineModels', name)
        self.dataset = GraphDataset(name, level="line", transform=transform)
        self.in_channels, self.hidden_channels, self.out_channels = self.dataset.num_features, self.dataset.num_features*2, config['latent_size']
        #out_channels hab ich mal verdoppelt
        self.name = name

        self.encoder = GCNEncoder(self.in_channels, self.out_channels)
        #self.encoder = GCN(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=1, out_channels=self.out_channels)


        self.decoder = GCNDecoder(self.in_channels, self.out_channels)
        self.model = GAE(self.encoder, self.decoder)
        #model = GAE(LinearEncoder(in_channels, out_channels))
        #model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
        #model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

        self.epochs = 10000000 #100000

        self.model = self.model.to(device)
        if osp.exists(self.model_path):
            print("MODEL EXISTS. LOADING...")
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        ##self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        ##self.scheduler = ReduceLROnPlateau(self.optimizer, patience=60, cooldown=20)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
       
        #lr war mal 0.1. 0.001 war jetzt besser

    def loss_function(self, out, target, mu, logvar, epoch):

        criterionMSE = torch.nn.MSELoss()
        lossMSE = criterionMSE(out, target)

        beta_norm = 0.01 #(1000 * config['latent_size']) / (config["nrPoints"]*2)
        kl_annealing = 0.001 #0.001
        kl_weight = max(0, min( (epoch -200) * kl_annealing, 1 ) )


        mse_annealing = kl_annealing
        mse_weight = 1 #1 - max(0.5, min( (epoch -1200) * mse_annealing, 1 ) ) + 0.5


        kld_loss = torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
        loss = (mse_weight * lossMSE) + (kl_weight * beta_norm * kld_loss)

        return loss, (mse_weight * lossMSE), (kl_weight * beta_norm * kld_loss), kl_weight, mse_weight


    def trainModel(self, progress_callback=None):
        self.model.train()
        criterionMSE = torch.nn.MSELoss()
        criterionKLD = torch.nn.KLDivLoss(reduction='batchmean')

        loss_list = []

        for epoch in range(self.epochs):

            # Initializing variable for storing
            # loss
            running_loss = 0

            # Iterating over the training dataset
            for train_data in self.dataset:

                # Loading image(s) and
                # reshaping it into a 1-d vector


                # Generating output
                mu, logvar = self.model.encode(train_data.x, train_data.edge_index)
                z = self.reparameterize(mu, logvar)
                out = self.model.decode(z)

                # Calculating loss
                loss, l1, l2, w, m = self.loss_function(out, train_data.x, mu, logvar, epoch)

                # Updating weights according
                # to the calculated loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #gradient clipping to prevent loss becoming NaN
                self.optimizer.step()


                # Incrementing loss
                running_loss += loss.item()


            # Averaging out loss over entire batch
            running_loss /= self.dataset.len()

            #if epoch > 1400 and running_loss<200:
                #self.scheduler.step(running_loss)

            loss_list.append(running_loss)
            if len(loss_list) > 30:
                loss_list = loss_list[1:]

            print("Epoch:", epoch, "Loss:", running_loss, l1, l2, m, w)

            if math.isnan(running_loss):
                die()

            diff = 0
            for d in range(1,len(loss_list)):
                diff = diff + abs(loss_list[d] - loss_list[d-1])

            print("total", diff)
            if running_loss < 0.1:
                break

            if epoch % 100 == 0:
                torch.save(self.model.state_dict(), self.model_path)
                print("saving...")
                
                
            if progress_callback:
                
                if epoch % 15 == 0:
                    print("extracting vectors for animation")
                    vectors, originpoints = self.extractOriginLineVectors()
                    
                    progress_callback(self, vectors)
                #else:
                #    progress_callback(self, l1.item())

        torch.save(self.model.state_dict(), self.model_path)


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def extractOriginLineVectors(self):
        self.model.eval()
        vectors = []
        originpoints = []
        for i in range(self.dataset.len()):
            mu, logvar = self.model.encode(self.dataset[i].x, self.dataset[i].edge_index)
            z = self.reparameterize(mu, logvar)
            vectors.append(z)
            originpoints.append(self.dataset[i].x)
        return vectors, originpoints

    def encodeLineVector(self, x, edge_index):
        self.model.eval()

        mu, logvar = self.model.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)

        return z


    #slightly changes existing lines to keep their scaling and rotation information. random latent vectors would
    #not have this information currently
    def generate(self, nr, faktor='random'):
        self.model.eval()

        lines = []
        scales = []
        rotations = []

        for i in range(nr):
            if faktor is 'random':
                lines.append(self.model.decode(self.randomLatentVector()))
                scales.append(1)
                rotations.append(random.random())
            else:
                datanum = random.randint(0,self.dataset.len()-1)



                mu1, logvar1 = self.model.encode(self.dataset[datanum].x, self.dataset[datanum].edge_index)
                z1 = self.reparameterize(mu1, logvar1)

                datanum2 = random.randint(0,self.dataset.len()-1)
                mu2, logvar2 = self.model.encode(self.dataset[datanum2].x, self.dataset[datanum2].edge_index)
                z2 = self.reparameterize(mu2, logvar2)

                #jitter = 0.1

                #z = torch.multiply(torch.rand(z1.size()), jitter)
                #ToDo beim randomnessfaktor könnte man gut aus einer gaussverteilung ziehen. Könnte man hier nicht generell aus der gaussverteilung ziehen?
                z = torch.add( torch.multiply(self.randomLatentVector(), random.random()*faktor), z1)

                #z = z1

                #z = torch.add( torch.divide(torch.subtract(z2,z1), random.randint(10,20)), z1)
                lines.append(self.model.decode(z))
                scales.append(self.dataset[datanum].scale.item())
                rotations.append(self.dataset[datanum].rotation.item())

        #for i in range(nr):
        #    z = self.randomLatentVector()
        #    lines.append(self.model.decode(z))



        return lines, scales, rotations


    def generateFlower(self, nr):
        self.model.eval()

        lines = []



        #num1 = random.randint(0,self.dataset.len()-1)
        #num2 = random.randint(0,self.dataset.len()-1)
        #z1 = self.encodeLineVector(self.dataset[num1].x, self.dataset[num1].edge_index)
        #z2 = self.encodeLineVector(self.dataset[num2].x, self.dataset[num2].edge_index)

        z1 = self.randomLatentVector()
        z2 = self.randomLatentVector()

        z_dir = torch.subtract(z2,z1)
        z_step = torch.divide(z_dir, nr)

        for i in range(nr):
            z_fin = torch.add(z1, torch.multiply(z_step, i))
            lines.append(self.model.decode(z_fin))

        return lines

    def randomLatentVector(self):

        z = torch.randn(config['latent_size'])

        #z = z.to(current_device)
        return z

    def decode_latent_vector(self, z):
        self.model.eval()


        return self.model.decode(z)

class PatternTrainer():

    def __init__(self, name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = T.Compose([
        # T.NormalizeFeatures(),
            T.ToDevice(device),
        # T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                   split_labels=True, add_negative_train_samples=False),
        ])

        self.model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'patternModels', name)
        self.dataset = GraphDataset(name, level="pattern", transform=transform)
        self.loader = DataLoader(self.dataset, batch_size=config['batch_size'])
        
        self.in_channels = self.out_channels = self.dataset.num_features
        self.hidden_channels = self.dataset.num_features*2


        #out_channels hab ich mal verdoppelt
        self.name = name

        self.model = PatternEncoder(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=1, out_channels=self.out_channels)

        #model = GAE(LinearEncoder(in_channels, out_channels))
        #model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
        #model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

        self.epochs = 100000 #100000

        self.model = self.model.to(device)
        if osp.exists(self.model_path):
            print("MODEL EXISTS. LOADING...")
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer)
        #self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        #self.scheduler = LinearLR(self.optimizer, total_iters=200, verbose=True)



        #for train_data in self.dataset:
        #    print(train_data)




    def trainModel(self):
        self.model.train()
        criterion = torch.nn.MSELoss(reduction='sum')

        loss_list = []
        avg_epoch_loss = 0

        for epoch in range(self.epochs):

            running_loss = 0

            #for train_data in self.dataset:
            for train_data in self.loader:

                out = self.model.forward(train_data.x, train_data.edge_index, train_data.batch)
                #print(train_data.x[2,:])
                #print(train_data.x.size())
                #print(out[1:14])
                #print(train_data.y.size())


                #loss = criterion(out, train_data.y)
                loss = self.loss_function(out, train_data.y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
               



            running_loss /= len(self.loader.sampler)
            self.scheduler.step(running_loss)


            print("Epoch:", epoch, "Loss:", running_loss)
            avg_epoch_loss += running_loss


            if epoch % 100 == 0:
                torch.save(self.model.state_dict(), self.model_path)
                print("saving...", "Epoch:", epoch, "Loss:", avg_epoch_loss/100, "current loss:", running_loss)
                avg_epoch_loss = 0

        torch.save(self.model.state_dict(), self.model_path)
        
    def loss_function(self, out, ground_truth):
        res = torch.unflatten(out, 0, (-1, 4+config['latent_size']))
        test = torch.unflatten(ground_truth, 0, (-1, 4+config['latent_size']))
        
        resPos = res[:, 0:4]
        resVec = res[:, 4:]
        testPos = test[:, 0:4]
        testVec = test[:, 4:]
        
        posLoss = torch.nn.MSELoss(reduction='mean')(resPos, testPos)
        vecLoss = torch.nn.MSELoss(reduction='mean')(resVec, testVec)
       
        return posLoss*50 + vecLoss
    
    def getDatasetSample(self):
        self.model.eval()
        datanum = random.randint(0,self.dataset.len()-1)
        return self.dataset[datanum].x

    def generate(self):
        #wir haben hier noch nie probiert mit einem richtigen random z was zu generieren, oder?
        self.model.eval()

        datanum = random.randint(0,self.dataset.len()-1)
        print(datanum)
        z = self.model.forward(self.dataset[datanum].x, self.dataset[datanum].edge_index)
        print(z)
        print(self.dataset[datanum].y)
        return z, self.dataset[datanum].y, self.dataset[datanum].x

    def predict(self, x, edge_index):
        self.model.eval()
        return self.model.forward(x, edge_index)
    
  







#for epoch in range(1, epochs + 1):
#    for train_data, val_data, test_data in dataset:
#        loss = train(train_data)
#        auc, ap = test(test_data)
#        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
