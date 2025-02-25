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
from torch_geometric.nn import GAE, VGAE, GCNConv,GraphConv, global_mean_pool, global_add_pool, Linear, TransformerConv,  SAGEConv, GCN
from DrawData import GraphDatasetHandler
from line import Line
from config import config
from torch_geometric.loader import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import numpy as np


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        num_features = 2
        

        self.conv1 =  GCNConv(num_features, 2 * num_features)
        self.conv2 =  GCNConv(2 * num_features, 4 * num_features)
        #self.lin = Linear(-1, config['latent_size'])

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

        batch_size = x.size(0)/config['nrPoints']

        x = self.conv1(x, edge_index).relu()
        x, edge_index = self.pool_pairs(x)


        x = self.conv2(x, edge_index).relu()
        x, edge_index = self.pool_pairs(x)

        #reshape to linear
        
        x = x.flatten()
      
        second_size = x.size(0)/batch_size
        x = x.view(int(batch_size), int(second_size))


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
        batch_size = x.size(0)

        #add batch dimension if not present
        if len(x.size()) == 1:
            x = x.unsqueeze(0)

        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x)       

        x = x.view(-1, 60, 2)
        x = x.flatten(0, 1)

        #remove batch dimension if there is no batch
        if x.size(0) == 1:
            x = x.squeeze(0)
       
        return x

        #reshape
        #return x.unflatten(0, ( config['nrPoints'] * batch_size, 2 ) )




class PatternEncoder(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super().__init__()

        extended_lat_size = int( config['latent_size'] + 4 ) # +4 fuer posx, posy, scale, rot

        self.conv = GCNConv( extended_lat_size , extended_lat_size * 8 )

        self.lin = Linear(-1, extended_lat_size)
        

    def delaunay_pool(self, x, edge_index, batch_vector):
        print(x.requires_grad) 
        print("x init", x.size())

        num_nodes = x.size(0)

        # Convert edge_index to adjacency list format, ignoring self-loops
        edge_list = edge_index.t().tolist()
        print("edge_list", edge_list)
        adj = [[] for _ in range(num_nodes)]
        for u, v in edge_list:
            if u != v:  # Skip self-loops
                adj[u].append(v)
                adj[v].append(u)

        # Find all triangles
        triangles = set()
        for u, v in edge_list:
            if u == v:  # Skip self-loops
                continue
            if batch_vector[u] != batch_vector[v]:
                continue
            for w in adj[v]:
                if w != u and w != v and w in adj[u] and batch_vector[w] == batch_vector[u]:
                    triangle = tuple(sorted([u,v,w]))
                    triangles.add(triangle)

        print("triangles", triangles)

        # Create new features and batch assignments for collapsed triangles
        new_features = []
        new_batch = []
        processed_nodes = set()

        # Process triangles
        for triangle in triangles:
            # Average features of triangle nodes
            triangle_features = torch.mean(x[list(triangle), :], dim=0)
            new_features.append(triangle_features)
            new_batch.append(batch_vector[triangle[0]])  # Use batch of first node
            
            for node in triangle:
                processed_nodes.add(node)

        # Handle remaining nodes that weren't part of any triangle
        for i in range(num_nodes):
            if i not in processed_nodes:
                new_features.append(x[i])
                new_batch.append(batch_vector[i])
                print("not part of any triangle", i)

        # Convert to tensor format
        new_x = torch.stack(new_features)
        new_batch_vector = torch.tensor(new_batch, device=x.device)
        
        # Sort based on batch vector
        sorted_indices = torch.argsort(new_batch_vector)
        new_x = new_x[sorted_indices]
        new_batch_vector = new_batch_vector[sorted_indices]
        
        # Return dummy edge_index since it will be rebuilt later
        dummy_edge_index = torch.empty((2, 0), device=edge_index.device)

        print("new_x shape:", new_x.shape)
        print("dummy_edge_index shape:", dummy_edge_index)
        print("new_batch_vector shape:", new_batch_vector)

        for batch in torch.unique(new_batch_vector):
            batch_mask = new_batch_vector == batch
            batch_pos = new_x[batch_mask, 0:2].detach()
            print("\nProcessing batch:", batch)
            print("Number of points in batch:", len(batch_pos))
            print("Points:", batch_pos)
            
            data = T.Delaunay()(Data(pos=batch_pos))
            if data.face is not None:
                data = T.FaceToEdge()(data)
            print("Created edges:", data.edge_index.t().tolist())
            print("Number of edges:", data.edge_index.shape[1])
            

        return new_x, dummy_edge_index, new_batch_vector
    
    
    def readout(self, x, batch_vector):
        return torch.cat([global_mean_pool(x, batch_vector), global_add_pool(x, batch_vector)], dim=-1)


    def forward(self, x, edge_index, batch_vector=None):
        print(batch_vector)
        print(x[0:10,0])
        #x = self.conv(x, edge_index)
        x, edge_index, batch_vector = self.delaunay_pool(x, edge_index, batch_vector)
        x = self.readout(x, batch_vector)
        x = self.lin(x)
        return x



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
        self.dataset = GraphDatasetHandler.load_data(name, "line")
        self.loader = DataLoader(self.dataset.data, batch_size=config['batch_size_line'], shuffle=True)
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
            print("LINE MODEL EXISTS. LOADING...", self.model_path)
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        ##self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        ##self.scheduler = ReduceLROnPlateau(self.optimizer, patience=60, cooldown=20)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        
        self.epoch_training_offset = 1000 #200
        self.kl_annealing = 0.001
       
        #lr war mal 0.1. 0.001 war jetzt besser

    def loss_function(self, out, target, mu, logvar, epoch):

        criterionMSE = torch.nn.MSELoss()
        lossMSE = criterionMSE(out, target)

        beta_norm = 1 #0.01 #(1000 * config['latent_size']) / (config["nrPoints"]*2)
        kl_annealing = self.kl_annealing #0.001
        kl_weight = max(0, min( (epoch - self.epoch_training_offset) * kl_annealing, 1 ) )


        mse_annealing = kl_annealing
        mse_weight = 1 #1 - max(0.5, min( (epoch -1200) * mse_annealing, 1 ) ) + 0.5

        #ToDo: kld_loss braucht das ein batch mean oder so?
        kld_loss = torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
        loss = (mse_weight * lossMSE) + (kl_weight * beta_norm * kld_loss)

        return loss, (mse_weight * lossMSE), (kl_weight * beta_norm * kld_loss), kl_weight, mse_weight


    def trainModel(self, progress_callback=None):
        self.model.train()
        criterionMSE = torch.nn.MSELoss()
        criterionKLD = torch.nn.KLDivLoss(reduction='batchmean')

        loss_list = []
        min_loss = 1000
        min_loss_epoch = 1

        for epoch in range(self.epochs):

            # Initializing variable for storing
            # loss
            running_loss = 0

            # Iterating over the training dataset
            #ToDo: hier muss man die loader verwenden?
            for train_data in self.loader:

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
            running_loss /= len(self.loader)

            #if epoch > 1400 and running_loss<200:
                #self.scheduler.step(running_loss)

            loss_list.append(running_loss)
            if len(loss_list) > 30:
                loss_list = loss_list[1:]

            print("Epoch:", epoch, "Loss:", running_loss, l1, l2, m, w)

            if math.isnan(running_loss):
                die_linetrainen_runningloss_nan()

            diff = 0
            for d in range(1,len(loss_list)):
                diff = diff + abs(loss_list[d] - loss_list[d-1])

          
            
            
            if epoch > (1/self.kl_annealing + self.epoch_training_offset):
                if running_loss < min_loss:
                    min_loss = running_loss
                    min_loss_epoch = epoch
                    torch.save(self.model.state_dict(), self.model_path)
                    print("saving...")
                    
                if epoch - 100 > min_loss_epoch:
                    print("FINISHED TRAINING as loss:", min_loss)
                    if progress_callback:
                        progress_callback(self, 100, "finished")
                    break
            
                
            #if epoch % 100 == 0:
            #    torch.save(self.model.state_dict(), self.model_path)
            #    print("saving...")
                
                
            if progress_callback:
                
                if epoch % 15 == 0:
                    print("extracting vectors for animation")
                    vectors, originpoints = self.extractOriginLineVectors()
                    
                    progress_callback(self, vectors)
                elif epoch % 5 == 0: 
                    label = "training"
                   
                    min_epochs = 1/self.kl_annealing + self.epoch_training_offset
                    percent = min_epochs / 90
                        
                    if epoch <= self.epoch_training_offset:
                        label = "warming up"
                    if epoch > (1/self.kl_annealing + self.epoch_training_offset):
                        label = "finalizing"
                    progress_callback(self, round(min(epoch, 1/self.kl_annealing + self.epoch_training_offset)/percent), label)

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
        z = eps * std + mu
        if z.size(0) == 1:
            z = z.squeeze(0)
        #print("reparameterize", z.size(), z)
        return z

    def extractOriginLineVectors(self):
        self.model.eval()
        vectors = []
        originpoints = []
        for data in self.dataset.original_data:
            mu, logvar = self.model.encode(data.x, data.edge_index)
            z = self.reparameterize(mu, logvar)
            vectors.append(z)
            originpoints.append(data.x)
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
        

        for i in range(nr):
            if faktor is 'random':
                z = self.randomLatentVector()

                
                l = Line(self.model.decode(z))
                lines.append(l)
            else:
                datanum = random.randint(0,self.dataset.len()-1)



                mu1, logvar1 = self.model.encode(self.dataset[datanum].x, self.dataset[datanum].edge_index)
                z1 = self.reparameterize(mu1, logvar1)
                print("encoded", z1)

                datanum2 = random.randint(0,self.dataset.len()-1)
                mu2, logvar2 = self.model.encode(self.dataset[datanum2].x, self.dataset[datanum2].edge_index)
                z2 = self.reparameterize(mu2, logvar2)

                #jitter = 0.1

                #z = torch.multiply(torch.rand(z1.size()), jitter)
                #ToDo beim randomnessfaktor könnte man gut aus einer gaussverteilung ziehen. Könnte man hier nicht generell aus der gaussverteilung ziehen?
                z = torch.add( torch.multiply(self.randomLatentVector(), random.random()*faktor), z1)

                #z = z1

                #z = torch.add( torch.divide(torch.subtract(z2,z1), random.randint(10,20)), z1)
                
                l = Line.from_tensor(self.model.decode(z), self.dataset[datanum].scale.item(), self.dataset[datanum].rotation.item())
                lines.append(l)
        #for i in range(nr):
        #    z = self.randomLatentVector()
        #    lines.append(self.model.decode(z))


        print(lines)
        return lines

    def randomLatentVector(self):

        z = torch.randn(config['latent_size'])

        #z = z.to(current_device)
        return z
    
    def randomInitPoint(self):
        z = torch.randn(config['latent_size'])
        pos = (torch.rand(2) * 2) - 1
        scalerot = torch.rand(2)
        return torch.cat( (pos,scalerot, z), 0)
    
    def getClosestMatch(self, z):
        latentvectors, _ = self.extractOriginLineVectors()
        dist = float('inf')
        latentvectors.insert(0, z)
        latentTensor = torch.stack(latentvectors, dim=0)
        dists = torch.cdist(latentTensor, latentTensor, p=2)
        idx = torch.argmin(dists[0][1:]) + 1 #+1 to account for inserted z at beginning
        closest = latentvectors[idx]
        return closest

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
        self.dataset = GraphDatasetHandler.load_data(name, "pattern")
        print("dataset info after loading ", len(self.dataset), self.dataset.level)
        self.loader = DataLoader(self.dataset.data, batch_size=config['batch_size_pattern'], shuffle=True)
        
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
            print("PATTERNMODEL EXISTS. LOADING...", self.model_path)
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer)
        #self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        #self.scheduler = LinearLR(self.optimizer, total_iters=200, verbose=True)



        #for train_data in self.dataset:
        #    print(train_data)




    def trainModel(self, progress_callback=None):
        self.model.train()
        criterion = torch.nn.MSELoss(reduction='sum')

        loss_list = []
        avg_epoch_loss = 0

        for epoch in range(self.epochs):

            running_loss = 0


            #for train_data in self.dataset:
            for train_data in self.loader:
                out = self.model.forward(train_data.x, train_data.edge_index, train_data.batch)


                #loss = criterion(out, train_data.y)
                loss = self.loss_function(out, train_data.y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
               


            #ToDo: num_graphs auch beim line training?
            running_loss /= len(self.loader)
            self.scheduler.step(running_loss)


            print("Epoch:", epoch, "Loss:", running_loss, "sampler", len(self.loader))
            avg_epoch_loss += running_loss


            if epoch % 50 == 0:
                torch.save(self.model.state_dict(), self.model_path)
                print("saving...", "Epoch:", epoch, "Loss:", avg_epoch_loss/100, "current loss:", running_loss)
                avg_epoch_loss = 0
                if progress_callback:
                    progress_callback(self.name)

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
       
        #ToDo: die gewichtung hier ist zufällig, finde eine bessere
        return posLoss*10 + vecLoss
    
    def getDatasetSample(self):
        self.model.eval()
        datanum = random.randint(0,self.dataset.len()-1)
        return self.dataset[datanum].x

    def generate(self):
        #wir haben hier noch nie probiert mit einem richtigen random z was zu generieren, oder?
        self.model.eval()
        #print("dataset info", len(self.dataset), self.dataset.level)

        data = self.dataset.get_random_item()
       
        z = self.model.forward(data.x, data.edge_index)
        return z, data.y, data.x

    def predict(self, x, edge_index):
        self.model.eval()
        return self.model.forward(x, edge_index)
    
  







#for epoch in range(1, epochs + 1):
#    for train_data, val_data, test_data in dataset:
#        loss = train(train_data)
#        auc, ap = test(test_data)
#        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
