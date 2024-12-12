import torch
import os
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GAE, GCNConv
from config import config
import random

#Loads the dataset and handles delivery
class GraphDataset(InMemoryDataset):
    def __init__(self, name, level, transform=None, pre_transform=None, pre_filter=None):
        self.base_name = name
        self.name = name+"-"+level
        self.level = level
        #self.maxVal = 0

        root = osp.join(osp.dirname(osp.realpath(__file__)), 'data', self.name)


        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        print(self.name)

    @property
    def raw_file_names(self):
        return [self.name + '.pt']


    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def num_nodes(self):
        return self.get(0).num_nodes

    def download(self):
        print("Nothing to download...")

    def process(self):
        # Read data into huge `Data` list.
        complete_data = torch.load(osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', self.name +'.pt'))

        print("handling data at level ", self.level)

        data_list = complete_data[self.level]
        print("Loading Dataset of length: ", len(data_list))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphHandler:
    def __init__(self):
        self.clear()

    def clear(self):
        self.raw_data = []
        self.outer_ids = []

    def init_raw(self,raw_data):
        #print(raw_data)
        self.raw_data = []
        self.outer_ids = []
        self.add_raw(raw_data)
            
    def add_raw(self, raw_data):
        self.raw_data += raw_data['list']
        if('outer_ids' in raw_data):
            self.outer_ids += raw_data['outer_ids']


    def get_path_name(self, name, type_name):
        return osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', name +'-'+ type_name +'.pt')



    def add_line_latentspace(self,lineTrainer):
        name = lineTrainer.name
        print(self.raw_data)
        for line in self.raw_data:
            x, edge_index = self.create_line_graph(line['points'])
            z = lineTrainer.encodeLineVector(x, edge_index)
            if 'latent_vector' not in line.keys():
                line['latent_vector'] = {}
            if name not in line['latent_vector']:
                line['latent_vector'][name] = z
                print("added latent space vectors to Graph Handler")

        print("known lines:", len(self.raw_data))
        
    #def add_line_drawdata(self, data, lineTrainer):
    #    name = lineTrainer.name
    #    for line in data:
    #        x, edge_index = self.create_line_graph(line['points'])
    #        z = lineTrainer.encodeLineVector(x, edge_index)
    #        if 'latent_vector' not in line.keys():
    #            line['latent_vector'] = {}
    #        line['latent_vector'][name] = z
    #        self.raw_data.append(line)

    #    print("added latent space vectors to Graph Handler", len(data), "now:", len(self.raw_data))



    def save_line_training_data(self, name):
        data_list = []
        print("raw data:", self.raw_data)
        for line in self.raw_data:
            x, edge_index = self.create_line_graph(line['points'])
            data = Data(x=x, edge_index=edge_index, scale=line['scale'], rotation=line['rotation'])
            data_list.append(data)

        print(data_list)
        torch.save({ 'line': data_list }, self.get_path_name(name, 'line'))



    def create_line_graph(self, points): #graph per stroke
        connections = []
        hidden_states = []
        for i in range(1,len(points)):
            connections.append([i-1,i])
            connections.append([i,i-1])

            connections.append([0,i])
            connections.append([i,0])

        if config['double_ended'] :
            for i in range(0,len(points)-1):
                connections.append([len(points)-1,i])
                connections.append([i,len(points)-1])

        edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()

        for point in points:
            hidden_states.append([point['x']-points[0]['x'], point['y']-points[0]['y']])

        x = torch.tensor(hidden_states, dtype=torch.float)

        return x, edge_index



    def create_pattern_graph(self, pred_id, ids, latent_name):
        print("create pattern graph - ", pred_id, "on", ids)
        hidden_states = []
        
        x_coords = [self.raw_data[i]['points'][0]['x'] for i in ids]
        y_coords = [self.raw_data[i]['points'][0]['y'] for i in ids]
        print(x_coords, y_coords)
        center_point = [ (max(x_coords)+min(x_coords))/2, (max(y_coords)+min(y_coords))/2 ]

        for i in ids:
            hid = self.assemble_node_hidden_state(i, center_point, self.raw_data[i]['latent_vector'][latent_name])
            hidden_states.append(hid)

        if pred_id is not None:
            ground_truth = self.assemble_node_hidden_state(pred_id, center_point, self.raw_data[pred_id]['latent_vector'][latent_name])
        else:
            ground_truth = None
            print("no prediction id given. is this correct?", pred_id)

        #fully connect der nähesten k nodes
        connections = torch.combinations(torch.arange(0,len(ids), dtype=torch.int64))

        edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
        x = torch.stack(hidden_states, dim=0) #vllt nochmal checken ob der jetzt "richtig rum" ist

        return x, edge_index, ground_truth
    

    #center_point ist der punkt, der den referenzpunkt für das datensample darstellt
    def assemble_node_hidden_state(self, current_id, center_point, current_latent_vector):
        lat_vec = current_latent_vector
        posX = self.raw_data[current_id]['points'][0]['x'] - center_point[0] #delta zur main node position
        posY = self.raw_data[current_id]['points'][0]['y'] - center_point[1]
        # versuch das relativ anzugeben im bezug zur ... maxdist?
        posX = posX / config['max_dist']
        posY = posY / config['max_dist']
        
        rot = self.raw_data[current_id]['rotation']
        scale = self.raw_data[current_id]['scale']
        #print(lat_vec.size(), posX, posY, rot, scale)
        
        return torch.cat( (torch.tensor( [posX, posY, rot, scale], dtype=torch.float), lat_vec), 0)


    def sample_graph(self, pred_id, latent_name, max_dist=config['max_dist']):
        dists = self.get_distance_matrix()
        max_dist = config['max_dist']
        dists = dists * (dists < max_dist)
        sorted_dists, indices = torch.sort(dists)

        current = sorted_dists[pred_id]
        current_ids = indices[pred_id]
        not_zero = current!=0
        current = current[not_zero]
        ids = current_ids[not_zero]
        
        x, edge_index, ground_truth = self.create_pattern_graph(pred_id, ids, latent_name)

        return x, edge_index, ground_truth
    
    def sample_complete_graph(self, latent_name):
        samples = []
        for i in range(len(self.raw_data)):
            x, edge_idx = self.sample_graph(i, latent_name)
            samples.append({ 
                "x":x, 
                "edge_index":edge_idx,
                "pred_id": i
            })
        return samples


    def save_pattern_training_data(self, latent_name, name):
        data_list = []

        for i in range(len(self.raw_data)):
            x, edge_idx, ground_truth = self.sample_graph(i, latent_name)     
            data = Data(x=x, edge_index=edge_idx, y=ground_truth)
            data_list.append(data)
                
        print("saving dataset of length ", len(data_list))
        torch.save({ 'pattern': data_list }, self.get_path_name(name, 'pattern'))



    def getAbsolutePosition(reference_id):
        return self.raw_data[reference_id]['points'][0]['x'], self.raw_data[reference_id]['points'][0]['y']

    
        


    def decompose_node_hidden_state(z):
        info = {
            "posX" : z[0].item(),
            "posY" : z[1].item(),
            "rot" : z[2].item(),
            "scale" : z[3].item(),
            "latVec" : z[4:]
        }
        
        return info


    def get_distance_matrix(self):

        dist_list = []
        for line1 in self.raw_data:
            dist_list.append( [ line1['points'][0]['x'], line1['points'][0]['y'] ] )

        dist_tensor = torch.tensor(dist_list).float()
        return torch.cdist(dist_tensor, dist_tensor, p=2)







