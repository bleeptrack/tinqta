import torch
import os
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GAE, GCNConv
from config import config
from line import Line
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
        self.lines = []

    def init_lines(self,data):
        # list with points, scale, rotation
        # points: list of dicts with x,y like {'x': 610, 'y': 325}
        # scale: float
        # rotation: float
        self.lines = []
        self.add_lines(data)
            
    def add_lines(self, data):
        for line in data:
            position = line['position'] if 'position' in line else None
            self.lines.append(Line(line['points'], line['scale'], line['rotation'], position=position))


    
    def init_random(self, num_samples, lineTrainer):
        #self.raw_data = []
        
        for i in range(num_samples):
            z = lineTrainer.randomInitPoint()
            line = GraphHandler.decompose_node_hidden_state(z, lineTrainer)
            line.position_type = "absolute"
            line.position['x'] *= config['max_dist']
            line.position['y'] *= config['max_dist']
            self.lines.append(line)
            #lines.append(prediction2obj(line, lineTrainer))

    
            

    
    def get_path_name(self, name, type_name):
        return osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', name +'-'+ type_name +'.pt')


    
    def add_line_latentspace(self,lineTrainer):
        name = lineTrainer.name
        
        for line in self.lines:
            x, edge_index = line.create_line_graph()
            z = lineTrainer.encodeLineVector(x, edge_index)
            line.add_latent_vector(z, name)

        print("known lines:", len(self.lines))
        
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
        
        for line in self.lines:
            print("LINE", line)
            x, edge_index = line.create_line_graph()
            data = Data(x=x, edge_index=edge_index, scale=line.scale, rotation=line.rotation, position=line.position)
            data_list.append(data)

        
        torch.save({ 'line': data_list }, self.get_path_name(name, 'line'))



    



    
    def create_pattern_graph(self, pred_id, ids, latent_name):
        print("create pattern graph - ", pred_id, "on", ids)
        hidden_states = []

        if pred_id in ids:
            ids.remove(pred_id)
            print("removed prediction id from ids")
        
        x_max_coord = max([self.lines[i].get_absoulte_maxX() for i in ids])
        x_min_coord = min([self.lines[i].get_absoulte_minX() for i in ids])
        y_max_coord = max([self.lines[i].get_absoulte_maxY() for i in ids])
        y_min_coord = min([self.lines[i].get_absoulte_minY() for i in ids])
        print(f"center: ({x_min_coord}, {y_min_coord}), ({x_max_coord}, {y_max_coord})")
        center_point = [ (x_max_coord+x_min_coord)/2, (y_max_coord+y_min_coord)/2 ]
        

        for i in ids:
            hid = self.assemble_node_hidden_state(i, center_point, latent_name)
            hidden_states.append(hid)

        if pred_id is not None:
            ground_truth = self.assemble_node_hidden_state(pred_id, center_point, latent_name)
        else:
            ground_truth = None
            raise ValueError("no prediction id given. is this correct?", pred_id)

        #fully connect der nähesten k nodes
        connections = torch.combinations(torch.arange(0,len(ids), dtype=torch.int64))

        edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
        x = torch.stack(hidden_states, dim=0) #vllt nochmal checken ob der jetzt "richtig rum" ist

        return x, edge_index, ground_truth
    

    
    #center_point ist der punkt, der den referenzpunkt für das datensample darstellt
    def assemble_node_hidden_state(self, current_id, center_point, latent_name):

        line = self.lines[current_id]
        
        lat_vec = line.latent_vectors[latent_name]

        if line.position_type == "absolute":
            delta_posX = line.position['x'] - center_point[0] #delta zur main node position
            delta_posY = line.position['y'] - center_point[1]
        else:
            raise ValueError("Relative position in line while assembling node hidden state")

        # versuch das relativ anzugeben im bezug zur ... maxdist?
        delta_posX = delta_posX / config['max_dist']
        delta_posY = delta_posY / config['max_dist']
        
        rot = line.rotation
        scale = line.scale
        #print(lat_vec.size(), posX, posY, rot, scale)
        
        return torch.cat( (torch.tensor( [delta_posX, delta_posY, rot, scale], dtype=torch.float), lat_vec), 0)


    
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
    
    
    ###
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


    
    def save_pattern_training_data(self, latent_name, name=None):
        data_list = []
        if name is None:
            name = latent_name

        for i in range(len(self.lines)):
            x, edge_idx, ground_truth = self.sample_graph(i, latent_name)     
            data = Data(x=x, edge_index=edge_idx, y=ground_truth)
            data_list.append(data)
                
        print("saving dataset of length ", len(data_list))
        torch.save({ 'pattern': data_list }, self.get_path_name(name, 'pattern'))





    
    def get_distance_matrix(self):

        dist_list = []
        for line1 in self.lines:
            if line1.position_type == "absolute":
                dist_list.append( [ line1.position['x'], line1.position['y'] ] )
            else:
                raise ValueError("prediction with relative position in lines")

        dist_tensor = torch.tensor(dist_list).float()
        return torch.cdist(dist_tensor, dist_tensor, p=2)
        


    
    
    @staticmethod
    def decompose_node_hidden_state(z, line_trainer):
        posX = z[0].item()
        posY = z[1].item()
        rot = z[2].item()
        scale = z[3].item()
        latVec = z[4:]
        points = line_trainer.decode_latent_vector(latVec)
        l = Line(points, scale, rot, position={"x":posX, "y":posY}, position_type="relative")
        return l
    
    



    







