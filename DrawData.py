import torch
import os
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GAE, GCNConv
from config import config
from line import Line
from itertools import product
import torch_geometric.transforms as T
import random


""" #Loads the dataset and handles delivery
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
        print(osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', self.name +'.pt'))

        data_list = complete_data[self.level]
        print("Processing Dataset of length: ", len(data_list))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
 """

class GraphDatasetHandler():
    def __init__(self, name, level):
        self.name = name
        self.level = level

    def save_data(self, data):

        file_path = osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', self.name +'-'+ self.level +'.pt')
        if osp.exists(file_path):
            print("removing old data")
            os.remove(file_path)
        

        print("PREPARING DATA FOR SAVING")
        self.data = data
        self.original_data = data.copy()
        self.config = config


        if self.level == "pattern":


            if self.config['jitter_pattern'] > 0 :
                print("jittering pattern data")

          



        if self.level == "line":

            if self.config['jitter_line'] > 0:
                print("jittering line data")
                for data in self.original_data:
                    for i in range(self.config['jitter_line_additional_lines']):
                        new_data = data.clone()
                        new_data.x = data.x + torch.randn(data.x.size()) * self.config['stroke_normalizing_size'] * self.config['jitter_line']
                        self.data.append(new_data)

            

        print("SAVING DATA", len(self.data), "originals:", len(self.original_data))
        torch.save(self, osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', self.name +'-'+ self.level +'.pt'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value



    def get_random_original_item(self):
        return self.original_data[random.randint(0, len(self.original_data)-1)]
    
    def get_random_item(self):
        return self.data[random.randint(0, len(self.data)-1)]
        #return self.data[26]

    @property
    def num_features(self):
        return self.data[0].num_features
    

    @classmethod
    def load_data(cls, name, level):
        print("LOADING DATA")
        return torch.load(osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', name +'-'+ level +'.pt'))

class GraphHandler:
    def __init__(self):
        self.clear()

    def clear(self):
        self.lines = []
        self.gen_step = []
        self.pattern_trainer = None
        self.line_trainer = None

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

            print("lines added", self.lines[-1])

    def set_default_trainers(self, pattern_trainer=None, line_trainer=None):
        self.pattern_trainer = pattern_trainer
        self.line_trainer = line_trainer
    
    def init_random(self, num_samples, lineTrainer=None):
        #self.raw_data = []
        if lineTrainer is None:
            lineTrainer = self.line_trainer
        
        for i in range(num_samples):
            z = lineTrainer.randomInitPoint()
            line = GraphHandler.decompose_node_hidden_state(z, lineTrainer)
            line.position_type = "absolute"
            line.position['x'] *= config['max_dist']
            line.position['y'] *= config['max_dist']
            self.lines.append(line)
            #lines.append(prediction2obj(line, lineTrainer))

    def init_original(self, lineTrainer=None, patternTrainer=None):
        if lineTrainer is None:
            lineTrainer = self.line_trainer
        if patternTrainer is None:
            patternTrainer = self.pattern_trainer

        data = patternTrainer.dataset.get_random_item()
        self.test_data = data
        #pred = self.decompose_node(data.x)
        #self.lines.append(pred)

        noisy_data = data.y.clone() + torch.randn(data.y.size()) * 0.01
        ground_truth = self.decompose_node(noisy_data)
        ground_truth.update_position_from_reference({"x":0, "y":0})
        ground_truth.is_fixed = True
        self.lines.append(ground_truth)
        for i in range(data.x.size()[0]):
            n = self.decompose_node(data.x[i])
            n.update_position_from_reference({"x":0, "y":0})
            n.is_fixed = True
            self.lines.append(n)

        #add 5 random lines
        for i in range(1):
            z = self.line_trainer.randomInitPoint()
            line = GraphHandler.decompose_node_hidden_state(z, self.line_trainer)
            line.update_position_from_reference({"x":0, "y":0})
            self.lines.append(line)

        

        

        

    def calculate_gen_step(self):
        
        self.gen_step = []
        adaption_rate = 0.1
        


        for i in range(len(self.lines)):
            if self.lines[i].adaption_rate == 0.5:
                firsttime = True
            else:
                firsttime = False

            #self.lines[i].dropout *= 0.995
            self.lines[i].dropout = 0.1
            self.lines[i].adaption_rate *= 0.995
            self.lines[i].adaption_rate = max(self.lines[i].adaption_rate, 0.2)

            data = self.sample_graph(i, node_dropout=self.lines[i].dropout)
            
            if data is not None:
                z = self.pattern_trainer.predict(data.x, data.edge_index, data.target_point)
                previous_line_z = self.lines[i].get_pattern_z(center_position=data.center_point)
                if self.lines[i].is_fixed is False:
                

                    adapted_z = previous_line_z + self.lines[i].adaption_rate * (z - previous_line_z)
                    if firsttime:
                        adapted_z = z
                    
                    line = self.decompose_node(adapted_z)
                    
                    
                    
                    line.update_position_from_reference(data.center_point)
                    line.dropout = self.lines[i].dropout
                    

                    self.gen_step.append(line)
                else:
                    self.gen_step.append(self.lines[i])

       
           

        return self.gen_step
    
    def apply_gen_step(self):
        self.lines = [line for line in self.gen_step]
    
    def get_path_name(self, name, type_name):
        return osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', name +'-'+ type_name +'.pt')


    
    def add_line_latentspace(self,lineTrainer=None):
        if lineTrainer is None:
            lineTrainer = self.line_trainer
        name = lineTrainer.name
        
        for line in self.lines:
            x, edge_index = line.create_line_graph()
            z = lineTrainer.encodeLineVector(x, edge_index)
            line.add_latent_vector(z, name)

        print("known lines:", len(self.lines))
        
    
    def save_line_training_data(self, name=None, lineTrainer=None):
        if lineTrainer is None:
            lineTrainer = self.line_trainer
        if name is None:
            name = lineTrainer.name

        data_list = []
        
        for line in self.lines:
            print("LINE", line)
            x, edge_index = line.create_line_graph()
            data = Data(x=x, edge_index=edge_index, scale=line.scale, rotation=line.rotation, position=line.position)
            data_list.append(data)

        line_data = GraphDatasetHandler(name, "line")
        line_data.save_data(data_list)



    



    
    def create_pattern_graph(self, pred_id, ids, latent_name=None):
        if len(ids) == 0:
            raise ValueError("no ids given to create pattern graph")


        hidden_states = []

        if latent_name is None:
            latent_name = self.pattern_trainer.name

        if pred_id is not None and pred_id in ids:
            print("removed prediction id from ids", pred_id, ids)
            ids = ids[ids != pred_id]

        centers_X = [self.lines[i].position['x'] for i in ids]
        centers_Y = [self.lines[i].position['y'] for i in ids]
        center_point = { "x": sum(centers_X)/len(centers_X), "y": sum(centers_Y)/len(centers_Y) }
        

        #position of each node
        pos = torch.tensor([[self.lines[i].position['x'],self.lines[i].position['y']] for i in ids], dtype=torch.float)

        for i in ids:
            hid = self.assemble_node_hidden_state(i, center_point, latent_name)
            hidden_states.append(hid)

        if pred_id is not None:
            ground_truth = self.assemble_node_hidden_state(pred_id, center_point, latent_name)
        else:
            ground_truth = None
            #raise ValueError("no prediction id given. is this correct?", pred_id)

        #fully connect der nähesten k nodes
        #connections = torch.combinations(torch.arange(0,len(ids), dtype=torch.int64))
        #edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()

        
        x = torch.stack(hidden_states, dim=0) #vllt nochmal checken ob der jetzt "richtig rum" ist

        target_point = ground_truth[:2].unsqueeze(0) if ground_truth is not None else None
        
        data = Data(x=x, y=ground_truth, center_point=center_point, pos=pos, target_point=target_point)
        
        data = T.Delaunay()(data)
        if data.face is not None:
            data = T.FaceToEdge()(data)
        #transform = T.Compose([T.ToUndirected()])
        #data = transform(data)
        
        return data
    

    
    #center_point ist der punkt, der den referenzpunkt für das datensample darstellt
    def assemble_node_hidden_state(self, current_id, center_point, latent_name=None):
        if latent_name is None:
            latent_name = self.pattern_trainer.name

        line = self.lines[current_id]
        
        lat_vec = line.latent_vectors[latent_name]

        if line.position_type == "absolute":
            delta_posX = line.position['x'] - center_point['x'] #delta zur main node position
            delta_posY = line.position['y'] - center_point['y']
        else:
            raise ValueError("Relative position in line while assembling node hidden state")

        # versuch das relativ anzugeben im bezug zur ... maxdist?
        delta_posX = delta_posX / config['max_dist']
        delta_posY = delta_posY / config['max_dist']
        
        rot = line.rotation
        scale = line.scale

        
        return torch.cat( (torch.tensor( [delta_posX, delta_posY, rot, scale], dtype=torch.float), lat_vec), 0)


    
    def sample_graph(self, pred_id, latent_name=None, max_dist=config['max_dist'], include_pred_id=False, with_combinations=False, node_dropout=0.0):
        if latent_name is None:
            latent_name = self.pattern_trainer.name

        dists = self.get_distance_matrix()
        dists = dists * (dists < max_dist)
        sorted_dists, indices = torch.sort(dists)

        current = sorted_dists[pred_id]
        current_ids = indices[pred_id]
        not_zero = current!=0
        current = current[not_zero]
        ids = current_ids[not_zero]

        if include_pred_id:
            ids = torch.cat([torch.tensor([pred_id]), ids])
            pred_id = None

        if len(ids) == 0:
            return None
        
        if node_dropout > 0 and len(ids) > 1:
            #drop node_dropout% of the nodes
            keepers = []
            for i in range(len(ids)):
                if random.random() > node_dropout:
                    keepers.append(ids[i])

            if len(keepers) == 0:
                keepers.append(random.choice(ids))

            print("dropping nodes.", len(keepers), "left from", len(ids))
            ids = keepers
            
        
        if with_combinations:
            combinations = torch.tensor(list(product([False, True], repeat=len(ids))))
            #combinations = combinations[combinations.sum(dim=1) <= 3]
            combinations = combinations[combinations.any(dim=1)]

            data_list = []

            for combo in combinations:
                combo_ids = ids[combo]
                data_list.append(self.create_pattern_graph(pred_id, combo_ids, latent_name))
            return data_list

        else:    

            return self.create_pattern_graph(pred_id, ids, latent_name)
    
    
    def save_pattern_training_data(self, latent_name=None, name=None):
        if latent_name is None:
            latent_name = self.pattern_trainer.name

        if name is None:
            name = latent_name

        data_list = []

        for i in range(len(self.lines)):
            if config['create_pattern_combinations']:
                x_list = self.sample_graph(i, latent_name, with_combinations=True)
                if x_list is not None:
                    data_list.extend(x_list)
            else:
                data = self.sample_graph(i, latent_name)  
                if data is not None:
                    data_list.append(data)

        
            
                
        print("saving dataset of length ", len(data_list))
        pattern_data = GraphDatasetHandler(name, "pattern")
        pattern_data.save_data(data_list)
        





    
    def get_distance_matrix(self):

        dist_list = []
        for line1 in self.lines:
            if line1.position_type == "absolute":
                dist_list.append( [ line1.position['x'], line1.position['y'] ] )
            else:
                raise ValueError("prediction with relative position in lines")

        dist_tensor = torch.tensor(dist_list).float()
        return torch.cdist(dist_tensor, dist_tensor, p=2)
        


    def decompose_node(self, z, line_trainer=None):
        if line_trainer is None:
            line_trainer = self.line_trainer
        return GraphHandler.decompose_node_hidden_state(z, line_trainer)
    
    @staticmethod
    def decompose_node_hidden_state(z, line_trainer):

        if z.dim() > 1:
            # Break apart tensor into list of 1D tensors
            z_list = []
            for i in range(z.size(0)):
                z_list.append(z[i])
            return [GraphHandler.decompose_node_hidden_state(z_single, line_trainer) for z_single in z_list]

        posX, posY, rot, scale, latVec = GraphHandler.extract_parts(z, line_trainer)
        points = line_trainer.decode_latent_vector(latVec)
        l = Line(points, scale, rot, position={"x":posX, "y":posY}, position_type="relative")
        l.add_latent_vector(latVec, line_trainer.name)
        return l
    
    @staticmethod
    def extract_parts(z, lineTrainer):
        posX = z[0].item()
        posY = z[1].item()
        rot = z[2].item()
        scale = z[3].item()
        latVec = z[4:]

        return posX, posY, rot, scale, latVec
    
    



    







