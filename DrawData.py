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
from sklearn.cluster import DBSCAN
import numpy as np


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

    def save_data(self, data, line_positions=None):

        file_path = osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', self.name +'-'+ self.level +'.pt')
        if osp.exists(file_path):
            print("removing old data")
            os.remove(file_path)
        

        print("PREPARING DATA FOR SAVING")
        self.data = data
        self.original_data = data.copy()
        self.config = config
        self.line_positions = line_positions


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
        torch.save(
            self,
            osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', self.name +'-'+ self.level +'.pt'),
            _use_new_zipfile_serialization=True  # Use newer format
        )

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
        try:
            # Try loading with weights_only=False to handle the new format
            return torch.load(
                osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', name +'-'+ level +'.pt'),
                weights_only=False
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            # If that fails, try the old format with map_location
            return torch.load(
                osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', name +'-'+ level +'.pt'),
                map_location=torch.device('cpu')
            )

class GraphHandler:
    def __init__(self):
        self.clear()

    def clear(self):
        self.lines = []
        self.gen_step = []
        self.ghost_lines = []
        self.original_lines = []
        self.pattern_trainer = None
        self.line_trainer = None

    def init_lines(self,data):
        # list with points, scale, rotation
        # points: list of dicts with x,y like {'x': 610, 'y': 325}
        # scale: float
        # rotation: float
        self.lines = []
        self.add_lines(data)


    def calculate_original_lines(self):
        original_lines = self.line_trainer.dataset.original_data
        for o in original_lines: 
            l = Line(o.x, o.scale, o.rotation, o.position)
            x, edge_index = l.create_line_graph()
            z = self.line_trainer.encodeLineVector(x, edge_index)
            l.add_latent_vector(z, self.line_trainer.name)
            self.original_lines.append(l)
        print("original lines", len(self.original_lines))
        
            
    def add_lines(self, data):
        for line in data:
            position = line['position'] if 'position' in line else None
            self.lines.append(Line(line['points'], line['scale'], line['rotation'], position=position))


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

    def init_noisy_line_at_position(self, position, lineTrainer=None, patternTrainer=None, noise_level=0.01):
        if lineTrainer is None:
            lineTrainer = self.line_trainer
        if patternTrainer is None:
            patternTrainer = self.pattern_trainer
        
        sample = patternTrainer.dataset.get_random_item()
        noisy_data = sample.y.clone() + torch.randn(sample.y.size()) * noise_level
        ground_truth = self.decompose_node(noisy_data)
        ground_truth.update_position_from_reference(position)
        ground_truth.is_fixed = True
        return ground_truth

    def init_original(self, lineTrainer=None, patternTrainer=None, noise_level=0.01):
        if lineTrainer is None:
            lineTrainer = self.line_trainer
        if patternTrainer is None:
            patternTrainer = self.pattern_trainer

        distance = 50
        #for i in range(3):
        for i in range(1):
            for j in range(1):

                data = patternTrainer.dataset.get_random_item()
                self.test_data = data
                #pred = self.decompose_node(data.x)
                #self.lines.append(pred)
                reference_position = {"x":i*(distance + config['max_dist']*3), "y":j*(distance + config['max_dist']*3)}

                noisy_data = data.y.clone() + torch.randn(data.y.size()) * noise_level
                ground_truth = self.decompose_node(noisy_data)
                ground_truth.update_position_from_reference(reference_position)
                ground_truth.is_fixed = True
                self.lines.append(ground_truth)
                for i in range(data.x.size()[0]):
                    n = self.decompose_node(data.x[i])
                    n.update_position_from_reference(reference_position)
                    n.is_fixed = True
                    self.lines.append(n)

    def random_fill(self, fieldX=800, fieldY=800, retry_count=300, lineTrainer=None, patternTrainer=None, noise_level=0.01):
        if lineTrainer is None:
            lineTrainer = self.line_trainer
        if patternTrainer is None:
            patternTrainer = self.pattern_trainer

        count = retry_count

        max_dist = config['max_dist']
        
        while count > 0:
            sample = patternTrainer.dataset.get_random_item()
            noisy_data = sample.y.clone() + torch.randn(sample.y.size()) * noise_level
            ground_truth = self.decompose_node(noisy_data)
            ground_truth.update_position_from_reference({"x":random.randint(0, fieldX), "y":random.randint(0, fieldY)})
            ground_truth.is_fixed = True
            self.lines.append(ground_truth)

            dist_matrix = self.get_distance_matrix()
            new_dist = dist_matrix[-1][:-1]
            
            print("new_dist", new_dist)
            # Check if any of the distances in new_dist is smaller than max_dist
            if any(d < max_dist for d in new_dist) and len(self.lines) > 1:
                print("At least one distance is smaller than max_dist.")
                self.lines.pop(-1)
                count -= 1

            print("retry count", count)




        

        

    def handle_ghost_lines(self):
        
        print("lines before handling ghost lines", len(self.lines), "|ghosts:", len(self.ghost_lines))
        diff_threshold = 1
        
        
        #for line in self.ghost_lines:
            #print(line.get_pattern_z())
        iterations = []

         
        for i, fixed_line in enumerate(self.lines):
            
            iteration_line = []
            #add original line to the mean mix
            iteration_line.append(fixed_line.get_pattern_z(center_position=fixed_line.position))
          

            #add ghost lines that are close to the original line to the mean mix
            for ghost_line in self.ghost_lines:
                ghost_line_z = ghost_line.get_pattern_z(center_position=fixed_line.position)

                diff = torch.sum(torch.abs(torch.sub(iteration_line[0], ghost_line_z)))
                if diff < diff_threshold:
                    iteration_line.append(ghost_line_z)
                
            
            averaged_line_z = torch.stack(iteration_line, dim=0).mean(dim=0)

            
            averaged_line = self.decompose_node(averaged_line_z)
            averaged_line.update_position_from_reference(fixed_line.position)
            averaged_line.is_fixed = True
          

            iterations.append(averaged_line)


        

        self.lines = iterations
        print("lines after handling ghost lines", len(self.lines))

        
                    
    def remove_duplicate_lines(self):
        message = ""
        i = 0
        while i < len(self.lines):
            j = i + 1
            while j < len(self.lines):
                if self.lines[i].diff(self.lines[j]) < 10:  # Using same threshold as in calculate_gen_step
                    print("removing duplicate line", i, j)
                    message += "removing duplicate line " + str(i) + " " + str(j) + "\n"
                    self.lines.pop(j)
                else:
                    j += 1
            i += 1

        if message:
            return message
        else:
            return None
        
    def get_closest_original_line(self, line):
        closest_line = None
        closest_diff = float('inf')
        for original_line in self.original_lines:
            diff = line.latent_line_diff(original_line)
            if diff < closest_diff:
                closest_diff = diff
        
        return closest_line, closest_diff
        
    def reject_abnormal_lines(self):
        threshhold = 0.8
        accepted_lines = []
        nr_lines = len(self.ghost_lines)

        print("rejecting abnormal lines. Current lines:", len(self.ghost_lines))
        for line in self.ghost_lines:
            closest_original_line, closest_diff = self.get_closest_original_line(line)
            print("DIFF", closest_diff)
            if closest_diff < threshhold:
                accepted_lines.append(line)
                    

        self.ghost_lines = accepted_lines
        if len(accepted_lines) < nr_lines:
            print("not all lines were accepted. Rejecting", nr_lines-len(accepted_lines), "lines")
            return "not all lines were accepted. Rejecting " + str(nr_lines-len(accepted_lines)) + " lines"
        else:
            return None
        

   

    def calculate_gen_step(self):
        
        self.gen_step = []
        diff_threshold = 0.01

        if not hasattr(self, "ghost_lines"):
            self.ghost_lines = []
        


        for i in range(len(self.lines)):
            if self.lines[i].is_fixed is False:

                #adapted_z = previous_line_z + self.lines[i].adaption_rate * (z - previous_line_z)

                #print(len(self.lines))
                
                
                self.lines[i].dropout = 0

                
                
                

                data = self.sample_graph(i, node_dropout=self.lines[i].dropout, with_combinations=True)
                if data is None:
                    print("line out of reference reach. Setting a new line here")
                    new_line = self.init_noisy_line_at_position(self.lines[i].position)
                    self.lines[i] = new_line
                    continue

                if not hasattr(self.lines[i], "used_ids"):
                    self.lines[i].used_ids = data[0].used_ids
                    data = data[0]
                else:
                    #print(len(data), "line used ids", [ d.used_ids for d in data ])
                    
                    
                    data_filtered = [d for d in data if all(used_id in d.used_ids for used_id in self.lines[i].used_ids)]
                    if len(data_filtered) > 0:
                        #print("choosing options", [d.used_ids for d in data_filtered])
                        if self.lines[i].stopped: 
                            if len(data_filtered) > 1:
                                print("choosing second option", data_filtered[1].used_ids)
                                
                                self.ghost_lines.append(self.lines[i])
                                print("creating ghost line. Now:", len(self.ghost_lines))
                                data = data_filtered[1]
                                
                            else:
                                print("DONE")
                                return False
                                data = data_filtered[0]
                        else:
                            
                            data = data_filtered[0]
                    else:

                        print("no data found for old ids (went out of reach of the old id set)", self.lines[i].used_ids)
                        data = data[0]
                        print("using new data", data.used_ids)
                    
                
                #flexi_rate = (self.lines[i].adaption_rate/len(self.lines[i].used_ids))
                flexi_rate = self.lines[i].adaption_rate
                #print("data", data.used_ids, self.lines[i].used_ids, flexi_rate)
                previous_line_z = self.lines[i].get_pattern_z(center_position=data.center_point)
                
                
                z = self.pattern_trainer.predict(data.x, data.edge_index, data.target_point)
                adapted_z = previous_line_z + flexi_rate * (z - previous_line_z)       
                        
                    
                    
                line = self.decompose_node(adapted_z)
                line.update_position_from_reference(data.center_point)

                line.used_ids = data.used_ids
                line.adaption_rate = flexi_rate
               
               
                if self.lines[i].pos_diff(line) < diff_threshold:
                    print("pos diff reached:", self.lines[i].pos_diff(line))
                    line.stopped = True
                else:
                    line.stopped = False

                if not hasattr(self.lines[i], "history"):
                    self.lines[i].history = []
                    self.lines[i].last_history = []
                    self.lines[i].wiggle_count = 0
                self.lines[i].history.append(line.used_ids)
                if len(self.lines[i].history) > 50:
                    self.lines[i].history.pop(0)
                line.history = self.lines[i].history
                line.last_history = self.lines[i].last_history
                line.wiggle_count = self.lines[i].wiggle_count

                if len(line.history) > 1 and line.history[-1] != line.history[-2]:
                    count1 = line.history.count(line.history[-1])
                    count2 = line.history.count(line.history[-2])
                    unique_elements = len([list(x) for x in set(tuple(l) for l in line.history)])
                    print("history", unique_elements, "count of last:", count1, count2, line.last_history)
                    if len(line.last_history) > 0:
                        wiggle_diff = (line.last_history[0]-count2 + line.last_history[1]-count1)
                        print("wiggle diff", wiggle_diff)
                        if wiggle_diff < 2:
                            line.wiggle_count += 1
                            print("WIGGLE DETECTED", line.wiggle_count)
                            line.adaption_rate *= 0.9
                            if line.wiggle_count > 20:
                                print("WIGGLE STOP")
                                line.stopped = False
                                #print("adding wiggle line to ghost lines")
                                #self.ghost_lines.append(line)
                                continue
                            #line.stopped = True
                    line.last_history = [count1, count2]
                    

                self.gen_step.append(line)
            else:
                self.gen_step.append(self.lines[i])

       
        if all(line.is_fixed for line in self.lines):
            print("ALL LINES FIXED")
            return False
           

        return True
    
    def apply_gen_step(self):
        self.lines = [line for line in self.gen_step]
        
    def start_new_line(self):


        #rejection funktioniert nicht gut rein über den latenten vektor
        self.reject_abnormal_lines()

        #for ghost_line in self.ghost_lines:
        #    self.lines.append(ghost_line)

        
        #elf.lines = self.cluster_and_average(func1=self.find_latent_clusters, func2=self.find_position_clusters, eps1=0.2, eps2=100, message="latent first")
        #self.lines = self.cluster_and_average(func1=self.find_position_clusters, func2=self.find_latent_clusters, eps1=30, eps2=2, message="pos first")

        #reject again if an average makes no sense
        self.reject_abnormal_lines()

        for line in self.lines:
            if line.is_fixed is False:
                self.ghost_lines.append(line)
                self.lines.remove(line)
            #line.is_fixed = True

        z = self.line_trainer.randomInitPoint()
        line = GraphHandler.decompose_node_hidden_state(z, self.line_trainer)
        line.update_position_from_reference({"x":random.randint(0, 800), "y":random.randint(0, 800)})
        self.lines.append(line)
        #self.ghost_lines = []
        
    
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

        if len(ids) == 0:
            raise ValueError("No IDs left to create pattern graph", pred_id, ids)
        
        centers_X = [self.lines[i].position['x'] for i in ids]
        centers_Y = [self.lines[i].position['y'] for i in ids]
        center_point = { "x": sum(centers_X)/len(centers_X), "y": sum(centers_Y)/len(centers_Y) }
        ###### hier fliegt der divide by zero fehler
        

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
        if with_combinations and node_dropout > 0:
            print("node dropout and combinations not supported")
            exit()
        
        if latent_name is None:
            latent_name = self.pattern_trainer.name

        eps = 0.05

        dists = self.get_distance_matrix()
        dists = dists * (dists < max_dist)
        
        sorted_dists, indices = torch.sort(dists)

        current = sorted_dists[pred_id]
       
        current_ids = indices[pred_id]
        
        not_zero = current > eps
        current = current[not_zero]
        ids = current_ids[not_zero]
       
        if pred_id in ids:
            print("!! pred_id in ids", pred_id, ids)
            idx = (ids == pred_id).nonzero(as_tuple=True)[0]
            ids = torch.cat([ids[:idx], ids[idx+1:]])
            current = torch.cat([current[:idx], current[idx+1:]])

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
            combinations = combinations.flip(dims=[1])  # Reverse each combination

            data_list = []

            for combo in combinations:
                combo_ids = ids[combo]
                if len(combo_ids) == 1 and combo_ids[0] == pred_id:
                    print("skipping combination with only pred_id", combo_ids, pred_id)
                    continue
                if pred_id in combo_ids:
                    print("ERROR: pred_id in combo_ids", combo_ids, pred_id)
                    exit()
                data = self.create_pattern_graph(pred_id, combo_ids, latent_name)
                data.used_ids = combo_ids.tolist()
                data_list.append(data)
            
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

        positions = [line.position for line in self.lines]
            
                
        print("saving dataset of length ", len(data_list))
        pattern_data = GraphDatasetHandler(name, "pattern")
        pattern_data.save_data(data_list, positions)
        





    
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
    

    def cluster_and_average(self, func1, func2, eps1, eps2, message):
        lines = self.lines
        print("CLUSTERING", message, len(lines), "lines")
        final_lines = []
        if(len(lines) == 0):
            return lines
        clusters_position = func1(lines, eps1)
        for cluster_label, lines_in_cluster in clusters_position.items():
            if cluster_label == -1:
                
                final_lines.extend(lines_in_cluster)
            else:
              
                if len(lines_in_cluster) > 1:
                    cluster_latent = func2(lines_in_cluster, eps2)
                    for cluster_label_latent, lines_in_cluster_latent in cluster_latent.items():
                        if cluster_label_latent == -1:
                           
                            final_lines.extend(lines_in_cluster_latent)
                        else:
                            
                            averaged_latent = GraphHandler.average_latent_vectors(lines_in_cluster_latent, lines_in_cluster_latent[0].position)
                            line = GraphHandler.decompose_node_hidden_state(averaged_latent, self.line_trainer)
                            line.update_position_from_reference(lines_in_cluster_latent[0].position)
                            final_lines.append(line)
                            print(message, "averaged lines", len(lines_in_cluster_latent))
        print("FINAL LINES", message, len(final_lines))

        return final_lines
                            
        # clusters_position, clusters_latent = Line.find_position_clusters(lines)
        
        # for cluster_label, lines_in_cluster in clusters_position.items():
        #     if cluster_label == -1:
        #         print(f"Noise cluster: {len(lines_in_cluster)} lines")
        #     else:
        #         print(f"Cluster {cluster_label}: {len(lines_in_cluster)} lines")
        #         center_position = lines_in_cluster[0].position
        #         zs = []
        #         for line in lines_in_cluster:
        #             if line not in clusters_latent[cluster_label]:
        #                 print(f"Line {line.id} not in latent cluster {cluster_label}")
        #                 continue
        #             z = line.get_pattern_z(center_position=center_position)
        #             zs.append(z)
        #         zs = torch.mean(torch.stack(zs), dim=0)
        #         print(zs.shape)
               
        #         exit()
        
        return lines
    
    @staticmethod
    def average_latent_vectors(lines, center_position):
        zs = []
        for line in lines:
            z = line.get_pattern_z(center_position=center_position)
            zs.append(z)
        return torch.mean(torch.stack(zs), dim=0)

    @staticmethod
    def find_position_clusters(lines, eps):
        positions = np.array([[line.position['x'], line.position['y']] for line in lines])
        dbscan_position = DBSCAN(eps, min_samples=2)
        labels_position = dbscan_position.fit_predict(positions)
        
        # Group lines by cluster label
        clusters_position = {}
        for idx, label in enumerate(labels_position):
            if label not in clusters_position:
                clusters_position[label] = []
            clusters_position[label].append(lines[idx])
        
        # Now clusters[label] contains the list of lines in that cluster
        # Note: label -1 means noise/outliers
        #print("Clustered lines by position:", {label: len(clusters_position[label]) for label in clusters_position})
       
        return clusters_position
    
    @staticmethod
    def find_latent_clusters(lines, eps):
        latent_vectors = np.array([line.get_latent_vector().detach().numpy() for line in lines])
        dbscan_latent = DBSCAN(eps, min_samples=2)
        labels_latent = dbscan_latent.fit_predict(latent_vectors)

        # nur latent vector gerade. sollte da scale und rotation rein?
        clusters_latent = {}
        for idx, label in enumerate(labels_latent):
            if label not in clusters_latent:
                clusters_latent[label] = []
            clusters_latent[label].append(lines[idx])
        
        # Now clusters[label] contains the list of lines in that cluster
        # Note: label -1 means noise/outliers
        #print("Clustered lines by latent:", {label: len(clusters_latent[label]) for label in clusters_latent})
       
        return clusters_latent

    # def get_total_tensor(self):
    #     total_position_points = self._points2Tensor()
    #     # Scale and rotate points
    #     points_tensor = self._points2Tensor()
        
    #     # Create rotation matrices
    #     theta = torch.tensor(self.rotation * 360 * torch.pi / 180, dtype=torch.float)
    #     rot_matrix = torch.tensor([
    #         [torch.cos(theta), -torch.sin(theta)],
    #         [torch.sin(theta), torch.cos(theta)]
    #     ])
        
    #     # Apply scale and rotation
    #     scaled_points = points_tensor * self.scale * config['max_dist']
    #     rotated_points = torch.matmul(scaled_points, rot_matrix.T)
        
    #     # Create position tensor of same shape as rotated_points and add it
    #     position_tensor = torch.tensor([self.position['x'], self.position['y']]).repeat(rotated_points.shape[0], 1)
    #     total_position_points = rotated_points + position_tensor

    #     return total_position_points
    
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
    
    



    







