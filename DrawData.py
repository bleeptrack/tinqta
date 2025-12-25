import torch
import os
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GAE, GCNConv
from config import config
from line import Line
from itertools import product, combinations
import torch_geometric.transforms as T
import random
from sklearn.cluster import DBSCAN
import numpy as np
import math


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
        self.max_dist = None

    def get_file_path(name, level):
        return osp.join(osp.dirname(osp.realpath(__file__)), 'baseData', name +'-'+ level +'.pt')

    def save_data(self, data, line_positions=None, max_dist=None):

        file_path = GraphDatasetHandler.get_file_path(self.name, self.level)
        if osp.exists(file_path):
            print("removing old data")
            os.remove(file_path)
        

        print("PREPARING DATA FOR SAVING")
        self.data = data
        self.original_data = data.copy()
        self.config = config
        self.line_positions = line_positions
        self.max_dist = max_dist


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
        for line in self.lines:
            line.is_fixed = True
            line.stopped = True


    def calculate_original_lines(self):
        original_lines = self.line_trainer.dataset.original_data
        for o in original_lines: 
            l = Line(o.x, o.scale, o.rotation, o.position)
            x, edge_index = l.create_line_graph()
            z = self.line_trainer.encodeLineVector(x, edge_index)
            l.add_latent_vector(z, self.line_trainer.name)
            self.original_lines.append(l)
        print("original lines", len(self.original_lines))

    def load_template_from_pattern_trainer(self, pattern_trainer=None):
        if pattern_trainer is None:
            pattern_trainer = self.pattern_trainer
        self.lines = pattern_trainer.template_data["lines"].copy()
        print("template loaded with lines: ", len(self.lines))
       

    def calculate_line_thresholds(self):

        #calc delaugney über alle original lines
        
        # Extract positions from all original lines
        positions = []
        for line in self.original_lines:
            positions.append([line.position['x'], line.position['y']])
        
        # Convert to tensor
        pos_tensor = torch.tensor(positions, dtype=torch.float)
        print(f"Calculating Delaunay triangulation for {len(self.original_lines)} lines")
        print(f"Position tensor shape: {pos_tensor.shape}")
        
        # Create Data object with positions
        data = Data(pos=pos_tensor)
        
        # Apply Delaunay triangulation transform
        data = T.Delaunay()(data)
        
        # Convert faces to edges if faces exist
        if data.face is not None:
            data = T.FaceToEdge()(data)
            edge_index = data.edge_index
        else:
            print("No faces found in Delaunay triangulation")
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"Delaunay triangulation created {edge_index.shape[1]} edges")
        
        # Store the triangulation for further use
        delaunay_edges = edge_index
        line_positions = pos_tensor
        
        #für jede linie finde connections
        #finde den nähesten und den ähnlichsten
        #
        self.original_connection_info = {}
        self.avg_latent_diff = []
        self.avg_pos_diff = []

        for line_idx, line in enumerate(self.original_lines):
            print(f"Line {line_idx}: {line}")
            
            # Find all lines connected to this line via Delaunay triangulation
            connected_indices = []
            
            # Check both directions of edges (since edge_index contains [src, dst] pairs)
            for edge_idx in range(delaunay_edges.shape[1]):
                src, dst = delaunay_edges[:, edge_idx]
                
                # If this line is the source, add the destination
                if src.item() == line_idx:
                    connected_indices.append(dst.item())
                # If this line is the destination, add the source
                elif dst.item() == line_idx:
                    connected_indices.append(src.item())
            
            # Remove duplicates (in case of self-loops or multiple edges)
            connected_indices = list(set(connected_indices))
            
            print(f"  Connected to lines: {connected_indices}")
            
            # Get the actual connected line objects
            connected_lines = [self.original_lines[idx] for idx in connected_indices]
            
            # Calculate distances to connected lines
           
            for connected_idx, connected_line in zip(connected_indices, connected_lines):
                distance = line.pos_diff(connected_line)
                latent_diff = line.latent_line_diff(connected_line)
                print(f"    Line {connected_idx}: distance = {distance:.3f}, latent_diff = {latent_diff:.3f}")
                if line_idx not in self.original_connection_info:
                    self.original_connection_info[line_idx] = {}
                self.original_connection_info[line_idx][connected_idx] = {
                    "distance": distance,
                    "latent_diff": latent_diff
                }
                self.avg_latent_diff.append(latent_diff)
                self.avg_pos_diff.append(distance)
            
            print()  # Empty line for readability

        self.min_latent_diff = min(self.avg_latent_diff)
        self.min_pos_diff = min(self.avg_pos_diff)
        self.avg_latent_diff = float(sum(self.avg_latent_diff)) / len(self.avg_latent_diff)
        self.avg_pos_diff = float(sum(self.avg_pos_diff)) / len(self.avg_pos_diff)

        print("Average latent diff:", self.avg_latent_diff)
        print("Average pos diff:", self.avg_pos_diff)
        print("Min latent diff:", self.min_latent_diff)
        print("Min pos diff:", self.min_pos_diff)
        
       
    def match_two_lines(self, line1, line2):
        
        # Find the most similar line from original_lines in latent space for line
        closest_line1, closest_diff1, closest_idx1 = self.get_closest_original_line(line1)
        allowed_indices = [idx for idx in self.original_connection_info[closest_idx1].keys()]
        closest_line2, closest_diff2, closest_idx2 = self.get_closest_original_line(line2, allowed_indices)

        if closest_idx1 == closest_idx2:
            print("Lines have the same closest original line", line1.pos_diff(line2))
            return True
        
        # Check if closest_idx1 has connection info and if closest_idx2 is connected to it
        if (closest_idx1 in self.original_connection_info and 
            closest_idx2 in self.original_connection_info[closest_idx1]):
            info = self.original_connection_info[closest_idx1][closest_idx2]

            pos_diff = line1.pos_diff(line2)
            latent_diff = line1.latent_line_diff(line2)
            pos_percentage = pos_diff / info["distance"]
            latent_percentage = latent_diff / info["latent_diff"]

            if pos_diff < info["distance"] and latent_diff < info["latent_diff"]:
                if pos_percentage + latent_percentage < 1.1:
                #if pos_percentage < 0.5 and latent_percentage < 0.5:
                    print("Lines are similar:", pos_diff, "vs ", info["distance"]/2, "and", latent_diff, "vs", info["latent_diff"]/2, "and percentage", pos_percentage + latent_percentage)
                    return True
                else:
                    print(
                        f"CLOSE CALL: pos_diff = {pos_diff:.4f} (threshold: {info['distance']/2:.4f}, percentage: {pos_percentage:.4f}), "
                        f"latent_diff = {latent_diff:.4f} (threshold: {info['latent_diff']/2:.4f}, percentage: {latent_percentage:.4f})"
                    )
                    return False
            else:                    
                return False
        else:
            print("No connection info found for lines", closest_idx1, closest_idx2)
            return False
            
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
        
        # Get max_dist from pattern_trainer if available
        max_dist = self.pattern_trainer.max_dist if self.pattern_trainer else 150
        
        for i in range(num_samples):
            z = lineTrainer.randomInitPoint()
            line = GraphHandler.decompose_node_hidden_state(z, lineTrainer)
            line.position_type = "absolute"
            line.position['x'] *= max_dist
            line.position['y'] *= max_dist
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

        max_dist = patternTrainer.max_dist
        distance = max_dist*3.5
        #for i in range(3):
        sample_distance = max_dist*4
        min_coverage = 0.5
        outsider_distance = sample_distance/3.5

        max_x = 2
        max_y = 2
        for i in range(max_x):
            for j in range(max_y):
                print("i", i, "j", j)
                data = None
                while data is None:
                    reference_pos = GraphHandler._get_random_sampling_position(patternTrainer.template_data["lines"], max_dist)
                    print("reference_pos", reference_pos)
                    data = GraphHandler._sample_pattern_from_position(patternTrainer.template_data["lines"], reference_pos, latent_name=self.pattern_trainer.name, max_dist=sample_distance, inference=True)
                print("data", data.x.shape)

                self.insert_lines_from_sample_data(data, distance, x=i, y=j, outsider_distance=outsider_distance, min_coverage=min_coverage, patch_id=(i,j))

        return {"x":max_x, "y":max_y, "distance":distance, "outsider_distance":outsider_distance}

     
      
        # for i in range(0, 100):
        #     reference_position = {"x":random.randint(0,2*distance), "y":random.randint(0,2*distance)}
        #     filler_line = random.choice(self.original_lines).clone()
        #     filler_line.update_position_from_reference(reference_position)
            
        #     if all(line.pos_diff(filler_line) > config['max_dist']*1.1 for line in self.lines):
                
        #         filler_line.is_fixed = True
                    
        #         self.lines.append(filler_line)
        #         print("Added filler line at", reference_position)
                    

    def random_fill(self, fieldX=800, fieldY=800, retry_count=300, lineTrainer=None, patternTrainer=None, noise_level=0.01):
        if lineTrainer is None:
            lineTrainer = self.line_trainer
        if patternTrainer is None:
            patternTrainer = self.pattern_trainer

        count = retry_count

        max_dist = patternTrainer.max_dist
        
        while count > 0:
            sample = patternTrainer.dataset.get_random_item()
            noisy_data = sample.y.clone() + torch.randn(sample.y.size()) * noise_level
            ground_truth = self.decompose_node(noisy_data)
            ground_truth.update_position_from_reference({"x":random.randint(0, fieldX), "y":random.randint(0, fieldY)}, max_dist=max_dist)
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
        max_dist = self.pattern_trainer.max_dist if self.pattern_trainer else 150
        
        
        #for line in self.ghost_lines:
            #print(line.get_pattern_z())
        iterations = []

         
        for i, fixed_line in enumerate(self.lines):
            
            iteration_line = []
            #add original line to the mean mix
            iteration_line.append(fixed_line.get_pattern_z(center_position=fixed_line.position, max_dist=max_dist))
          

            #add ghost lines that are close to the original line to the mean mix
            for ghost_line in self.ghost_lines:
                ghost_line_z = ghost_line.get_pattern_z(center_position=fixed_line.position, max_dist=max_dist)

                diff = torch.sum(torch.abs(torch.sub(iteration_line[0], ghost_line_z)))
                if diff < diff_threshold:
                    iteration_line.append(ghost_line_z)
                
            
            averaged_line_z = torch.stack(iteration_line, dim=0).mean(dim=0)

            
            averaged_line = self.decompose_node(averaged_line_z)
            averaged_line.update_position_from_reference(fixed_line.position, max_dist=max_dist)
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
        
    def get_closest_original_line(self, line, allowed_indices=None):
        closest_line = None
        closest_diff = float('inf')
        closest_idx = None
        for idx, original_line in enumerate(self.original_lines):
            diff = line.latent_line_diff(original_line)
            if diff < closest_diff and (allowed_indices is None or idx in allowed_indices):
                closest_diff = diff
                closest_line = original_line
                closest_idx = idx
        
        return closest_line, closest_diff, closest_idx
        
    def reject_abnormal_lines(self):
        threshhold = self.avg_latent_diff /3
        print("rejecting abnormal lines. Threshhold:", threshhold)
        accepted_lines = []
        nr_lines = len(self.lines)

        print("rejecting abnormal lines. Current lines:", len(self.lines))
        for line in self.lines:
            _ , closest_diff, _ = self.get_closest_original_line(line)
            if closest_diff < threshhold:
                accepted_lines.append(line)
                    

        self.lines = accepted_lines
        if len(accepted_lines) < nr_lines:
            print("not all lines were accepted. Rejecting", nr_lines-len(accepted_lines), "lines")
            return "not all lines were accepted. Rejecting " + str(nr_lines-len(accepted_lines)) + " lines"
        else:
            return None
        


    def diffuse(self, line_idx):
        
      
        saved_copies = [line.clone() for line in self.lines]
        self.lines[line_idx].is_fixed = False
        self.lines[line_idx].stopped = False
      

        count = 0
        while self.calculate_gen_step(use_combinations=False, adaption_rate=0.01):
            count += 1
            if count > 1:
                #print("LOOP LIMIT reached")
                break
            #self.apply_gen_step()
        
       
        predicted_line = self.gen_step[line_idx]
        predicted_line.stopped = True
        predicted_line.is_fixed = True
        self.lines = saved_copies


        #print("MOVED:", predicted_line.position['x'] - self.lines[line_idx].position['x'], predicted_line.position['y'] - self.lines[line_idx].position['y'])
        return predicted_line

    def apply_momentum_to_line(self, current_line, predicted_line, diff_threshold):
        """
        Apply Exponential Moving Average smoothing to line movement to prevent oscillation.
        
        EMA provides stable, lag-based smoothing without overshoot.
        
        Args:
            current_line: The line in its current state
            predicted_line: The predicted next position for the line
            diff_threshold: Threshold for considering line as converged
        
        Returns:
            The predicted_line with smoothed position based on EMA
        """
        # Initialize EMA position on first iteration
        if not hasattr(current_line, "ema_position"):
            current_line.ema_position = current_line.position.copy()
        
        # Calculate movement magnitude from EMA to prediction
        delta_x = predicted_line.position['x'] - current_line.ema_position['x']
        delta_y = predicted_line.position['y'] - current_line.ema_position['y']
        movement_magnitude = (delta_x**2 + delta_y**2) ** 0.5
        
        # Adaptive alpha: more responsive near convergence, very smooth during movement
        if movement_magnitude < diff_threshold * 2:
            # Near convergence: allow faster response to snap into place
            alpha = 0.4
        elif movement_magnitude < diff_threshold * 5:
            # Close but not converged: moderate smoothing
            alpha = 0.25
        else:
            # Still moving: heavy smoothing to prevent oscillation
            alpha = 0.15
        
        # Update EMA: weighted average of prediction and previous EMA
        # alpha = 0 means no movement (pure smoothing)
        # alpha = 1 means no smoothing (jump to prediction)
        current_line.ema_position['x'] = (alpha * predicted_line.position['x'] + 
                                        (1 - alpha) * current_line.ema_position['x'])
        current_line.ema_position['y'] = (alpha * predicted_line.position['y'] + 
                                        (1 - alpha) * current_line.ema_position['y'])
        
        # Use EMA position instead of raw prediction
        predicted_line.position['x'] = current_line.ema_position['x']
        predicted_line.position['y'] = current_line.ema_position['y']
        
        # Carry EMA state forward to next iteration
        predicted_line.ema_position = current_line.ema_position.copy()
        
        return predicted_line

    def predict_with_multi_position_averaging(self, line, max_dist, sample_offsets=None):
        """
        Predict line properties by averaging predictions from multiple nearby positions.
        This reduces sensitivity to reference boundary effects and prevents oscillation.
        
        Args:
            line: The line to predict for
            max_dist: Maximum distance for reference line sampling
            sample_offsets: List of (dx, dy) tuples for sample positions. 
                           If None, uses default pattern.
        
        Returns:
            Averaged prediction in latent space (z)
        """
        if sample_offsets is None:
            # Default sampling pattern: current + 6 nearby positions
            sample_offsets = [
                (0, 0),      # Current position
                (2, 0),     # Right
                (-2, 0),    # Left
                (0, 2),     # Down
                (0, -2),    # Up
                (1, 0),     # Right
                (-1, 0),    # Left
                (0, 1),     # Down
                (0, -1),    # Up

                (2, 2),     # Right
                (-2, 2),    # Left
                (2, -2),     # Down
                (-2, -2),    # Up
                (1, 1),     # Right
                (-1, 1),    # Left
                (1, -1),     # Down
                (-1, -1),    # Up

                (10, 0),     # Right
                (-10, 0),    # Left
                (0, 10),     # Down
                (0, -10),    # Up
            ]
        
        predictions = []
        
        for dx, dy in sample_offsets:
            test_pos = {
                'x': line.position['x'] + dx,
                'y': line.position['y'] + dy
            }
            
            # Sample from this test position
            test_data = self.sample_pattern_from_position(
                test_pos,
                latent_name=self.pattern_trainer.name,
                max_dist=max_dist,
                inference=True
            )
            
            if test_data is not None:
                # Get prediction from this position
                z_pred = self.pattern_trainer.predict(
                    test_data.x,
                    test_data.edge_index,
                    test_data.target_point
                )
                predictions.append(z_pred)
        
        # Average all valid predictions
        if len(predictions) > 0:
            z_averaged = torch.mean(torch.stack(predictions), dim=0)
            #print(f"Averaged {len(predictions)}/{len(sample_offsets)} predictions")
            return z_averaged
        else:
            # Fallback: if all positions failed, return None
            # Caller should handle this case
            print("Warning: All multi-position predictions failed")
            return None

    def test_dead_spot(self, line, max_dist):

        sample_offsets = [    #only check the corners of the reference area
            (2, 0),     # Right
            (-2, 0),    # Left
            (0, 2),     # Down
            (0, -2),    # Up
        ]
        
        predictions = []
        
        for dx, dy in sample_offsets:
            test_pos = {
                'x': line.position['x'] + dx,
                'y': line.position['y'] + dy
            }
            
            # Sample from this test position
            test_data = self.sample_pattern_from_position(
                test_pos,
                latent_name=self.pattern_trainer.name,
                max_dist=max_dist,
                inference=True
            )
            
            if test_data is not None:
                # Get prediction from this position
                z_pred = self.pattern_trainer.predict(
                    test_data.x,
                    test_data.edge_index,
                    test_data.target_point
                )
                test_line = self.decompose_node(z_pred)
                test_line.update_position_from_reference(test_data.center_point, max_dist=max_dist)
                movement_vec = torch.tensor([test_line.position['x'] - test_pos['x'], test_line.position['y'] - test_pos['y']])
                to_center_vec = torch.tensor([line.position['x'] - test_pos['x'], line.position['y'] - test_pos['y']])
                dot_product = torch.dot(movement_vec, to_center_vec)
                angle = torch.acos(dot_product / (torch.norm(movement_vec) * torch.norm(to_center_vec)))
                angle = angle * 180 / math.pi
                
                predictions.append(angle.item())
              
        return predictions

    def evaluate_ensemble(self, line, max_dist):
        predictions = []
        sample_offsets = [
            (0, 0),      # Current position
            (10,10),
            (10,-10),
            (-10,10),
            (-10,-10),
            (10, 0),     # Right
            (-10, 0),    # Left
            (0, 10),     # Down
            (0, -10),    # Up
            (1,1),
            (1,-1),
            (-1,1),
            (-1,-1),
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),
        ]
        for dx, dy in sample_offsets:
            test_pos = {
                'x': line.position['x'] + dx,
                'y': line.position['y'] + dy
            }
            data = self.sample_pattern_from_position(test_pos, latent_name=self.pattern_trainer.name, max_dist=max_dist, inference=True)
            if data is not None:
                z = self.pattern_trainer.predict(data.x, data.edge_index, data.target_point)
                test_line = self.decompose_node(z)
                test_line.update_position_from_reference(data.center_point, max_dist=max_dist)
                predictions.append(test_line)
        
        clusters = GraphHandler.find_position_clusters(predictions, 15)
        biggest_cluster = max(clusters.values(), key=len)
        averaged_z = GraphHandler.average_latent_vectors(biggest_cluster, biggest_cluster[0].position, max_dist)
        
        # Convert the averaged z tensor to a Line object
        averaged_line = self.decompose_node(averaged_z)
        averaged_line.update_position_from_reference(biggest_cluster[0].position, max_dist)
        
        return predictions, averaged_line


    def calculate_flow_grid(self, grid_resolution=50):
        max_dist = self.pattern_trainer.max_dist
        min_x = min([line.position['x'] for line in self.lines])
        max_x = max([line.position['x'] for line in self.lines])
        min_y = min([line.position['y'] for line in self.lines])
        max_y = max([line.position['y'] for line in self.lines])

        flow_data = []

        for x in range(int(min_x-max_dist), int(max_x+max_dist), grid_resolution):
            print("x", x)
            for y in range(int(min_y-max_dist), int(max_y+max_dist), grid_resolution):
                
                data = self.sample_pattern_from_position({'x': x, 'y': y}, latent_name=self.pattern_trainer.name, max_dist=max_dist, inference=True)
                if data is not None:
                    z = self.pattern_trainer.predict(data.x, data.edge_index, data.target_point)
                    line = self.decompose_node(z)
                    line.update_position_from_reference(data.center_point, max_dist=max_dist)
                    
                    flow_data.append({'x': x, 'y': y, 'pred_x': line.position['x'], 'pred_y': line.position['y'], 'line': line.to_JSON()})

        return flow_data

    def accept_stationary_lines(self):
        for line in self.lines:
            
            if line is None:
                continue
            if not line.immutable:
                
                
                
                data = self.sample_pattern_from_position(line.position, latent_name=self.pattern_trainer.name, max_dist=self.pattern_trainer.max_dist, inference=True)
                if data is None:
                    continue
                next_z = self.pattern_trainer.predict(data.x, data.edge_index, data.target_point)
                old_z = line.get_pattern_z(latent_name=self.pattern_trainer.name, center_position=data.center_point, max_dist=self.pattern_trainer.max_dist)
                if next_z is None:
                    continue

                #next_z = next_z * 0.1 + old_z * (1 - 0.1)

                diff = torch.sum(torch.abs(next_z - old_z))
                print(f"\033[94mdiff stationary line check {diff}\033[0m")
                if diff < 3:
                    line.immutable = True

                new_line = self.decompose_node(next_z)
                new_line.update_position_from_reference(data.center_point, max_dist=self.pattern_trainer.max_dist)
                    
                self.ghost_lines.append(new_line)
            


    def calculate_gen_step(self, use_combinations=True, adaption_rate=0.1, average_predictions=False):

        exit_flag = False
        
        self.gen_step = []
        diff_threshold = 0.2
        max_dist = self.pattern_trainer.max_dist 


        if not hasattr(self, "ghost_lines"):
            self.ghost_lines = []
        
      

        for i in range(len(self.lines)):

            if self.lines[i] is None:
                self.gen_step.append(None)
                continue


            if not hasattr(self.lines[i], "stopped"):
                self.lines[i].stopped = False

            if self.lines[i].stopped:
                self.gen_step.append(self.lines[i].clone())
                continue
            
            if self.lines[i].is_fixed is False:
                


                
                
                

                data = self.sample_pattern_from_position(self.lines[i].position, latent_name=self.pattern_trainer.name, max_dist=max_dist, inference=True)
                
                if data is None:
                    print("line out of reference reach")
                    self.gen_step.append(None)
                    exit_flag = True
                    continue
                   
                if self.lines[i].stopped: 
                    print("line cloned to ghost line")
                    l = self.lines[i].clone()
                    l.used_ids = data.used_ids
                    #self.ghost_lines.append(l)
                    self.gen_step.append(l)
                    continue

                
                # Get target prediction from multi-position averaging
                if average_predictions:
                    next_z = self.predict_with_multi_position_averaging(self.lines[i], max_dist)
                else:
                    next_z = self.pattern_trainer.predict(data.x, data.edge_index, data.target_point)
                if next_z is None:
                    print("no prediction found")
                    self.gen_step.append(None)
                    continue
                
                
                #pred_line = self.decompose_node(next_z)
                #pred_line.update_position_from_reference(data.center_point, max_dist=max_dist)
                #self.ghost_lines.append(pred_line)

                old_z = self.lines[i].get_pattern_z(latent_name=self.pattern_trainer.name, center_position=data.center_point, max_dist=max_dist)
                next_z = next_z * adaption_rate + old_z * (1 - adaption_rate)
                 
                line = self.decompose_node(next_z)
                line.update_position_from_reference(data.center_point, max_dist=max_dist)
                
                # Preserve immutable attribute from original line
                line.immutable = self.lines[i].immutable

                diff = torch.sum(torch.abs(next_z - old_z))

                
                

                
                line.is_fixed = False
                line.stopped = False

                if diff < 0.05:
                    #dist_variance = self.test_dead_spot(line, max_dist)
                    #print(f"DEBUG: dist_variance={dist_variance}, type={type(dist_variance)}, is_not_none={dist_variance is not None}, greater_than_1000={dist_variance > 1000 if dist_variance is not None else 'N/A'}")
                    #if dist_variance is not None and dist_variance > 300:
                    #    print("\033[92mKICK\033[0m")
                        
                    #    self.gen_step.append(None)
                    #else:


                    line.stopped = True
                    self.lines[i].stopped = True
                    print("line stopped at diff", diff)
                    line.used_ids = data.used_ids
                    #self.ghost_lines.append(line)
                    self.gen_step.append(line)
                    
                    
                    
                    continue

                
               
                
                 

                self.gen_step.append(line)
            else:
                self.gen_step.append(self.lines[i].clone())

        if exit_flag:
            print("exit flag set")
            return False
       
        if all(line.is_fixed or line.stopped for line in [line for line in self.lines if line is not None]):
            print("ALL LINES FIXED")
            return False
           

        return True
    
    def apply_gen_step(self):
        self.lines = [line for line in self.gen_step]
        #print("count fixed", sum(1 for line in self.lines if line.is_fixed))
        #print("count stopped", sum(1 for line in self.lines if line.stopped))
        

    def self_arrange(self):

        for line in self.lines:
            line.stopped = True
            line.is_fixed = True


        step_lines = []
       
        
        selected_indices = [i for i, line in enumerate(self.lines) if not line.immutable and line is not None]
        # INSERT_YOUR_CODE
        not_selected_lines = [line for line in self.lines if line.immutable]
        print("selected indices", selected_indices)
        for i in selected_indices:
            
            diff_line = self.diffuse(i)
            step_lines.append(diff_line)


        for line in step_lines:
            line.is_fixed = True
        self.lines = step_lines + not_selected_lines


        #print("averaging main lines")
        self.ghost_lines = []
        self.gen_step = []
        #self.lines = self.cluster_and_average(self.lines, func1=self.find_latent_clusters, func2=self.find_position_clusters, eps1=0.2, eps2=20, message="latent first")
        #print([line.averaged_from for line in self.lines])
        #self.lines = self.cluster_and_average(self.lines, func1=self.find_position_clusters, func2=self.find_latent_clusters, eps1=20, eps2=0.2, message="pos first")
        #print([line.averaged_from for line in self.lines])

    def merge_lines_by_used_ids(self, max_dist=None):
        if max_dist is None:
            max_dist = self.pattern_trainer.max_dist

        id_to_line = {}
        for line in self.ghost_lines:
            if not hasattr(line, 'used_ids') or line.used_ids is None:
                raise ValueError(f"Line missing used_ids attribute. All ghost_lines must have used_ids set. Line: {line}")
            
            id_str = sorted(line.used_ids)
            id_str = "".join([str(id) for id in id_str])
            
            if id_str not in id_to_line:
                id_to_line[id_str] = []
            id_to_line[id_str].append(line)

        merged_lines = []
        for id_str, lines in id_to_line.items():
            averaged_latent = GraphHandler.average_latent_vectors(lines, lines[0].position, max_dist)
            line = GraphHandler.decompose_node_hidden_state(averaged_latent, self.line_trainer)
            line.update_position_from_reference(lines[0].position, max_dist=max_dist)
            if any(line.is_fixed for line in lines):
                line.is_fixed = True
            line.averaged_from = len(lines)
            line.cluster_label = id_str
            # Preserve used_ids from the original lines (all lines in group have same used_ids)
            line.used_ids = lines[0].used_ids
            merged_lines.append(line)
            
        self.ghost_lines = merged_lines

    
    def choose_ghost_lines(self):
        #instead of latent first we could do a voronoi cell based clustering and see if the lines are close enough in space to belong together
        #self.ghost_lines = self.cluster_and_average(self.ghost_lines, func1=self.find_latent_clusters, func2=self.find_position_clusters, eps1=0.2, eps2=150, message="latent first")
        #print([line.averaged_from for line in self.ghost_lines])
        eps_pos = self.avg_pos_diff /2
        eps_lat = self.avg_latent_diff /4
        self.ghost_lines = self.cluster_and_average(self.ghost_lines, func1=self.find_position_clusters, func2=self.find_latent_clusters, eps1=eps_pos, eps2=eps_lat, message="pos first")  #70 1
        print([line.averaged_from for line in self.ghost_lines])
        #self.ghost_lines.sort(key=lambda x: x.averaged_from)
        #print([line.averaged_from for line in self.ghost_lines])

        #die top auswahl müsste am ende eigentlich auf die nicht schon vorhandenen linien angewendet werden?
        #self.ghost_lines = self.top_p(self.ghost_lines, 0.5)
        #print([line.averaged_from for line in self.ghost_lines])

    def combine_ghost_and_main_lines(self):
        print("Removing not fixed lines:", len([line for line in self.lines if not line.is_fixed]))
        self.lines = [line for line in self.lines if line.is_fixed]
        if len(self.ghost_lines) > 0:
            return self.match_to_fixed_lines(self.lines, self.ghost_lines)
        else:
            # If no ghost lines, return all lines as untouched, no not_matched, no merged
            return self.lines, [], []
        
    def start_new_line(self):
        
        
        max_dist = round(self.pattern_trainer.max_dist if self.pattern_trainer else 150)
        
        line = random.choice([line for line in self.lines if not line.is_fixed or not None])
                
        pos = {'x': line.position['x'] + random.randint(-max_dist, max_dist), 'y': line.position['y'] + random.randint(-max_dist, max_dist)}

        z = self.line_trainer.randomInitPoint()
        line = GraphHandler.decompose_node_hidden_state(z, self.line_trainer)
        line.update_position_from_reference(pos, max_dist=max_dist)
        line.is_fixed = False
        line.stopped = False
        self.lines.append(line)

                  
        #self.ghost_lines = []

    def top_p(self, lines, p):
        """
        Perform top-p (nucleus) sampling on lines based on their averaged_from values.
        
        Args:
            lines: List of lines to sample from
            p: Probability threshold (0.0 to 1.0) for nucleus sampling
            
        Returns:
            List of lines that fall within the top-p cumulative probability
        """
        if not lines:
            return lines
            
        # Convert averaged_from counts to percentages
        total_count = sum(line.averaged_from for line in lines)
        if total_count == 0:
            return lines
            
        # Calculate probabilities for each line
        line_probs = []
        for line in lines:
            prob = line.averaged_from / total_count
            line_probs.append((line, prob))
        
        # Sort by probability in descending order
        line_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Perform nucleus sampling
        cumulative_prob = 0.0
        selected_lines = []
        
        for line, prob in line_probs:
            cumulative_prob += prob
            selected_lines.append(line)
            
            if cumulative_prob >= p:
                break
                
        return selected_lines
    
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



    def insert_lines_from_sample_data(self, sample_data, distance, x=None, y=None, outsider_distance=None, min_coverage=0.5, patch_id=None):
        for z in sample_data.x:
            line = GraphHandler.decompose_node_hidden_state(z, self.line_trainer)
            pos = {"x":x * distance, "y":y * distance}
            line.update_position_from_reference(pos, max_dist=sample_data.max_dist)
            line.is_fixed = True
            line.stopped = True
            line.immutable = True
            line.patch_position = {"x":x, "y":y}
         
            outside_distance, eval_directions = self.are_line_points_outside_distance(line, pos, outsider_distance, min_coverage)
            if outside_distance:
                line.immutable = False
                line.outside_directions = eval_directions
                line.patch_id = patch_id

            self.lines.append(line)

    def are_line_points_outside_distance(self, line, sample_center, distance, min_coverage):
        len_points = len(line.points)
        mc = min_coverage * len_points
        right = 0
        left = 0
        top = 0
        bottom = 0
        
        # Apply scale and rotation transformations to points
        cos_r = math.cos(line.rotation * 2 * math.pi)
        sin_r = math.sin(line.rotation * 2 * math.pi)
        
        for point in line.points:
            # Apply scale
            scaled_x = point['x'] * line.scale
            scaled_y = point['y'] * line.scale
            
            # Apply rotation
            rotated_x = scaled_x * cos_r - scaled_y * sin_r
            rotated_y = scaled_x * sin_r + scaled_y * cos_r
            
            # Calculate absolute position
            abs_x = rotated_x + line.position['x']
            abs_y = rotated_y + line.position['y']
            
            # Check if outside the box (relative to sample_center)
            rel_x = abs_x - sample_center['x']
            rel_y = abs_y - sample_center['y']
            
            if rel_x > distance:
                right += 1
            if rel_x < -distance:
                left += 1
            if rel_y > distance:
                bottom += 1
            if rel_y < -distance:
                top += 1
        
        eval_directions = {"right":right>mc, "left":left>mc, "top":top>mc, "bottom":bottom>mc}
        accumulated_eval = right>mc or left>mc or top>mc or bottom>mc

        print("eval_directions", eval_directions)
        return accumulated_eval, eval_directions

    def create_pattern_graph(self, pred_id, ids, latent_name=None, max_dist=None, dropped_out_ids=None, target_pos=None):
        return GraphHandler._create_pattern_graph(self.lines, pred_id, ids, latent_name, max_dist, dropped_out_ids, target_pos)

    @staticmethod
    def _create_pattern_graph(lines, pred_id, ids, latent_name=None, max_dist=None, dropped_out_ids=None, target_pos=None):
        
        if len(ids) == 0:
            raise ValueError("no ids given to create pattern graph")


        hidden_states = []


        if pred_id is not None and pred_id in ids:
            print("removed prediction id from ids", pred_id, ids)
            ids = ids[ids != pred_id]

        if len(ids) == 0:
            raise ValueError("No IDs left to create pattern graph", pred_id, ids)
        
        centers_X = [lines[i].position['x'] for i in ids]
        centers_Y = [lines[i].position['y'] for i in ids]
        center_point = { "x": sum(centers_X)/len(centers_X), "y": sum(centers_Y)/len(centers_Y) }
        ###### hier fliegt der divide by zero fehler
        

        #position of each node
        pos = torch.tensor([[lines[i].position['x'],lines[i].position['y']] for i in ids], dtype=torch.float)

        for i in ids:
            hid = GraphHandler._assemble_node_hidden_state(lines, i, center_point, latent_name, max_dist=max_dist)
            hidden_states.append(hid)

        if pred_id is not None:
            ground_truth = GraphHandler._assemble_node_hidden_state(lines, pred_id, center_point, latent_name, max_dist=max_dist)
        else:
            ground_truth = None
            #raise ValueError("no prediction id given. is this correct?", pred_id)

        #fully connect der nähesten k nodes
        #connections = torch.combinations(torch.arange(0,len(ids), dtype=torch.int64))
        #edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()

        
        x = torch.stack(hidden_states, dim=0) #vllt nochmal checken ob der jetzt "richtig rum" ist

        # Store target_point with batch dimension [1, 2] for proper PyG batching
        # PyG will stack these correctly: [1, 2] + [1, 2] -> [batch_size, 2]
        if target_pos is not None:
            # Use provided target position, normalized relative to center point
            target_pos_tensor = torch.tensor([target_pos['x'], target_pos['y']], dtype=torch.float)
            center_tensor = torch.tensor([center_point['x'], center_point['y']], dtype=torch.float)
            target_point = ((target_pos_tensor - center_tensor) / max_dist).unsqueeze(0)
        else:
            # Use existing behavior: target from ground truth
            target_point = ground_truth[:2].unsqueeze(0) if ground_truth is not None else None
        
        data = Data(x=x, y=ground_truth, center_point=center_point, pos=pos, target_point=target_point, max_dist=max_dist)
        
        # Store dropped-out node IDs (will be empty list if not provided)
        data.dropped_out_ids = dropped_out_ids if dropped_out_ids is not None else []
        
        data = T.Delaunay()(data)
        if data.face is not None:
            data = T.FaceToEdge()(data)
        #transform = T.Compose([T.ToUndirected()])
        #data = transform(data)
        
        return data
    

    def assemble_node_hidden_state(self, current_id, center_point, latent_name=None, max_dist=None):
        return GraphHandler._assemble_node_hidden_state(self.lines, current_id, center_point, latent_name, max_dist)

    
    @staticmethod
    def _assemble_node_hidden_state(lines, current_id, center_point, latent_name=None, max_dist=None):

        line = lines[current_id]
        
        lat_vec = line.latent_vectors[latent_name]

        if line.position_type == "absolute":
            delta_posX = line.position['x'] - center_point['x'] #delta zur main node position
            delta_posY = line.position['y'] - center_point['y']
        else:
            raise ValueError("Relative position in line while assembling node hidden state")

        # versuch das relativ anzugeben im bezug zur ... maxdist?
        delta_posX = delta_posX / max_dist
        delta_posY = delta_posY / max_dist
        
        rot = line.rotation
        scale = line.scale

        
        return torch.cat( (torch.tensor( [delta_posX, delta_posY, rot, scale], dtype=torch.float), lat_vec), 0)

    @staticmethod
    def _has_nearby_lines(lines, ground_truth_id, kept_ids, max_dist):
        """
        Check if the ground truth line has other lines within max_dist reach.
        
        Args:
            lines: List of Line objects
            ground_truth_id: ID of the ground truth line
            kept_ids: Tensor of kept line IDs to check against
            max_dist: Maximum distance threshold
            
        Returns:
            bool: True if at least one kept line is within max_dist of ground truth
        """
        ground_truth_line = lines[ground_truth_id]
        for kept_id in kept_ids:
            other_line = lines[kept_id.item()]
            dist = ground_truth_line.pos_diff(other_line)
            if dist < max_dist:
                return True
        return False

    
    # def sample_graph(self, pred_id, latent_name=None, max_dist=None, include_pred_id=False, with_combinations=False, node_dropout=0.0):
    #     if max_dist is None:
    #         raise ValueError("max_dist is required for sample_graph")
    #     if with_combinations and node_dropout > 0:
    #         print("node dropout and combinations not supported")
    #         exit()
        
    #     if latent_name is None:
    #         latent_name = self.pattern_trainer.name

    #     eps = 0.05

    #     dists = self.get_distance_matrix()
    #     dists = dists * (dists < max_dist)
        
    #     sorted_dists, indices = torch.sort(dists)

    #     current = sorted_dists[pred_id]
    #     current_ids = indices[pred_id]

    #     # Filter out nodes that have is_fixed == False
    #     fixed_mask = torch.tensor([getattr(self.lines[i], "is_fixed", False) for i in current_ids], dtype=torch.bool)
    #     not_zero = (current > eps) & fixed_mask
    #     current = current[not_zero]
    #     ids = current_ids[not_zero]
       
    #     if pred_id in ids:
    #         print("!! pred_id in ids", pred_id, ids)
    #         idx = (ids == pred_id).nonzero(as_tuple=True)[0]
    #         ids = torch.cat([ids[:idx], ids[idx+1:]])
    #         current = torch.cat([current[:idx], current[idx+1:]])

    #     if include_pred_id:
    #         ids = torch.cat([torch.tensor([pred_id]), ids])
    #         pred_id = None

    #     if node_dropout > 0 and len(ids) > 1:
    #         #drop node_dropout% of the nodes
    #         keepers = []
    #         for i in range(len(ids)):
    #             if random.random() > node_dropout:
    #                 keepers.append(ids[i])

    #         if len(keepers) == 0:
    #             keepers.append(random.choice(ids))

    #         print("dropping nodes.", len(keepers), "left from", len(ids))
    #         ids = keepers

    #     if len(ids) == 0:
    #         print("NO IDS in GRAPH SAMPLE")
    #         return None
            
        
    #     if with_combinations:
    #         combinations = torch.tensor(list(product([False, True], repeat=len(ids))))
    #         #combinations = combinations[combinations.sum(dim=1) <= 3]
    #         combinations = combinations[combinations.any(dim=1)]
    #         combinations = combinations.flip(dims=[1])  # Reverse each combination

    #         data_list = []

    #         for combo in combinations:
    #             combo_ids = ids[combo]
    #             if len(combo_ids) == 1 and combo_ids[0] == pred_id:
    #                 print("skipping combination with only pred_id", combo_ids, pred_id)
    #                 continue
    #             if pred_id in combo_ids:
    #                 print("ERROR: pred_id in combo_ids", combo_ids, pred_id)
    #                 exit()
                
    #             # Calculate dropped-out nodes: all proximity nodes not in this combination
    #             dropped_out_mask = ~combo
    #             dropped_out_ids = ids[dropped_out_mask].tolist()
                
    #             data = self.create_pattern_graph(pred_id, combo_ids, latent_name, max_dist=max_dist, dropped_out_ids=dropped_out_ids)
    #             data.used_ids = combo_ids.tolist()
    #             data_list.append(data)
            
    #         return data_list

    #     else:    
    #         # No combinations means all nearby nodes are used, so no dropped-out nodes
    #         data = self.create_pattern_graph(pred_id, ids, latent_name, max_dist=max_dist, dropped_out_ids=[])
    #         data.used_ids = ids.tolist()
    #         return data

    def get_random_sampling_position(self, noise_level=1):
        return GraphHandler._get_random_sampling_position(self.lines, self.pattern_trainer.max_dist, noise_level)

    @staticmethod
    def _get_random_sampling_position(lines, max_dist, noise_level=1):
        reference_pos = random.choice(lines).position.copy()
        reference_pos["x"] = reference_pos["x"] + random.random() * max_dist * noise_level
        reference_pos["y"] = reference_pos["y"] + random.random() * max_dist * noise_level
        return reference_pos

    def calculate_base_dataset(self):
        dataset = []
        for line in self.lines:
            sample_data = self.sample_pattern_from_position(line.position, latent_name=self.pattern_trainer.name, max_dist=self.pattern_trainer.max_dist, ref_pos_as_ground_truth=True)
            if sample_data is not None:
                dataset.extend(sample_data)
        print("Generated base dataset with", len(dataset), "samples")

        return dataset

    def calculate_dataset_onthefly(self, nr_samples=config['dataset_size_pattern'], noise_level=1, inference=False):
        dataset = []
        while len(dataset) < nr_samples:
            reference_pos = self.get_random_sampling_position(noise_level)
            sample_data = self.sample_pattern_from_position(reference_pos, latent_name=self.pattern_trainer.name, max_dist=self.pattern_trainer.max_dist, inference=inference)
            if sample_data is not None:
                dataset.append(sample_data)

        print("Generated", len(dataset), "samples")
        return dataset

    def sample_pattern_from_position(self, reference_pos, latent_name=None, max_dist=None, ref_pos_as_ground_truth=False, inference=False):
        return GraphHandler._sample_pattern_from_position(self.lines, reference_pos, latent_name, max_dist, ref_pos_as_ground_truth, inference)
    
    @staticmethod
    def _sample_pattern_from_position(lines, reference_pos, latent_name, max_dist, ref_pos_as_ground_truth=False, inference=False):
        """
        Sample a pattern graph from an absolute position for on-the-fly training data generation.
        
        Args:
            reference_pos: dict with 'x' and 'y' keys representing absolute position
            latent_name: name of the latent space to use
            max_dist: maximum distance for line proximity
        
        Returns:
            Data object with graph structure, or None if no lines in proximity
        """
        if inference and ref_pos_as_ground_truth:
            raise ValueError("inference mode requires ref_pos_as_ground_truth to be False")

        if max_dist is None:
            raise ValueError("max_dist is required for sample_pattern_from_position")
        
        # Calculate distances from reference_pos to all lines in self.lines
        ref_point = torch.tensor([reference_pos['x'], reference_pos['y']], dtype=torch.float)

        check_fixed = any(line.is_fixed for line in lines)
        
        #ToDo: vereinfachen? und auf batch umbauen im besten fall
        distances = []
        line_indices = []
        for i, line in enumerate(lines):
            if line is None:
                continue
            if check_fixed and line.is_fixed is False:
                continue
            line_pos = torch.tensor([line.position['x'], line.position['y']], dtype=torch.float)
            dist = torch.norm(ref_point - line_pos)
            distances.append(dist)
            line_indices.append(i)
        
        distances = torch.tensor(distances)
        line_indices = torch.tensor(line_indices)
        
        # Filter lines within max_dist proximity
        within_proximity = distances < max_dist
        nearby_distances = distances[within_proximity]
        nearby_indices = line_indices[within_proximity]
        
        # Around line 1163 in DrawData.py
        if len(nearby_indices) == 0:
            return None
        
        # Sort lines by distance to reference position
        sorted_distances, sort_order = torch.sort(nearby_distances)
        sorted_indices = nearby_indices[sort_order]


        if inference:
            kept_ids = sorted_indices.tolist()
            dropped_out_ids = []
            ground_truth_id = None

            data = GraphHandler._create_pattern_graph(
                lines,
                ground_truth_id, 
                kept_ids, 
                latent_name, 
                max_dist=max_dist, 
                dropped_out_ids=dropped_out_ids,
                target_pos=reference_pos
            )
            data.used_ids = kept_ids
            data.ground_truth_id = ground_truth_id
            
            return data
        


        
        if ref_pos_as_ground_truth:
            #mit 0 wird es so immer die näheste linie
            ground_truth_idx = 0
        else:
            # Randomly select one line as ground truth (this becomes the target)
            ground_truth_idx = random.randint(0, len(sorted_indices) - 1)
        ground_truth_id = sorted_indices[ground_truth_idx].item()
        ground_truth_distance = sorted_distances[ground_truth_idx].item()
        
        # Check that the ground truth has other lines within max_dist reach
        # otherwise the prediction process will drift out of reach
        # Exclude ground truth from the list of candidate lines to check
        candidate_indices = sorted_indices[sorted_indices != ground_truth_id]
        if not GraphHandler._has_nearby_lines(lines, ground_truth_id, candidate_indices, max_dist):
            #print("no nearby lines for ground truth in initial check")
            return None
        
        # Remove ground truth from the candidate list
        remaining_indices = torch.cat([sorted_indices[:ground_truth_idx], sorted_indices[ground_truth_idx+1:]])
        remaining_distances = torch.cat([sorted_distances[:ground_truth_idx], sorted_distances[ground_truth_idx+1:]])

        # Separate lines into must-keep (closer than ground truth) and droppable (farther)
        closer_mask = remaining_distances < ground_truth_distance
        must_keep_ids = remaining_indices[closer_mask].tolist()
        droppable_ids = remaining_indices[~closer_mask].tolist()
        #print("must keep ids", len(must_keep_ids), "droppable ids", len(droppable_ids))
        
        if ref_pos_as_ground_truth:
            
            
            # Generate all possible combinations of droppable lines
            result_data_list = []
            for r in range(0, len(droppable_ids) + 1):
                for combo in combinations(droppable_ids, r):
                    # Combine must-keep lines with current combination of droppable lines
                    kept_ids = must_keep_ids + list(combo)
                    
                    # Skip if no lines are kept (require at least one line)
                    if len(kept_ids) == 0:
                        continue
                    
                    # Lines that are dropped out: droppable lines not in this combination
                    dropped_out_ids = [id for id in droppable_ids if id not in combo]
                    
                    # Convert kept_ids to tensor for _create_pattern_graph
                    kept_ids_tensor = torch.tensor(kept_ids)
                    
                    # Verify ground truth still has nearby lines for this combination
                    if not GraphHandler._has_nearby_lines(lines, ground_truth_id, kept_ids_tensor, max_dist):
                        #print("no nearby lines for ground truth in combination check")
                        continue  # Skip this combination
                    
                    # Create pattern graph for this combination
                    data = GraphHandler._create_pattern_graph(
                        lines,
                        ground_truth_id,
                        kept_ids_tensor,
                        latent_name,
                        max_dist=max_dist,
                        dropped_out_ids=dropped_out_ids,
                        target_pos=reference_pos
                    )
                    data.used_ids = kept_ids
                    data.ground_truth_id = ground_truth_id
                    
                    result_data_list.append(data)
            
            return result_data_list
            
        else:
            # Apply dropout: lines closer than ground truth stay, lines farther have 50% chance to drop
            kept_ids = must_keep_ids
            dropped_out_ids = []
            
            for i in droppable_ids:
                if random.random() < 0.5:
                    kept_ids.append(i)
                else:
                    dropped_out_ids.append(i)
            
            if len(kept_ids) == 0:
                #print("no kept ids after dropout")
                return None
            
            kept_ids = torch.tensor(kept_ids)
            
            # Verify ground truth still has nearby lines after dropout
            if not GraphHandler._has_nearby_lines(lines, ground_truth_id, kept_ids, max_dist):
                #print("no nearby lines for ground truth after dropout")
                return None
            
            # Pass remaining line IDs to create_pattern_graph with ground truth line ID
            # Pass the reference position as target_pos parameter
            data = GraphHandler._create_pattern_graph(
                lines,
                ground_truth_id, 
                kept_ids, 
                latent_name, 
                max_dist=max_dist, 
                dropped_out_ids=dropped_out_ids,
                target_pos=reference_pos
            )
            data.used_ids = kept_ids.tolist()
            data.ground_truth_id = ground_truth_id
            
            return data
    
    
    def save_pattern_training_data(self, latent_name=None, name=None):
        if latent_name is None:
            latent_name = self.pattern_trainer.name

        if name is None:
            name = latent_name

        # Calculate max_dist from the dataset
        print("Calculating max_dist for pattern dataset...")
        self.calculate_original_lines()
        self.calculate_line_thresholds()
        max_dist = self.avg_pos_diff * config['max_dist_factor']
        print(f"Calculated max_dist: {max_dist} (avg_pos_diff: {self.avg_pos_diff}, factor: {config['max_dist_factor']})")
                
        file_path = GraphDatasetHandler.get_file_path(name, "pattern")
        data = {
            "lines": self.lines,
            "max_dist": max_dist
        }
        print("saving data", data)
        torch.save(data, file_path)
        
        




    

    
    def get_distance_matrix(self):
        return GraphHandler.get_distance_matrix_static(self.lines)

    @staticmethod
    def get_distance_matrix_static(lines):
        dist_list = []
        for line1 in lines:
            if line1.position_type == "absolute":
                dist_list.append([line1.position['x'], line1.position['y']])
            else:
                raise ValueError("prediction with relative position in lines")

        dist_tensor = torch.tensor(dist_list).float()
        return torch.cdist(dist_tensor, dist_tensor, p=2)
        


    def decompose_node(self, z, line_trainer=None):
        if line_trainer is None:
            line_trainer = self.line_trainer
        return GraphHandler.decompose_node_hidden_state(z, line_trainer)

    def match_to_fixed_lines(self, lines, ghost_lines):

        print("DEBUG: matching ghost lines to fixed lines", len(ghost_lines), "ghost lines and", len(lines), "fixed lines")

        line_buckets = {}
        not_matched = []
        
        for Gidx, ghost_line in enumerate(ghost_lines):
            belongs = False
            
            for idx, line in enumerate(lines):
                #das ist gerade der erst best passende statt der näheste
                #vllt über distanz matrix vorauswählen und dann nur lat diff vergleichen und da den besten nehmen?
               
                if self.match_two_lines(line, ghost_line):
                    if idx not in line_buckets:
                        line_buckets[idx] = []
                    line_buckets[idx].append(ghost_line)
                    belongs = True
                    break

            if not belongs:    
                ghost_line.is_fixed = True
                #ghost_line.averaged_from = 1
                not_matched.append(ghost_line)

        untouched_lines = [line for idx, line in enumerate(lines) if idx not in line_buckets]
        print("DEBUG: untouched_lines", len(untouched_lines))
        print("DEBUG: line_buckets keys:", list(line_buckets.keys()))
        print("DEBUG: total input lines:", len(lines))
        print("DEBUG: lines that will be merged:", len(line_buckets))

        # Pretty print the line_buckets and not_matched for inspection
        print("line_buckets content:")
        merged_lines = []
        for line_idx, ghosts in line_buckets.items():
            line = lines[line_idx]
            print(f"  Line id={id(line)}: {len(ghosts)} ghost(s)")
            #averaged_latent = GraphHandler.average_latent_vectors([line, *ghosts], line.position)
            #avg_line = GraphHandler.decompose_node_hidden_state(averaged_latent, self.line_trainer)
            #avg_line.update_position_from_reference(line.position)
            #avg_line.is_fixed = True
            #merged_lines.append(avg_line)
            merged_lines.append(line)
            


        print("DEBUG: not_matched content:")
        print(f"DEBUG: {len(not_matched)} ghost line(s) not matched")
        print(f"DEBUG: merged_lines created:", len(merged_lines))
        print(f"DEBUG: CHECK: untouched({len(untouched_lines)}) + merged({len(merged_lines)}) = {len(untouched_lines) + len(merged_lines)} vs input lines({len(lines)})")

        return untouched_lines, not_matched, merged_lines
     

                    
    

    def cluster_and_average(self, lines, func1, func2, eps1, eps2, message):
      
        print("CLUSTERING", message, len(lines), "lines")
        final_lines = []
        if(len(lines) == 0):
            return lines
        max_dist = self.pattern_trainer.max_dist if self.pattern_trainer else 150
        clusters_position = func1(lines, eps1)
        

        for label, cluster_lines in clusters_position.items():
            if label == -1 or len(cluster_lines) == 1:
                final_lines.extend(cluster_lines)
            else:
                
                #final_lines.extend(cluster_lines)
                clusters_latent = func2(cluster_lines, eps2)
                
                for latent_label, latent_lines in clusters_latent.items():
                    if latent_label == -1 or len(latent_lines) == 1:
                        final_lines.extend(latent_lines)
                    else:
                        averaged_latent = GraphHandler.average_latent_vectors(latent_lines, latent_lines[0].position, max_dist)
                        line = GraphHandler.decompose_node_hidden_state(averaged_latent, self.line_trainer)
                        line.update_position_from_reference(latent_lines[0].position, max_dist=max_dist)
                        if any(line.is_fixed for line in latent_lines):
                            line.is_fixed = True
                        line.averaged_from = len(latent_lines)
                        line.cluster_label = label
                        final_lines.append(line)
        
      
        for line in final_lines:
            if not hasattr(line, "averaged_from"):
                line.averaged_from = 1

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
    def average_latent_vectors(lines, center_position, max_dist):
        zs = []
        for line in lines:
            z = line.get_pattern_z(center_position=center_position, max_dist=max_dist)
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
    def get_target_pos_from_sample_data(sample_data):
        target_pos = sample_data.target_point.squeeze(0) * sample_data.max_dist
        return {'x': target_pos[0].item() + sample_data.center_point['x'], 'y': target_pos[1].item() + sample_data.center_point['y']}
    
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
    
    



    







