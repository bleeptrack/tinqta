from config import config
import torch
import math
import copy


class Line():
    def __init__(self, points, scale=1, rotation=0, position=None, position_type="absolute"):
        self.points = points
        self.scale = scale
        self.rotation = rotation
        self.position = position
        self.position_type = position_type
        self.latent_vectors = {}
        self.is_fixed = False
        self.immutable = False
        #self.stopped = False
        self.dropout = 1
        self.adaption_rate = 1

        if isinstance(points, torch.Tensor):
            self.points = Line._tensor2Points(points)

        if(position is None and position_type == "absolute"):
            self.position = {
                "x": self.points[0]['x'],
                "y": self.points[0]['y']
            }
            for point in self.points:
                point['x'] -= self.position['x']
                point['y'] -= self.position['y']
    
    @staticmethod
    def _tensor2Points(x):
        points = []

        for i in range(x.shape[0]):
            points.append({
                'x': x[i][0].item(),
                'y': x[i][1].item()
                })
        return points

    def are_similar(self, other, relaxation=1):
        pos_diff = self.pos_diff(other) < 50 * (1 + relaxation)
        latent_diff = self.latent_line_diff(other) < 1 * (1 + relaxation)
        rot_diff_tmp = abs(self.rotation - other.rotation)
        rotation_diff = min(rot_diff_tmp, 1 - rot_diff_tmp) < 0.1 * (1 + relaxation)
        scale_diff_tmp = abs(self.scale - other.scale)
        scale_diff = min(scale_diff_tmp, 1 - scale_diff_tmp) < 0.1 * (1 + relaxation)
        if not pos_diff:
            print("FAILED pos_diff", self.pos_diff(other))
        if not latent_diff:
            print("FAILED latent_diff", self.latent_line_diff(other))
        if not rotation_diff:
            print("FAILED rotation_diff", min(rot_diff_tmp, 1 - rot_diff_tmp))
        if not scale_diff:
            print("FAILED scale_diff", min(scale_diff_tmp, 1 - scale_diff_tmp))
        return pos_diff and latent_diff and rotation_diff and scale_diff
    
    def diff(self, other):
        if self.position_type == "relative" or other.position_type == "relative":
            raise ValueError("Relative position type not supported for diff")
        return torch.sum(torch.abs(torch.tensor([[point['x'] - other.points[i]['x'], point['y'] - other.points[i]['y']] for i, point in enumerate(self.points)], dtype=torch.float)))
    
    def get_latent_vector(self, latent_name=None):
        if latent_name is None:
            if len(self.latent_vectors.keys()) == 1:
                latent_name = list(self.latent_vectors.keys())[0]
            else:
                raise ValueError("No latent name provided to fetch latent vector")
        return self.latent_vectors[latent_name]
    
    #compares only the shape latent vectors
    def latent_line_diff(self, other, latent_name=None):
        if latent_name is None:
            if len(self.latent_vectors.keys()) == 1:
                latent_name = list(self.latent_vectors.keys())[0]
            else:
                raise ValueError("No latent name provided to fetch latent vector")
        z1 = self.latent_vectors[latent_name]
        z2 = other.latent_vectors[latent_name]
        return torch.dist(z1, z2, p=2)

    def pattern_latent_diff(self, other, latent_name=None, max_dist=None):
        if latent_name is None:
            if len(self.latent_vectors.keys()) == 1:
                latent_name = list(self.latent_vectors.keys())[0]
            else:
                raise ValueError("No latent name provided to fetch latent vector")

        center_position = self.position
        z1 = self.get_pattern_z(latent_name=latent_name, center_position=center_position, max_dist=max_dist)
        z2 = other.get_pattern_z(latent_name=latent_name, center_position=center_position, max_dist=max_dist)
        print("diff", torch.dist(z1, z2, p=2))
        return torch.dist(z1, z2, p=2)
    
    def pos_diff(self, other):
        if self.position_type == "relative" or other.position_type == "relative":
            print("RELATIVE POSITION compared")
        return math.sqrt(
            (self.position['x'] - other.position['x']) ** 2 +
            (self.position['y'] - other.position['y']) ** 2
        )
    
    def _points2Tensor(self):
        return torch.tensor([[point['x'], point['y']] for point in self.points], dtype=torch.float)
    
    def get_pattern_z(self, latent_name=None, center_position=None, max_dist=None):
        if latent_name is None:
            if len(self.latent_vectors.keys()) == 1:
                latent_name = list(self.latent_vectors.keys())[0]
            else:
                raise ValueError("No latent name provided and multiple latent vectors found")
        # Concatenate scalar values with the latent vector tensor

        posX = self.position['x']
        posY = self.position['y']

        if center_position is not None:
            if max_dist is None:
                raise ValueError("max_dist is required when center_position is provided")
            posX -= center_position['x']
            posY -= center_position['y']
            posX /= max_dist
            posY /= max_dist
            

        if self.position_type == "absolute" and center_position is None:
            raise ValueError("Center position is required for absolute position")
        return torch.cat([
            torch.tensor([posX, posY, self.rotation, self.scale], dtype=torch.float),
            self.latent_vectors[latent_name]
        ])

    def update_position_from_reference(self, point, max_dist=None):
        #print("updating position from reference", self.position, point)
        if(self.position_type == "relative"):
            if max_dist is None:
                raise ValueError("max_dist is required when converting from relative position")
            self.position['x'] *= max_dist
            self.position['y'] *= max_dist
        if(self.position_type == "absolute"):
            #print("updating position on a line that is already absolute. Setting to 0,0 first")
            self.position['x'] = 0
            self.position['y'] = 0

        self.position['x'] += point['x']
        self.position['y'] += point['y']
        self.position_type = "absolute"
    
    def add_latent_vector(self, latent_vector, latent_name):
        self.latent_vectors[latent_name] = latent_vector
    
    def create_line_graph(self): #graph per stroke
        connections = []
        hidden_states = []
        for i in range(1,len(self.points)):
            connections.append([i-1,i])
            connections.append([i,i-1])

            connections.append([0,i])
            connections.append([i,0])

        if config['double_ended'] :
            for i in range(0,len(self.points)-1):
                connections.append([len(self.points)-1,i])
                connections.append([i,len(self.points)-1])

        edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()

        for point in self.points:
            hidden_states.append([point['x'], point['y']])

        x = torch.tensor(hidden_states, dtype=torch.float)

        return x, edge_index
    
    def to_JSON(self):
        import torch
        # Ensure all values are JSON-serializable (convert tensors to Python types)
        scale = float(self.scale.item()) if torch.is_tensor(self.scale) else float(self.scale)
        rotation = float(self.rotation.item()) if torch.is_tensor(self.rotation) else float(self.rotation)
        
        # Ensure position dict values are also JSON-serializable
        position = {
            'x': float(self.position['x'].item()) if torch.is_tensor(self.position['x']) else float(self.position['x']),
            'y': float(self.position['y'].item()) if torch.is_tensor(self.position['y']) else float(self.position['y'])
        }
        
        line = {
            "points": self.points,
            "scale": scale,
            "rotation": rotation,
            "position": position,
            "position_type": self.position_type,
        }
        if hasattr(self, 'used_ids'):
            line["used_ids"] = self.used_ids
        if hasattr(self, 'is_fixed'):
            line["is_fixed"] = self.is_fixed
        if hasattr(self, 'immutable'):
            line["immutable"] = self.immutable
        if hasattr(self, 'outside_directions'):
            line["outside_directions"] = self.outside_directions
        if hasattr(self, 'patch_id'):
            line["patch_id"] = self.patch_id
        if hasattr(self, 'cluster_number'):
            line["cluster_number"] = int(self.cluster_number)  # Ensure it's a Python int for JSON serialization
        return line
    
    def clone(self):
        # Create a new Line object with copied attributes
        # Use deep copy for points to avoid modifying the original
        points_copy = copy.deepcopy(self.points)
        cloned = Line(points_copy, self.scale, self.rotation, self.position, self.position_type)
        
        # Deep copy the latent_vectors dictionary and clone any tensors
        cloned.latent_vectors = {}
        for name, tensor in self.latent_vectors.items():
            if isinstance(tensor, torch.Tensor):
                cloned.latent_vectors[name] = tensor.clone()
            else:
                cloned.latent_vectors[name] = copy.deepcopy(tensor)
        
        # Copy other attributes
        cloned.is_fixed = self.is_fixed
        cloned.dropout = self.dropout
        cloned.adaption_rate = self.adaption_rate
        
        # Copy any additional attributes that might exist
        if hasattr(self, 'used_ids'):
            cloned.used_ids = copy.deepcopy(self.used_ids)
        if hasattr(self, 'immutable'):
            cloned.immutable = self.immutable
        if hasattr(self, 'stop_count'):
            cloned.stop_count = self.stop_count
            
        return cloned
    
    def get_absoulte_maxX(self):
        return max([point['x'] for point in self.points]) + self.position['x']

    def get_absoulte_maxY(self):
        return max([point['y'] for point in self.points]) + self.position['y']
    
    def get_absoulte_minX(self):
        return min([point['x'] for point in self.points]) + self.position['x']

    def get_absoulte_minY(self):
        return min([point['y'] for point in self.points]) + self.position['y']

    def __str__(self):
        return f"Line scale={self.scale}, rotation={self.rotation}, position={self.position} ({self.position_type})"

    def __repr__(self):
        return f"Line({len(self.points)} points, scale={self.scale}, rotation={self.rotation}, position={self.position} ({self.position_type}))"