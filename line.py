from config import config
import torch
import math
from sklearn.cluster import DBSCAN
import numpy as np

class Line():
    def __init__(self, points, scale=1, rotation=0, position=None, position_type="absolute"):
        self.points = points
        self.scale = scale
        self.rotation = rotation
        self.position = position
        self.position_type = position_type
        self.latent_vectors = {}
        self.is_fixed = False
        self.dropout = 1
        self.adaption_rate = 0.1

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
    
    def diff(self, other):
        return torch.abs(torch.sum(torch.tensor([[point['x'] - other.points[i]['x'], point['y'] - other.points[i]['y']] for i, point in enumerate(self.points)], dtype=torch.float)))
    
    def get_latent_vector(self, latent_name=None):
        if latent_name is None:
            if len(self.latent_vectors.keys()) == 1:
                latent_name = list(self.latent_vectors.keys())[0]
            else:
                raise ValueError("No latent name provided to fetch latent vector")
        return self.latent_vectors[latent_name]
    
    def latent_line_diff(self, other, latent_name=None):
        if latent_name is None:
            if len(self.latent_vectors.keys()) == 1:
                latent_name = list(self.latent_vectors.keys())[0]
            else:
                raise ValueError("No latent name provided to fetch latent vector")
        z1 = self.latent_vectors[latent_name]
        z2 = other.latent_vectors[latent_name]
        return torch.abs(torch.sum(z1 - z2))
    
    def pos_diff(self, other):
        return math.sqrt(
            (self.position['x'] - other.position['x']) ** 2 +
            (self.position['y'] - other.position['y']) ** 2
        )
    
    def _points2Tensor(self):
        return torch.tensor([[point['x'], point['y']] for point in self.points], dtype=torch.float)
    
    def get_pattern_z(self, latent_name=None, center_position=None):
        if latent_name is None:
            if len(self.latent_vectors.keys()) == 1:
                latent_name = list(self.latent_vectors.keys())[0]
            else:
                raise ValueError("No latent name provided and multiple latent vectors found")
        # Concatenate scalar values with the latent vector tensor

        posX = self.position['x']
        posY = self.position['y']

        if center_position is not None:
            posX -= center_position['x']
            posY -= center_position['y']
            posX /= config['max_dist']
            posY /= config['max_dist']
            

        if self.position_type == "absolute" and center_position is None:
            raise ValueError("Center position is required for absolute position")
        return torch.cat([
            torch.tensor([posX, posY, self.rotation, self.scale], dtype=torch.float),
            self.latent_vectors[latent_name]
        ])

    def update_position_from_reference(self, point):
        #print("updating position from reference", self.position, point)
        if(self.position_type == "relative"):
            self.position['x'] *= config['max_dist']
            self.position['y'] *= config['max_dist']
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
        line = {
            "points": self.points,
            "scale": self.scale,
            "rotation": self.rotation,
            "position": self.position,
            "position_type": self.position_type,
        }
        if hasattr(self, 'used_ids'):
            line["used_ids"] = self.used_ids
        if hasattr(self, 'is_fixed'):
            line["is_fixed"] = self.is_fixed
        return line
    
    @staticmethod
    def cluster_and_average(lines):
        if(len(lines) == 0):
            return lines
        clusters_position, clusters_latent = Line.find_position_clusters(lines)
        
        for cluster_label, lines_in_cluster in clusters_position.items():
            if cluster_label == -1:
                print(f"Noise cluster: {len(lines_in_cluster)} lines")
            else:
                print(f"Cluster {cluster_label}: {len(lines_in_cluster)} lines")
                center_position = lines_in_cluster[0].position
                zs = []
                for line in lines_in_cluster:
                    if line not in clusters_latent[cluster_label]:
                        print(f"Line {line.id} not in latent cluster {cluster_label}")
                        continue
                    z = line.get_pattern_z(center_position=center_position)
                    zs.append(z)
                zs = torch.mean(torch.stack(zs), dim=0)
                print(zs.shape)
               
                exit()
        
        return lines

    @staticmethod
    def find_position_clusters(lines):
        positions = np.array([[line.position['x'], line.position['y']] for line in lines])
        latent_vectors = np.array([line.get_latent_vector().detach().numpy() for line in lines])
        dbscan_position = DBSCAN(eps=20, min_samples=2)
        labels_position = dbscan_position.fit_predict(positions)
        dbscan_latent = DBSCAN(eps=1, min_samples=2)
        labels_latent = dbscan_latent.fit_predict(latent_vectors)
        
        # Group lines by cluster label
        clusters_position = {}
        for idx, label in enumerate(labels_position):
            if label not in clusters_position:
                clusters_position[label] = []
            clusters_position[label].append(lines[idx])

        # nur latent vector gerade. sollte da scale und rotation rein?
        clusters_latent = {}
        for idx, label in enumerate(labels_latent):
            if label not in clusters_latent:
                clusters_latent[label] = []
            clusters_latent[label].append(lines[idx])
        
        # Now clusters[label] contains the list of lines in that cluster
        # Note: label -1 means noise/outliers
        print("Clustered lines by position:", {label: len(clusters_position[label]) for label in clusters_position})
        print("Clustered lines by latent:", {label: len(clusters_latent[label]) for label in clusters_latent})
       
        return clusters_position, clusters_latent

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