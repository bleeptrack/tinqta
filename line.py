from config import config
import torch
class Line():
    def __init__(self, points, scale=1, rotation=0, position=None, position_type="absolute"):
        self.points = points
        self.scale = scale
        self.rotation = rotation
        self.position = position
        self.position_type = position_type
        self.latent_vectors = {}

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
        return {
            "points": self.points,
            "scale": self.scale,
            "rotation": self.rotation,
            "position": self.position,
            "position_type": self.position_type
        }
    
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