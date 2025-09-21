#!/usr/bin/env python3
"""
Test script for the new adaptive clustering approach.
This demonstrates how the proximity-based dynamic weighting works.
"""

import numpy as np
import torch
from DrawData import GraphHandler
from line import Line

def create_test_lines():
    """Create test lines with known latent and physical relationships."""
    lines = []
    
    # Group 1: Close in latent space, far in physical space
    for i in range(3):
        line = Line(
            points=[{'x': 0, 'y': 0}, {'x': 10, 'y': 0}],
            scale=1.0,
            rotation=0.0,
            position={'x': i * 200, 'y': 0}  # Far apart physically
        )
        # Similar latent vectors (close in latent space)
        latent_vec = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5] + [0.01 * i] * 10)
        line.add_latent_vector(latent_vec, "test")
        lines.append(line)
    
    # Group 2: Close in physical space, far in latent space
    for i in range(3):
        line = Line(
            points=[{'x': 0, 'y': 0}, {'x': 10, 'y': 0}],
            scale=1.0,
            rotation=0.0,
            position={'x': 100, 'y': i * 5}  # Close physically
        )
        # Very different latent vectors (far in latent space)
        latent_vec = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2] + [0.5 * i] * 10)
        line.add_latent_vector(latent_vec, "test")
        lines.append(line)
    
    # Group 3: Close in both spaces (should definitely cluster together)
    for i in range(2):
        line = Line(
            points=[{'x': 0, 'y': 0}, {'x': 10, 'y': 0}],
            scale=1.0,
            rotation=0.0,
            position={'x': 300, 'y': i * 2}  # Close physically
        )
        # Similar latent vectors (close in latent space)
        latent_vec = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9] + [0.01 * i] * 10)
        line.add_latent_vector(latent_vec, "test")
        lines.append(line)
    
    return lines

def test_adaptive_clustering():
    """Test the new adaptive clustering approach."""
    print("Testing Adaptive Clustering with Proximity-Based Dynamic Weighting")
    print("=" * 70)
    
    # Create test lines
    lines = create_test_lines()
    print(f"Created {len(lines)} test lines")
    
    # Create a mock GraphHandler for testing
    class MockGraphHandler(GraphHandler):
        def __init__(self):
            super().__init__()
            # Mock line trainer
            self.line_trainer = type('MockLineTrainer', (), {
                'name': 'test'
            })()
    
    handler = MockGraphHandler()
    
    # Test the clustering
    print("\nOriginal lines:")
    for i, line in enumerate(lines):
        print(f"  Line {i}: pos=({line.position['x']:.1f}, {line.position['y']:.1f}), "
              f"latent_norm={torch.norm(line.get_latent_vector()).item():.3f}")
    
    # Test with different epsilon values
    for eps in [0.1, 0.3, 0.5]:
        print(f"\n--- Testing with eps={eps} ---")
        clustered_lines = handler.cluster_and_average(lines, eps=eps)
        
        print(f"Clustered into {len(clustered_lines)} groups:")
        for i, line in enumerate(clustered_lines):
            if hasattr(line, 'averaged_from'):
                print(f"  Cluster {i}: pos=({line.position['x']:.1f}, {line.position['y']:.1f}), "
                      f"averaged_from={line.averaged_from} lines")
            else:
                print(f"  Cluster {i}: pos=({line.position['x']:.1f}, {line.position['y']:.1f}), "
                      f"single line")

def demonstrate_adaptive_behavior():
    """Demonstrate how the adaptive weighting works."""
    print("\n" + "=" * 70)
    print("ADAPTIVE BEHAVIOR DEMONSTRATION")
    print("=" * 70)
    
    print("""
The new clustering approach works as follows:

1. PROXIMITY CALCULATION:
   - For each line, calculates how close it is to other lines in latent space
   - For each line, calculates how close it is to other lines in physical space
   - Uses k-nearest neighbors (k=3) to determine local density

2. DYNAMIC WEIGHTING:
   - If lines are very close in latent space (proximity > 0.8):
     * Latent space gets high weight (0.8)
     * Physical space gets low weight (0.3)
   - If lines are very close in physical space (proximity > 0.8):
     * Physical space gets high weight (0.8) 
     * Latent space gets low weight (0.3)
   - Weights are normalized so they sum to 1

3. ADAPTIVE DISTANCE:
   - Final distance = (latent_weight * latent_distance) + (physical_weight * physical_distance)
   - This means: if very close in latent space, physical distance becomes less important
   - And vice versa: if very close in physical space, latent distance becomes less important

4. CLUSTERING:
   - Uses DBSCAN with the adaptive distance matrix
   - Lines that are close in the weighted space get clustered together
   - Averages the latent vectors of clustered lines

EXPECTED BEHAVIOR:
- Group 1 (close latent, far physical): Should cluster together (latent space dominates)
- Group 2 (far latent, close physical): Should cluster together (physical space dominates)  
- Group 3 (close both): Should definitely cluster together
""")

if __name__ == "__main__":
    test_adaptive_clustering()
    demonstrate_adaptive_behavior()
