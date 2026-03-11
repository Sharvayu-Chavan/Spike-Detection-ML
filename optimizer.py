import numpy as np

class PSOOptimizer:
    def __init__(self, num_particles=30, num_iterations=50):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        
        # Thresholds: [Ta, Ts, Tw]
        # Ta = D threshold (Sharpness) -> usually > 0
        # Ts = R bounds threshold (Ratio) -> usually 0 to 2
        # Tw = S1 steepness threshold -> usually > 0
        
        self.bounds = [
            (0.0, 1000.0), # Ta bounds
            (0.0, 5.0),    # Ts bounds
            (0.0, 500.0)   # Tw bounds
        ]
        
    def _evaluate(self, particle, features_list, labels):
        ta, ts, tw = particle
        
        correct = 0
        for features, label in zip(features_list, labels):
            s1 = features['S1']
            r = features['R']
            d = features['D']
            
            # is_spike logic based on Dingle criteria (as simple threshold rules):
            # D > Ta, |1 - R| < Ts, S1 > Tw
            is_spike = (d > ta) and (abs(1.0 - r) < ts) and (s1 > tw)
            predicted = 1 if is_spike else 0
            
            if predicted == label:
                correct += 1
                
        return correct / len(labels) if len(labels) > 0 else 0.0

    def optimize(self, features_list, labels):
        """
        Runs PSO to find optimal Ta, Ts, Tw thresholds.
        features_list: List of dictionaries with 'S1', 'S2', 'R', 'D'
        labels: List of integers (1 for True Spike, 0 for Artifact)
        """
        # Filter out 'Unsure' (-1) labels if any
        valid_indices = [i for i, lbl in enumerate(labels) if lbl in [0, 1]]
        clean_features = [features_list[i] for i in valid_indices]
        clean_labels = [labels[i] for i in valid_indices]
        
        if len(clean_labels) == 0:
            print("No valid labels provided for optimization.")
            return {'Ta': 0.0, 'Ts': 0.0, 'Tw': 0.0}
            
        print(f"Starting PSO with {len(clean_labels)} labeled samples.")
            
        # Initialize particles
        dim = len(self.bounds)
        particles = np.zeros((self.num_particles, dim))
        for i in range(dim):
            low, high = self.bounds[i]
            particles[:, i] = np.random.uniform(low, high, self.num_particles)
            
        velocities = np.zeros((self.num_particles, dim))
        
        pbest_positions = np.copy(particles)
        pbest_scores = np.array([-1.0] * self.num_particles)
        
        gbest_position = np.zeros(dim)
        gbest_score = -1.0
        
        w = 0.5  # Inertia
        c1 = 1.5 # Cognitive constant
        c2 = 1.5 # Social constant
        
        for iteration in range(self.num_iterations):
            for i in range(self.num_particles):
                score = self._evaluate(particles[i], clean_features, clean_labels)
                
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = np.copy(particles[i])
                    
                if score > gbest_score:
                    gbest_score = score
                    gbest_position = np.copy(particles[i])
                    
            # Update velocities and positions
            r1 = np.random.rand(self.num_particles, dim)
            r2 = np.random.rand(self.num_particles, dim)
            
            velocities = (w * velocities + 
                          c1 * r1 * (pbest_positions - particles) + 
                          c2 * r2 * (gbest_position - particles))
                          
            particles += velocities
            
            # Apply bounds
            for i in range(dim):
                low, high = self.bounds[i]
                particles[:, i] = np.clip(particles[:, i], low, high)
                
        print(f"PSO completed. Best accuracy: {gbest_score:.2f}")
        return {
            'Ta': gbest_position[0],
            'Ts': gbest_position[1],
            'Tw': gbest_position[2]
        }
