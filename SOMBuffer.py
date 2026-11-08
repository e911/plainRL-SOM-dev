import numpy as np

class SOMBuffer:
    def __init__(self, grid_size=(10, 10), state_dim=4, input_dim=11, lr=0.1, sigma=3.0):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.initial_lr = lr
        self.initial_sigma = sigma
        self.unit_vectors = np.random.uniform(-0.1, 0.1,(grid_size[0], grid_size[1], input_dim))

        self.node_ct = np.zeros(grid_size)
        self.node_means = np.zeros_like(self.unit_vectors)
        self.node_vars = np.ones_like(self.unit_vectors) * 0.01

        x, y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), indexing='ij')
        self.grid_coords = np.stack([x, y], axis=-1)

    def _get_bmu(self, x_vec):
        distances = np.linalg.norm(self.unit_vectors - x_vec, axis=-1)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def add(self, state, action, reward, next_state, done, step=1):
        x_vec = np.hstack([state, action, reward, next_state, float(done)])
        bmu = self._get_bmu(x_vec)

        lr = self.initial_lr / (1 + 0.001 * step)
        sigma = self.initial_sigma / (1 + 0.001 * step)
        bmu_coord = np.array(bmu)
        dist_sq = np.sum((self.grid_coords - bmu_coord) ** 2, axis=-1)
        neighborhood_func = np.exp(-dist_sq / (2 * sigma ** 2))

        self.unit_vectors += lr * neighborhood_func[..., np.newaxis] * (x_vec - self.unit_vectors)
        self.node_ct[bmu] += 1
        n = self.node_ct[bmu]
        prev_mean = self.node_means[bmu].copy()

        self.node_means[bmu] += (x_vec - prev_mean) / n
        if n > 1:
            self.node_vars[bmu] += (x_vec - prev_mean) * (x_vec - self.node_means[bmu])

    def sample(self, batch_size):
        valid_nodes = np.argwhere(self.node_ct > 0)
        if len(valid_nodes) == 0:
            indices = np.random.randint(0, self.grid_size[0], size=(batch_size,2))
        else:
            rand_idx = np.random.choice(len(valid_nodes), size=batch_size)
            indices = valid_nodes[rand_idx]

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            bmu_tuple = tuple(idx)
            mean = self.node_means[bmu_tuple]
            var = self.node_vars[bmu_tuple] / max(self.node_ct[bmu_tuple], 1)
            std = np.sqrt(var) + 1e-5

            samples = np.random.normal(mean, std)

            state = samples[0:self.state_dim]
            action = int(np.clip(np.round(samples[self.state_dim]), 0, 1))
            reward = samples[self.state_dim + 1]
            next_state = samples[self.state_dim + 2 : self.state_dim + 2 + self.state_dim]
            done = bool(np.round(np.clip(samples[-1], 0, 1)))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (np.array(states), 
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

class WeightedSOMBuffer:
    def __init__(self, grid_size=(10, 10), state_dim=4, input_dim=11, lr=0.1, sigma=3.0, alpha=0.9):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.initial_lr = lr
        self.initial_sigma = sigma
        self.alpha = alpha
        self.unit_vectors = np.random.uniform(-0.1, 0.1,(grid_size[0], grid_size[1], input_dim))

        self.node_ct = np.zeros(grid_size, dtype=np.int32)
        self.node_means = np.zeros_like(self.unit_vectors)
        self.node_vars = np.zeros_like(self.unit_vectors)
        # self.node_covs = np.zeros((grid_size[0], grid_size[1], input_dim, input_dim))
        # for i in range(grid_size[0]):
        #     for j in range(grid_size[1]):
        #         self.node_covs[i, j] = np.eye(input_dim) * 1e-4

        pos_w = 0.5 # cart position
        vel_w = 0.8 # cart velocity
        angle_w = 2.5 # pole angle
        ang_vel_w = 1.2 # pole angular velocity (rotational momentum)
        action_w = 0.4
        reward_w = 0.15
        done_w = 1.0
        next_state_scale = 0.8
        self.unit_vector_mask = np.array([
            pos_w, vel_w, angle_w, ang_vel_w, 
            action_w, 
            reward_w, 
            pos_w * next_state_scale, vel_w * next_state_scale, angle_w * next_state_scale, ang_vel_w * next_state_scale,
            done_w])
        
        x, y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), indexing='ij')
        self.grid_coords = np.stack([x, y], axis=-1)

    def _get_bmu(self, x_vec):
        diff = self.unit_vectors - x_vec
        weighted_diff = self.unit_vector_mask * diff
        squared_diff = weighted_diff ** 2
        summed_distances = np.sum(squared_diff, axis=-1)
        distances = np.sqrt(summed_distances) # L2-norm
        # distances = np.sum(self.unit_vector_mask * np.abs(self.unit_vectors - x_vec), axis=-1) # L1-norm
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def add(self, state, action, reward, next_state, done, step=1):
        x_vec = np.hstack([state, action, reward, next_state, float(done)])
        bmu = self._get_bmu(x_vec)

        # lr = self.initial_lr / (1 + 0.001 * step)
        # sigma = self.initial_sigma / (1 + 0.001 * step)
        decay_rate = np.exp(-step / 180.0)
        lr = self.initial_lr * decay_rate
        sigma = max(self.initial_sigma * decay_rate, 0.4)

        bmu_coord = np.array(bmu)
        dist_sq = np.sum((self.grid_coords - bmu_coord) ** 2, axis=-1)
        neighborhood_func = np.exp(-dist_sq / (2 * sigma ** 2))

        self.unit_vectors += lr * neighborhood_func[..., np.newaxis] * (x_vec - self.unit_vectors)

        self.node_ct[bmu] += 1
        t = self.node_ct[bmu]

        diff = self.node_means[bmu] - x_vec
        # cov_prod = np.outer(diff, diff)
        self.node_means[bmu] = (1.0 - self.alpha) * self.node_means[bmu] + self.alpha * x_vec
        self.node_vars[bmu] = (1.0 - self.alpha) * self.node_vars[bmu] + self.alpha * (diff ** 2)
        # self.node_covs[bmu] = (1.0 - self.alpha) * self.node_covs[bmu] + self.alpha * cov_prod

    def sample(self, batch_size):
        valid_nodes = np.argwhere(self.node_ct > 0)
        if len(valid_nodes) == 0:
            indices_x = np.random.randint(0, self.grid_size[0], size=batch_size)
            indices_y = np.random.randint(0, self.grid_size[1], size=batch_size)
            return self._default_sample(batch_size, indices_x, indices_y)
        
        rand_idx = np.random.choice(len(valid_nodes), size=batch_size)
        indices = valid_nodes[rand_idx]
        indices_x = indices[:, 0]
        indices_y = indices[:, 1]

        raw_means = self.node_means[indices_x, indices_y]
        counts = self.node_ct[indices_x, indices_y][:, np.newaxis]
        raw_vars = self.node_vars[indices_x, indices_y]

        means = raw_means / (1.0 - np.power(self.alpha, counts))
        vars = raw_vars / (1.0 - np.power(self.alpha, counts))

        std = np.sqrt(np.maximum(vars, 1e-5)) + 1e-5
        samples = np.random.normal(means, std)

        states = samples[:, 0:self.state_dim]
        actions = np.clip(np.round(samples[:, self.state_dim]), 0, 1).astype(np.int32)
        rewards = samples[:, self.state_dim + 1]
        next_states = samples[:, self.state_dim + 2 : self.state_dim + 2 + self.state_dim]
        dones = np.round(np.clip(samples[:, -1], 0, 1)).astype(bool)

        return states, actions, rewards, next_states, dones

    def _default_sample(self, batch_size, indices_x, indices_y):
        samples = self.unit_vectors[indices_x, indices_y]
        states = samples[:, 0:self.state_dim]
        actions = np.clip(np.round(samples[:, self.state_dim]), 0, 1).astype(np.int32)
        rewards = samples[:, self.state_dim + 1]
        next_states = samples[:, self.state_dim + 2 : self.state_dim + 2 + self.state_dim]
        dones = np.zeros(batch_size, dtype=bool)
        return states, actions, rewards, next_states, dones