import torch

class DataGenerator:
    def __init__(self, 
                 L: int, 
                 dx: int, 
                 dy: int, 
                 number_of_samples: int,
                 noise_std: float = 0.0,
                 ) -> None:
        self.L = L
        self.dx = dx
        self.dy = dy
        self.d = dx + dy
        self.number_of_samples = number_of_samples
        self.noise_std = noise_std

        # initialize G matrix indexing each task's position
        self.G = torch.zeros(self.dx, self.dy)
        self.r = self.dx / self.dy
        for i in range(self.dy):
            self.G[int(i * self.r):int((i + 1) * self.r), i] = 1

    def generate_data(self):
        """
        Generate data

        Returns:
            z: (n, L, d)
            z_q: (n, 1, d)
            y_q: (n, 1, dy)
        """
        # generate x and x_q
        x = torch.randn(self.number_of_samples, self.L, self.dx)    # (n, L, dx)
        x_q = torch.randn(self.number_of_samples, 1, self.dx)    # (n, 1, dx)

        # generate beta
        beta = torch.randn(self.number_of_samples, self.dx, self.dy)    # (n, dx, dy)
        beta = torch.einsum('nxy,xy->nxy', beta, self.G)    # (n, dx, dy)
        
        # generate y
        y = torch.einsum('nlx,nxy->nly', x, beta)   # (n, L, dy)
        y += self.noise_std * torch.randn(self.number_of_samples, self.L, self.dy)
        y_q = torch.einsum('nlx,nxy->nly', x_q, beta)  # (n, 1, dy)

        # generate z by concatenating x and y
        z = torch.cat([x, y], dim = 2)
        z_q = torch.cat([x_q, torch.zeros_like(y_q)], dim = 2)
        return z, z_q, y_q
    
# test the data generator
if __name__ == '__main__':
    L = 10
    dx = 5
    dy = 2
    number_of_samples = 1000
    noise_std = 0.1
    data_generator = DataGenerator(L, dx, dy, number_of_samples, noise_std)
    z, z_q, y_q = data_generator.generate_data()
    print(z.shape)
    print(z_q.shape)
    print(y_q.shape)