"""Frozen Bunny data generator; author class copied without numerical changes."""
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data import Dataset

class BunnyDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        lifted=False,
        noise_level=1,
        bunny_path="data/bunny.obj",
        mode="heat",  # Choose 'texture', 'heat', etc.
        sigma=0.25,  # Only used for 'heat'
        mean_idx=10500,  # Only used for 'heat'
    ):
        self.num_samples = num_samples
        self.lifted = lifted
        self.noise_level = noise_level
        self.bunny_path = bunny_path
        self.mode = mode
        self.sigma = sigma
        self.mean_idx = mean_idx
        self.data = self._get_data()

    def _get_data(self):
        # Load bunny mesh
        bunny = trimesh.load_mesh(self.bunny_path)
        faces = torch.tensor(bunny.faces, dtype=torch.long)

        # Normalize vertices: center and scale to unit ball
        vertices = torch.tensor(bunny.vertices, dtype=torch.float32)
        vertices -= vertices.mean(dim=0)
        scale = vertices.norm(dim=1).max()
        vertices /= scale

        # Diagnostic
        max_radius = vertices.norm(dim=1).max().item()
        print(
            f"[BUNNY DIAGNOSTIC] Max radius from center: {max_radius:.4f}"
        )  # Should be ~1.0

        # Compute consistent vertex normals (PyTorch version)
        def compute_vertex_normals(vertices, faces):
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            face_normals = torch.cross(v1 - v0, v2 - v0)

            vertex_normals = torch.zeros_like(vertices)
            for i in range(3):
                vertex_normals.index_add_(0, faces[:, i], face_normals)

            vertex_normals = F.normalize(vertex_normals, dim=1)
            return vertex_normals

        vertex_normals = compute_vertex_normals(vertices, faces)

        # Update mesh to reflect transformed vertices
        bunny.vertices = vertices.numpy()
        bunny._cache.clear()
        _ = bunny.vertex_normals  # Force recomputation of any cached normals

        self.mesh = bunny  # Now the mesh and samples are aligned

        if self.mode == "texture":
            # Patterned texture-based distribution
            weights = self.texture_pattern(vertices)
            # Normalize and sample
            weights = weights.clamp(min=0)
            weights /= weights.sum()
            sampled_indices = torch.multinomial(
                weights, self.num_samples, replacement=True
            )
            samples_on_surface = vertices[sampled_indices]

        elif self.mode == "heat":
            nose_tip = torch.tensor([0.0, 0.1, -0.1], dtype=torch.float32)
            dists_to_nose = torch.norm(vertices - nose_tip, dim=1)
            self.mean_idx = dists_to_nose.argmin().item()
            # Geodesic heat kernel distribution
            dists = self.compute_geodesic_distances(self.mean_idx, vertices, faces)
            weights = torch.exp(-dists / (2 * self.sigma**2))

            # Set the mean vertex
            mu = vertices[self.mean_idx]

            # Compute geodesic distances using heat kernel
            dists = self.compute_geodesic_distances(self.mean_idx, vertices, faces)
            weights = torch.exp(-dists / (2 * self.sigma**2))
            weights /= weights.sum()

            # Sample indices based on geodesic heat kernel weights
            sampled_indices = torch.multinomial(
                weights, self.num_samples, replacement=True
            )
            samples_on_surface = vertices[sampled_indices]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Optional: add orthogonal noise
        if self.lifted:
            for i in range(samples_on_surface.size(0)):
                vertex_normal = vertex_normals[sampled_indices[i]]
                noise = self.noise_level * torch.randn(3)
                noise_orth = torch.dot(noise, vertex_normal) * vertex_normal
                samples_on_surface[i] += noise_orth

        return samples_on_surface

    def texture_pattern(self, vertices):
        # Center of rings
        center = torch.tensor([0.05, 0.1, 0.1], dtype=torch.float32).to(vertices.device)
        r = torch.norm(vertices - center, dim=1)

        # Ring parameters
        num_rings = 5
        r_min, r_max = r.min().item(), r.max().item()
        ring_centers = torch.linspace(r_min + 0.02, r_max - 0.02, num_rings).to(
            vertices.device
        )
        sigma = 0.003  # controls ring width (small = sharp)

        # Build the pattern as sum of Gaussians centered at ring radii
        pattern = torch.zeros_like(r)
        for rc in ring_centers:
            pattern += torch.exp(-0.5 * ((r - rc) / sigma) ** 2)

        # Add slight noise to edges
        torch.manual_seed(42)
        noise = 0.05 * torch.randn_like(pattern)
        pattern += noise

        return pattern.clamp(min=0, max=1)

    def compute_geodesic_distances(self, source_idx, vertices, faces):
        return torch.norm(vertices - vertices[source_idx], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
