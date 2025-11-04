import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Implements a basic residual block (Conv -> BN -> ReLU -> Conv -> BN + Identity)
    to facilitate deeper networks without vanishing gradients.
    """
    def __init__(self, channels):
        super().__init__()
        # 1x1 convolution for identity mapping if channels change (not used here)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Add the original input (identity mapping)
        return self.relu(out)



class Encoder(nn.Module):
    """
    The Encoder (E) that maps a 3x96x96 frame to a low-dimensional latent vector (Z_DIM).
    """
    def __init__(self, in_channels=3, z_dim=256, base_features=32):
        """
        :param in_channels: 3 for RGB.
        :param z_dim: The final dimension of the latent vector (e.g., 256).
        :param base_features: Number of feature maps in the first convolutional layer.
        """
        super().__init__()
        
        self.z_dim = z_dim
        
        # Define the feature map counts for progressive increase
        f1 = base_features       # 32
        f2 = base_features * 2   # 64
        f3 = base_features * 4   # 128
        f4 = base_features * 8   # 256 (Final feature map count)
        
        # --- Sequential Downsampling (96x96 -> 6x6) ---
        
        # 1. 96x96 -> 48x48 (Output Channels: f1=32)
        self.layer1 = self._make_downsample_block(in_channels, f1, stride=2)
        
        # 2. 48x48 -> 24x24 (Output Channels: f2=64)
        self.layer2 = self._make_downsample_block(f1, f2, stride=2)
        
        # 3. 24x24 -> 12x12 (Output Channels: f3=128)
        self.layer3 = self._make_downsample_block(f2, f3, stride=2)
        
        # 4. 12x12 -> 6x6 (Output Channels: f4=256)
        self.layer4 = self._make_downsample_block(f3, f4, stride=2)

        self.flattened_size = f4 * 6 * 6
        
        self.fc_latent = nn.Sequential(
            nn.Linear(self.flattened_size, z_dim),
            nn.Tanh()
        )

    def _make_downsample_block(self, in_channels, out_channels, stride):
        """Creates a block with strided convolution for downsampling + ResidualBlock."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        # x shape: (B, 3, 96, 96)
        x = self.layer1(x)  # (B, 32, 48, 48)
        x = self.layer2(x)  # (B, 64, 24, 24)
        x = self.layer3(x)  # (B, 128, 12, 12)
        x = self.layer4(x)  # (B, 256, 6, 6)
        
        # Flatten the spatial features
        x = x.view(x.size(0), -1) # (B, 9216)
        
        # Project to the final latent vector z
        z = self.fc_latent(x) # (B, Z_DIM)
        # z = F.normalize(z, dim=1)
        
        return z



class Decoder(nn.Module):
    """
    The Decoder (D) that maps a low-dimensional latent vector (Z_DIM)
    back to a frame (C, 96, 96).
    """
    def __init__(self, out_channels=3, z_dim=256, base_features=32):
        super().__init__()
        
        # Define the feature map counts (in reverse order of the encoder)
        f1 = base_features       # 32
        f2 = base_features * 2   # 64
        f3 = base_features * 4   # 128
        f4 = base_features * 8   # 256
        
        # Calculate the size to flatten back from (B, Z_DIM)
        self.flattened_size = f4 * 6 * 6  # 256 * 6 * 6 = 9216
        self.z_dim = z_dim

        # 1. Initial Expansion
        # Projects the latent vector back to the flattened feature map size
        self.fc_expand = nn.Linear(z_dim, self.flattened_size)

        # 2. Sequential Upsampling (6x6 -> 96x96)
        
        # 6x6 -> 12x12 (Input Channels: f4=256, Output Channels: f3=128)
        self.layer4_up = self._make_upsample_block(f4, f3, stride=2)
        
        # 12x12 -> 24x24 (Input Channels: f3=128, Output Channels: f2=64)
        self.layer3_up = self._make_upsample_block(f3, f2, stride=2)
        
        # 24x24 -> 48x48 (Input Channels: f2=64, Output Channels: f1=32)
        self.layer2_up = self._make_upsample_block(f2, f1, stride=2)
        
        # 3. Final Upsampling and Output (48x48 -> 96x96)
        self.layer1_up = nn.Sequential(
            nn.ConvTranspose2d(f1, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def _make_upsample_block(self, in_channels, out_channels, stride):
        """Creates a block with strided Transposed Convolution for upsampling + ResidualBlock."""
        return nn.Sequential(
            # Transposed Conv for upsampling spatial dimensions
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, z):
        # z shape: (B, Z_DIM)

        # 1. Expand and Reshape
        x = self.fc_expand(z)  # (B, 9216)
        # Reshape to the starting feature map volume (B, 256, 6, 6)
        x = x.view(-1, self.flattened_size // (6 * 6), 6, 6)
        
        # 2. Sequential Upsampling
        x = self.layer4_up(x)  # (B, 128, 12, 12)
        x = self.layer3_up(x)  # (B, 64, 24, 24)
        x = self.layer2_up(x)  # (B, 32, 48, 48)
        
        # 3. Final Output
        reconstructed_frame = self.layer1_up(x)  # (B, 3, 96, 96)
        
        return reconstructed_frame



class StandardAutoencoder(nn.Module):
    """
    Combines the Encoder (E) and Decoder (D) into a standard Autoencoder.
    This serves as the Identity Reconstruction component of your final model.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        
        # Components are passed in (assuming they are already defined instances)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """
        The forward pass for a single frame.
        
        :param x: Input frame (Frame_i). Shape: (B, C, H, W)
        :returns: A tuple containing the latent vector and the reconstructed frame.
        """
        # 1. Encode the Input Frame: x -> z
        # z is your Low-Dimensional Embedding (LowDim i)
        z = self.encoder(x)
        # 2. Decode the Latent Vector: z -> x_prime (Reconstructed Frame)
        x_prime = self.decoder(z)
        
        return z, x_prime
    


class TimeAwareAutoencoder(StandardAutoencoder):
    """
    Extends the Standard Autoencoder to handle pairs of consecutive frames,
    incorporating time-aware transformations in the latent space.
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.T1 = nn.Linear(encoder.z_dim, encoder.z_dim*2)
        self.relu = nn.ReLU()
        self.T2 = nn.Linear(encoder.z_dim*2, encoder.z_dim*2)
        self.relu = nn.ReLU()
        self.T3 = nn.Linear(encoder.z_dim*2, encoder.z_dim)
        self.MLP = nn.Sequential(
            self.T1,
            self.relu,
            self.T2,
            self.relu,
            self.T3
        )

    def forward(self, x_t, x_t_plus_1=None):
        """
        The forward pass for a pair of consecutive frames.
        
        :param x_t: Input frame at time t. Shape: (B, C, H, W)
        :param x_t_plus_1: Input frame at time t+1. Shape: (B, C, H, W)
        :returns: A tuple containing the latent vectors and reconstructed frames.
        """
        # Encode first frame
        z_t = self.encoder(x_t)
        x_t_prime = self.decoder(z_t)
        if x_t_plus_1 is None:
            return z_t, x_t_prime
        
        # Encode second frame
        z_t_plus_1 = self.encoder(x_t_plus_1)
        x_t_plus_1_prime = self.decoder(z_t_plus_1)
        
        # Time-aware transformations
        z_t_pred = self.MLP(z_t)
        return z_t, x_t_prime, z_t_plus_1, z_t_pred, x_t_plus_1_prime


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(8, 3, 96, 96)  # batch of 8 RGB images
    encoder = Encoder(in_channels=3, z_dim=256).to(DEVICE)
    z = encoder(x)
    print(z.shape)  # Should be (8, 256)
    decoder = Decoder(out_channels=3, z_dim=256).to(DEVICE)
    y = decoder(z)
    print(y.shape)  # Should be (8, 3, 96, 96))
