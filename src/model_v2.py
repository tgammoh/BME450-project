import torch
import torch.nn as nn
from config import NUM_CLASSES, NUM_CHANNELS, WINDOW_SIZE


class EMGNetV2(nn.Module):

    def __init__(self, dropout=0.3):
        super(EMGNetV2, self).__init__()

        # ── spatial conv ──────────────────────────────────
        # kernel size 1 = looks across channels at each time step
        # learns which channel combinations fire together
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(NUM_CHANNELS, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # ── multi-scale temporal convs ────────────────────
        # three parallel branches, each a different timescale
        self.temporal_small = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.temporal_med = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.temporal_large = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # ── merge multi-scale ─────────────────────────────
        # 3 branches × 64 filters = 192 channels concatenated
        self.merge = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # ── attention pooling ─────────────────────────────
        # learns which time steps matter most
        self.attention = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )

        # ── classifier ────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = x.permute(0, 2, 1)           # [batch, 12,  200]

        # spatial: cross-channel relationships
        x = self.spatial_conv(x)          # [batch, 32,  200]

        # multi-scale: three parallel branches
        x_small = self.temporal_small(x)  # [batch, 64,  200]
        x_med   = self.temporal_med(x)    # [batch, 64,  200]
        x_large = self.temporal_large(x)  # [batch, 64,  200]

        # concatenate all three scales
        x = torch.cat([x_small, x_med, x_large], dim=1)  # [batch, 192, 200]

        # merge and compress
        x = self.merge(x)                 # [batch, 128, 200]

        # attention pooling: weighted sum across time
        weights = self.attention(x)       # [batch, 1,   200]
        x = (x * weights).sum(dim=2)     # [batch, 128]

        # classify
        x = self.dropout(x)
        x = self.fc(x)                    # [batch, 18]

        return x


if __name__ == '__main__':
    model = EMGNetV2()
    print(model)

    dummy = torch.randn(32, WINDOW_SIZE, NUM_CHANNELS)
    output = model(dummy)
    print(f'Input shape:  {dummy.shape}')
    print(f'Output shape: {output.shape}')
    print(f'\nParameter count:')
    total = sum(p.numel() for p in model.parameters())
    print(f'  total: {total:,}')