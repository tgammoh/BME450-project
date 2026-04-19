import torch
import torch.nn as nn
from config import NUM_CLASSES, NUM_CHANNELS, WINDOW_SIZE, DROPOUT


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        weights = weights.unsqueeze(2)
        return x * weights


class EMGNetV2(nn.Module):

    def __init__(self, dropout=DROPOUT):
        super(EMGNetV2, self).__init__()

        # ── spatial conv ──────────────────────────────────
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(NUM_CHANNELS, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # ── multi-scale temporal convs ────────────────────
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

        # ── merge ─────────────────────────────────────────
        self.merge = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # ── channel attention ─────────────────────────────
        self.channel_attention = ChannelAttention(128, reduction=4)

        # ── temporal attention pooling ────────────────────
        self.attention = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )

        # ── classifier head ───────────────────────────────
        self.dropout = nn.Dropout(p=dropout)
        self.fc1     = nn.Linear(128, 64)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = x.permute(0, 2, 1)               # [batch, 12,  200]

        # spatial
        x = self.spatial_conv(x)              # [batch, 32,  200]

        # multi-scale parallel branches
        x_small = self.temporal_small(x)      # [batch, 64,  200]
        x_med   = self.temporal_med(x)        # [batch, 64,  200]
        x_large = self.temporal_large(x)      # [batch, 64,  200]

        # concatenate
        x = torch.cat([x_small, x_med, x_large], dim=1)  # [batch, 192, 200]

        # merge
        x = self.merge(x)                     # [batch, 128, 200]

        # channel attention: which channels matter most
        x = self.channel_attention(x)         # [batch, 128, 200]

        # temporal attention: which time steps matter most
        weights = self.attention(x)           # [batch, 1,   200]
        x = (x * weights).sum(dim=2)         # [batch, 128]

        # classify
        x = self.dropout(x)
        x = self.fc1(x)                       # [batch, 64]
        x = self.relu(x)
        x = self.fc2(x)                       # [batch, 18]

        return x


if __name__ == '__main__':
    model = EMGNetV2()
    print(model)

    dummy = torch.randn(32, WINDOW_SIZE, NUM_CHANNELS)
    output = model(dummy)
    print(f'Input shape:  {dummy.shape}')
    print(f'Output shape: {output.shape}')
    total = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total:,}')