class GNConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # Ä¬ÈÏ16
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))
class MSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dc3_3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.dc1_5 = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels)
        self.dc5_1 = nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels)
        self.dc1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dc7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dc1_15 = nn.Conv2d(channels, channels, kernel_size=(1, 15), padding=(0, 7), groups=channels)
        self.dc15_1 = nn.Conv2d(channels, channels, kernel_size=(15, 1), padding=(7, 0), groups=channels)
        self.gnconv = GNConv(channels, channels, k=1, p=0)
    def forward(self, inputs):
        inputs = self.gnconv(inputs)
        x_0 = self.dc3_3(inputs)
        x_1 = self.dc1_5(x_0)
        x_1 = self.dc5_1(x_1)
        x_2 = self.dc1_7(x_0)
        x_2 = self.dc7_1(x_2)
        x_3 = self.dc1_15(x_0)
        x_3 = self.dc15_1(x_3)
        x = x_1 + x_2 + x_3+ x_0
        out = self.gnconv(x)
        return out
