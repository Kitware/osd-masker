"""PyTorch model for masking burned-in on-screen displays.

Use OSDMasker to compute an on-screen display logit per pixel in a batch of grayscale images.
"""

import math

import torch


def _window_image(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    # x: N H W C
    x = x.view(x.shape[0], x.shape[1] // dim, dim, x.shape[2] // dim, dim, x.shape[3])
    return x.permute(0, 1, 3, 2, 4, 5)


def _unwindow_image(x: torch.Tensor) -> torch.Tensor:
    # x: N Hd Wd d d C
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3] * x.shape[4], x.shape[5])


def _init_linear(x: "torch.nn.Linear|torch.nn.Conv2d") -> None:
    torch.nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        torch.nn.init.normal_(x.bias, std=1e-4)


class _ConvMLP(torch.nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self._linear1 = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self._activation = torch.nn.ReLU()
        self._dropout = torch.nn.Dropout(0.1)
        self._linear2 = torch.nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

        _init_linear(self._linear1)
        _init_linear(self._linear2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1(x)
        x = self._activation(x)
        x = self._dropout(x)
        return self._linear2(x)


class _MLP(torch.nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self._linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self._activation = torch.nn.ReLU()
        self._dropout = torch.nn.Dropout(0.1)
        self._linear2 = torch.nn.Linear(hidden_dim, out_dim)

        _init_linear(self._linear1)
        _init_linear(self._linear2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1(x)
        x = self._activation(x)
        x = self._dropout(x)
        return self._linear2(x)


class _ConvLayer(torch.nn.Module):
    def __init__(self, *, in_dim: int, out_dim: int, kernel_size: int) -> None:
        super().__init__()
        self._conv = torch.nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding="same")
        self._dropout = torch.nn.Dropout(0.1)

        _init_linear(self._conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        return self._dropout(x)


class _AttentionConv(torch.nn.Module):
    def __init__(self, *, embed_dim: int, hidden_dim: int, window_dim: int, heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.window_dim = window_dim
        self.pad_dim = (window_dim - 1) // 2
        self.heads = heads  # TODO: Actually implement heads # noqa: FIX002 - We want this version of the model to be in the git history so we can run models trained with this version

        self._pe_k = torch.nn.Parameter(torch.empty((self.window_dim, self.window_dim, self.hidden_dim)))
        self._pe_v = torch.nn.Parameter(torch.empty((self.window_dim, self.window_dim, self.embed_dim)))
        self._norm1 = torch.nn.LayerNorm(embed_dim)
        self._q_proj = torch.nn.Linear(embed_dim, hidden_dim)
        self._k_proj = torch.nn.Linear(embed_dim, hidden_dim)
        self._v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self._dropout = torch.nn.Dropout(0.1)
        self._norm2 = torch.nn.LayerNorm(embed_dim)
        self._mlp = _MLP(in_dim=embed_dim, hidden_dim=embed_dim * 2, out_dim=embed_dim)

        _init_linear(self._q_proj)
        _init_linear(self._k_proj)
        _init_linear(self._v_proj)
        torch.nn.init.normal_(self._pe_k, std=1e-4)
        torch.nn.init.normal_(self._pe_v, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self._norm1(x)
        q: torch.Tensor = self._dropout(self._q_proj(x))
        k: torch.Tensor = self._dropout(self._k_proj(x))
        v: torch.Tensor = self._dropout(self._v_proj(x))
        k = torch.nn.functional.pad(k, (0, 0, self.pad_dim, self.pad_dim, self.pad_dim, self.pad_dim))
        v = torch.nn.functional.pad(v, (0, 0, self.pad_dim, self.pad_dim, self.pad_dim, self.pad_dim))
        ws = []
        for i in range(self.window_dim):
            for j in range(self.window_dim):
                k_ij = k[:, i : i + x.shape[1], j : j + x.shape[2], :]
                k_ij = k_ij + self._pe_k[i, j]
                w_ij = torch.sum(q * k_ij, dim=-1, keepdim=True) / math.sqrt(q.shape[-1])

                ws.append(w_ij)

        w = torch.cat(ws, dim=-1)
        w = torch.softmax(w, dim=-1)
        w = w.unflatten(-1, (self.window_dim, self.window_dim))

        x = torch.zeros_like(x)
        for i in range(self.window_dim):
            for j in range(self.window_dim):
                v_ij = v[:, i : i + x.shape[1], j : j + x.shape[2], :]
                v_ij = v_ij + self._pe_v[i, j]
                w_ij = w[:, :, :, i, j, None]
                x_ij = w_ij * v_ij
                x = x + x_ij

        x = x + res
        x = self._norm2(x)
        return self._mlp(x)


class OSDMasker(torch.nn.Module):
    """Predicts per-pixel logits for burned-in on-screen display elements.

    The model processes batches of single-channel (grayscale) images and preserves
    their spatial dimensions. Load trained weights before using the
    model to identify display elements such as text or graphics, since initialization
    yields random weights.

    Examples:
        Run the model on a batch of two grayscale images.

        >>> import torch
        >>> from osd_masker.model import OSDMasker
        >>> model = OSDMasker().eval()
        >>> images = torch.zeros(2, 1, 16, 16)
        >>> with torch.no_grad():
        ...     logits = model(images)
        >>> tuple(logits.shape)
        (2, 1, 16, 16)
    """

    def __init__(self) -> None:
        """Initializes the model with random weights.

        The model starts in training mode. Users should train directly
        from this or switch to eval() with loaded weights.

        Examples:
            Verify the model is in training mode.

            >>> import torch
            >>> from osd_masker.model import OSDMasker
            >>> model = OSDMasker()
            >>> model.training
            True
        """
        super().__init__()
        self._conv1 = _ConvLayer(in_dim=1, out_dim=4, kernel_size=3)
        self._conv2 = _ConvLayer(in_dim=1, out_dim=8, kernel_size=5)
        self._conv3 = _ConvLayer(in_dim=1, out_dim=4, kernel_size=7)
        self._mlp1 = _ConvMLP(in_dim=16, hidden_dim=32, out_dim=16)
        self._mlp2 = _ConvMLP(in_dim=32, hidden_dim=32, out_dim=16)
        self._conv4_in = _MLP(in_dim=16 * 8 * 8, hidden_dim=128 * 2, out_dim=128)
        self._conv4 = _AttentionConv(embed_dim=128, hidden_dim=64, window_dim=5, heads=4)
        self._conv4_out = _MLP(in_dim=128, hidden_dim=128 * 2, out_dim=16 * 8 * 8)
        self._mlp3 = _ConvMLP(in_dim=16, hidden_dim=32, out_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes an on-screen display (burn-in) logit mask over every pixel.

        Args:
            x:
                Grayscale images (torch.Tensor, shape-(N, 1, H, W)) where N is the batch size
                and H and W are positive multiples of 8. Use floating-point pixel values in the
                range [0.0, 1.0] that match the dtype of the model parameters (float32 by default).

        Returns:
            Raw logits mask (torch.Tensor, shape-(N, 1, H, W)). Larger values indicate stronger
            evidence that a pixel belongs to an on-screen display. Apply torch.sigmoid to convert
            to probabilities.

        Raises:
            ValueError: If x is not 4-dimensional, has a channel count other than 1, or has
                a height or width not divisible by 8.

        Examples:
            Call the model and convert logits to probabilities.

            >>> import torch
            >>> from osd_masker.model import OSDMasker
            >>> model = OSDMasker().eval()
            >>> image = torch.full((1, 1, 8, 16), 0.5)
            >>> with torch.no_grad():
            ...     probs = torch.sigmoid(model(image))
            >>> tuple(probs.shape)
            (1, 1, 8, 16)
        """
        if len(x.shape) != 4 or x.shape[1] != 1 or x.shape[2] % 8 or x.shape[3] % 8:
            raise ValueError(f"Expected input with shape (N, 1, H, W) where H and W are divisible by 8. Got {x.shape}")

        x = torch.cat([self._conv1(x), self._conv2(x), self._conv3(x)], dim=-3)
        x = self._mlp1(x)
        w = torch.softmax(x[:, 0:1].flatten(-2), dim=-1).unflatten(-1, x.shape[-2:])
        x = torch.cat([torch.sum(x * w, dim=(-2, -1), keepdim=True).expand_as(x), x], dim=-3)
        x = self._mlp2(x)
        x = x.permute(0, 2, 3, 1)
        x = _window_image(x, dim=8)
        shape = x.shape
        x = x.flatten(-3)
        x = self._conv4_in(x)
        x = self._conv4(x)
        x = self._conv4_out(x)
        x = _unwindow_image(x.reshape(shape))
        x = x.permute(0, 3, 1, 2)
        return self._mlp3(x)
