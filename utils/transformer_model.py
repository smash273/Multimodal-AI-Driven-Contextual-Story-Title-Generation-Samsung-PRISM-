"""
PRISM - Multimodal Transformer Model for Video Title Generation.

Architecture:
  1. Linear projection (512 → 512) for multimodal tokens
  2. Transformer Encoder (3 layers, 8 heads) fuses modalities
  3. Transformer Decoder (3 layers, 8 heads) generates title tokens
"""
import torch
import torch.nn as nn

from config import (
    EMBED_DIM, NUM_HEADS,
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    PAD_IDX,
)


class MultimodalTransformer(nn.Module):
    """
    Multimodal Transformer encoder-decoder for video title generation.

    Input tokens shape: (B, num_modalities, embed_dim) — e.g. (B, 3, 512)
    Captions shape: (B, seq_len)
    Output logits shape: (B, seq_len - 1, vocab_size)
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size

        # Project multimodal tokens
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_ENCODER_LAYERS
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=NUM_DECODER_LAYERS
        )

        # Token embedding for the decoder
        self.embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)

        # Output projection
        self.fc = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, tokens: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (teacher forcing).

        Args:
            tokens: (B, num_modalities, 512) — multimodal feature tokens
            captions: (B, seq_len) — target caption indices (with BOS at start)

        Returns:
            logits: (B, seq_len - 1, vocab_size)
        """
        # Encode multimodal inputs
        memory = self.encoder(self.proj(tokens))

        # Decode: teacher forcing, shift captions right by removing last token
        tgt = self.embed(captions[:, :-1])
        out = self.decoder(tgt, memory)

        return self.fc(out)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode multimodal tokens to memory."""
        return self.encoder(self.proj(tokens))

    def decode_step(
        self, generated: list, memory: torch.Tensor, device: str = "cuda"
    ) -> torch.Tensor:
        """Single autoregressive decode step."""
        inp = torch.tensor([generated], device=device)
        tgt = self.embed(inp)
        out = self.decoder(tgt, memory)
        logits = self.fc(out[:, -1])
        return logits
