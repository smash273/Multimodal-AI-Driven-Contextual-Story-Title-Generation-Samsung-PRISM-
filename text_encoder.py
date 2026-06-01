"""
PRISM - BERT-based text encoder for OCR/ASR text.
"""
import torch
from transformers import BertTokenizer, BertModel

from config import BERT_MODEL_NAME, BERT_MAX_LENGTH


def load_bert(device: str = "cpu"):
    """Load BERT tokenizer and model."""
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_text_bert(
    text: str,
    tokenizer: BertTokenizer,
    model: BertModel,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Encode text using BERT and return the mean-pooled hidden state.

    Args:
        text: Input text string
        tokenizer: BERT tokenizer
        model: BERT model
        device: Device to run on

    Returns:
        Tensor of shape (768,) — mean-pooled BERT embedding
    """
    if text is None or len(text.strip()) == 0:
        return torch.zeros(768)

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=BERT_MAX_LENGTH,
    ).to(device)

    output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze(0).cpu()
