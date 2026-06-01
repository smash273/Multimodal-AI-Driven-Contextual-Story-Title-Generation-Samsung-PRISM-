"""
PRISM - PyTorch Dataset for Video Title Generation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    CLIP_DIR, OCR_DIR, ASR_DIR,
    PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX,
    EMBED_DIM,
)
from data.data_utils import load_clip, load_text, Vocabulary
from data.text_encoder import encode_text_bert


class VideoTitleDataset(Dataset):
    """
    Dataset that loads multimodal embeddings (CLIP visual + BERT-encoded
    OCR + BERT-encoded ASR) and ground truth titles for training.
    """

    def __init__(self, dataframe, vocab: Vocabulary, bert_tokenizer=None, bert_model=None, device="cpu"):
        self.df = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = row.video_id

        # Load CLIP visual embedding
        clip_emb = torch.tensor(load_clip(vid)).float()

        # Load OCR/ASR text and encode via BERT
        ocr_text = load_text(vid, OCR_DIR)
        asr_text = load_text(vid, ASR_DIR)

        ocr_emb = encode_text_bert(
            ocr_text, self.bert_tokenizer, self.bert_model, self.device
        )[:EMBED_DIM]
        asr_emb = encode_text_bert(
            asr_text, self.bert_tokenizer, self.bert_model, self.device
        )[:EMBED_DIM]

        # Stack modality tokens: [visual, ocr, asr] → (3, 512)
        tokens = torch.stack([clip_emb, ocr_emb, asr_emb])

        # Encode ground truth title
        title_tokens = [BOS_IDX] + [
            self.vocab.word2idx.get(w, UNK_IDX)
            for w in self.vocab.tokenize(row.title)
        ] + [EOS_IDX]

        return tokens, torch.tensor(title_tokens, dtype=torch.long)


def collate_fn(batch):
    """
    Collate function that pads captions to the same length within a batch.
    """
    tokens_list, captions_list = zip(*batch)
    tokens = torch.stack(tokens_list)

    max_len = max(len(c) for c in captions_list)
    padded = torch.full((len(captions_list), max_len), PAD_IDX, dtype=torch.long)

    for i, c in enumerate(captions_list):
        padded[i, :len(c)] = c

    return tokens, padded
