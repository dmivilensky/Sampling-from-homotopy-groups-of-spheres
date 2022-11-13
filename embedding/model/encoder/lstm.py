import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class EncoderLSTM(torch.nn.Module):
    def __init__(
        self, generators,
        letter_dimension=2, dimension=256,
        bidirectional=True, layers=2, dropout=0.2
    ):
        super().__init__()
        self.generators = generators
        self.letter_dimension = letter_dimension
        self.dimension = dimension
        self.dropout = dropout

        if self.letter_dimension != self.generators:
            self.embedding = torch.nn.Embedding(
                1 + 2 * self.generators,
                self.letter_dimension
            )
            print("trainable embeddings mode")
        else:
            def embedding(texts):
                binary = F.one_hot(
                    (texts - 1).fmod(self.generators).long() + 1,
                    num_classes=self.generators + 1
                )[..., 1:]
                binary.mul_(torch.sign(
                    -torch.stack([texts, texts], dim=2) + self.generators + 0.5
                ).long())
                return binary.float()

            self.embedding = embedding
            print("crosshair embeddings mode")

        self.lstm = torch.nn.LSTM(
            self.letter_dimension, self.dimension // (
                layers * (2 if bidirectional else 1)),
            num_layers=layers, bidirectional=True, dropout=self.dropout, batch_first=True
        )

    def forward(self, text, text_lengths):
        packed_embedded = pack_padded_sequence(
            self.embedding(text), text_lengths,
            batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat([hidden[i, :, :]
                           for i in range(hidden.shape[0])], dim=1)
        return hidden
