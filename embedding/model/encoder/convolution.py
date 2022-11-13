import torch
from torch.nn import ModuleList, Conv2d, Dropout
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderConvolution(torch.nn.Module):
    def __init__(
        self, generators,
        letter_dimension=8, dimension=256,
        kernel_sizes=[3, 5]
    ):
        super().__init__()
        self.generators = generators
        self.letter_dimension = letter_dimension
        self.dimension = dimension

        self.kernel_sizes = kernel_sizes
        self.max_filter_size = max(kernel_sizes)
        self.kernel_num = dimension

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

        self.convs = ModuleList([
            Conv2d(1, self.kernel_num // len(self.kernel_sizes), (k, self.letter_dimension)) 
            for k in self.kernel_sizes
        ])
        self.dropout = Dropout(0.2)


    def forward(self, text, *args):
        embedded = self.embedding(text)
        max_len = embedded.size(1)

        if self.max_filter_size > max_len:
            tokens_zeros = Variable(embedded.data.new(
                embedded.size(0),
                self.max_filter_size - max_len,
                embedded.size(2)
            ))
            embedded = torch.cat([embedded, tokens_zeros], 1)
            max_len = embedded.size(1)

        embedded = embedded.unsqueeze(1)

        hidden = torch.cat(list(map(
            lambda x: F.max_pool1d(x, x.size(2)).squeeze(2),
            [
                F.relu(conv(embedded)).squeeze(3)
                for conv in self.convs
            ]
        )), 1)

        return hidden
