import einops 
from tqdm.notebook import tqdm
from torchsummary import summary
import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
import math

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

patch_size = 16
latent_size = 768
n_channels = 3
num_heads = 12
num_encoders = 12
dropout = 0.1
num_classes = 10
size = 224

epochs = 10
base_lr = 10e-3 
weight_decay = 0.03
batch_size = 8


class InputEmbedding(nn.Module):
    def __init__(self, patch_size=patch_size, n_channels=n_channels, device=device, latent_size=latent_size, batch_size=batch_size):
        super(InputEmbedding, self).__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.device = device
        self.batch_size = batch_size
        self.input_size = self.patch_size*self.patch_size*self.n_channels

        # Linear projection
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

        # Class token
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

    def forward(self, input_data):
        input_data = input_data.to(self.device)

        # Patchify input image
        patches = einops.rearrange(
            # b 0 batch, c = channels, h=height, w=width
            #Rule of einops (original input to new input)
            input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
        linear_projection = self.linearProjection(patches).to(self.device)
        b, n, _ = linear_projection.shape

        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m=n+1)
    
        linear_projection += pos_embed

        return linear_projection


test_input = torch.randn((8, 3, 224, 224))
test_class = InputEmbedding().to(device)
embed_test = test_class(test_input)


class ScaledDotProduct(nn.Module):
    def __init__(self, latent_size=latent_size, mask=None):
        super(ScaledDotProduct, self).__init__()
        
        self.dk = latent_size                 # dk = embed_len
        self.mask = mask
        self.softmax = nn.Softmax(dim=3)    # Softmax operator

    # Define the forward function
    def forward(self, queries, keys, values):       

        # First batch MatMul operation & scaling down by sqrt(dk).
        # Output 'compatibility' has shape:
        # (batch_size, num_heads, seq_len, seq_len)
        compatibility = torch.matmul(queries, torch.transpose(keys, 2, 3)) 
        compatibility = compatibility / math.sqrt((self.dk))               

        # Apply mask after scaling the result of MatMul of Q and K.
        # This is needed in the decoder to prevent the decoder from
        # 'peaking ahead' and knowing what word will come next.
        if self.mask is not None:
            compatibility = torch.tril(compatibility)
            
        # Normalize using Softmax
        compatibility_softmax = self.softmax(compatibility)        
               
        return torch.matmul(compatibility_softmax, torch.transpose(values, 1, 2))

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=num_heads, latent_size=latent_size, batch_size=batch_size, mask=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.batch_size = batch_size
        #self.embed_len = embed_len
        self.head_length = int(self.embed_len/self.num_heads)
        self.mask = mask
        self.concat_output = []

        # Q, K, and V have shape: (batch_size, seq_len, embed_len)
        self.q_in = self.k_in = self.v_in = self.embed_len

        # Linear layers take in embed_len as input 
        # dim and produce embed_len as output dim
        self.q_linear = nn.Linear(int(self.q_in), int(self.q_in))
        self.k_linear = nn.Linear(int(self.k_in), int(self.k_in))
        self.v_linear = nn.Linear(int(self.v_in), int(self.v_in))

        # Attention layer.
        if self.mask is not None:
            self.attention = ScaledDotProduct(mask=True) 
        else:
            self.attention = ScaledDotProduct()

        self.output_linear = nn.Linear(self.q_in, self.embed_len)

    def forward(self, queries, keys, values):

        # Query has shape: (batch_size, seq_len, num_heads, head_length)
        # Then transpose it: (batch_size, num_heads, seq_len, head_length)
        queries = self.q_linear(queries).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        queries = queries.transpose(1, 2)

        # Same for Key as for Query above.
        keys = self.k_linear(keys).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        keys = keys.transpose(1, 2)

        # Value has shape: (batch_size, seq_len, num_heads, head_length)
        values = self.v_linear(values).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)

        # 'sdp_output' here has size: 
        # (batch_size, num_heads, seq_len, head_length)
        sdp_output = self.attention.forward(queries, keys, values)

        # Reshape to (batch_size, seq_len, num_heads*head_length)
        sdp_output = sdp_output.transpose(1, 2).reshape(
            self.batch_size, -1, self.num_heads * self.head_length)

        # Return self.output_linear(sdp_output).
        # This has shape (batch_size, seq_len, embed_len)
        return self.output_linear(sdp_output)


class EncoderBlock(nn.Module):
    def __init__(self, latent_size=latent_size, num_heads=num_heads, device=device, dropout=dropout):
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device
        self.dropout = dropout

        # Normalization layer
        self.norm = nn.LayerNorm(self.latent_size)

        self.multihead = nn.MultiheadAttention(
            self.latent_size, self.num_heads, dropout=self.dropout
        )

        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size*4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size*4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, embedded_patches):
        firstnorm_out = self.norm(embedded_patches)
        attention_out = self.multihead(firstnorm_out, firstnorm_out, firstnorm_out)[0]

        # first residual connection
        first_added = attention_out + embedded_patches

        secondnorm_out = self.norm(first_added)
        ff_out = self.enc_MLP(secondnorm_out)

        return ff_out + first_added


test_encoder = EncoderBlock().to(device)
test_encoder(embed_test)

class Vit(nn.Module):
    def __init__(self, num_encoders=num_encoders, latent_size=latent_size, device=device, num_classes=num_classes, dropout=dropout):
        super(Vit, self).__init__()
        self.num_encoder = num_encoders
        self.latent_size = latent_size
        self.device = device
        self.num_classes = num_classes
        self.dropout = dropout

        self.embedding = InputEmbedding()

        # Create the stack of encoders
        self.encStack = nn.ModuleList([EncoderBlock() for i in range(self.num_encoder)])

        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes)
        )

    def forward(self, test_input):
        enc_output = self.embedding(test_input)

        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)

        cls_token_embed = enc_output[:, 0]

        return self.MLP_head(cls_token_embed)

model = Vit().to(device)
vit_output = model(test_input)
print(vit_output)
print(vit_output.size())
