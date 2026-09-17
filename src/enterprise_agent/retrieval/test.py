import torch
import torch.nn as nn
import math
def cross_entory(logits,target):
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    log_probs = logits - torch.log(
        torch.exp(logits).sum(dim=-1, keepdim=True)
    )
    loss = -log_probs[
        torch.arange(logits.shape[0]), target
    ]

    return loss.mean()

def softmax(logits):
    logits = logits - torch.max(logits,dim=-1,keepdim=True)[0]
    exp_logits = torch.exp(logits)
    return exp_logits/(exp_logits.sum(dim=-1, keepdim=True))


class SelfAttention(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(SelfAttention,self).__init__()

        self.Q = nn.Linear(input_size,hidden_size)
        self.K = nn.Linear(input_size,hidden_size)
        self.V = nn.Linear(input_size,hidden_size)

    def forward(self,x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        scores = Q@K.transpose(1,2)

        scores = scores / math.sqrt(K.shape[-1])

        Seq_length = K.shape[1]

        mask = torch.tril(
            torch.ones(
                Seq_length,
                Seq_length,
                dtype=torch.bool,
                device=K.device
            ),
        )

        scores = scores.masked_fill(
            mask==0,-math.inf
        )

        attn = softmax(scores)

        output = attn@V

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self,input_size,hidden_size,d_model,n_heads):
        super(MultiHeadAttention,self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        assert self.d_model % self.n_heads == 0

        self.d_head = self.d_model // self.n_heads

        self.W_Q = nn.Linear(input_size,hidden_size)
        self.W_K = nn.Linear(input_size,hidden_size)
        self.W_V = nn.Linear(input_size,hidden_size)
        self.W_out = nn.Linear(input_size,hidden_size)

    def forward(self,x):

        B,S,D = x.shape

        Q = self.W_Q(x).view(B,S,self.n_heads,self.d_head).transpose(1,2)
        K = self.W_K(x).view(B,S,self.n_heads,self.d_head).transpose(1,2)
        V = self.W_V(x).view(B,S,self.n_heads,self.d_head).transpose(1,2)

        scores = Q@K.transpose(-1,-2)

        scores = scores / math.sqrt(self.d_head)

        mask = torch.tril(
            torch.ones(
                S,S,dtype=torch.bool,device=K.device
            )
        )

        scores = scores.masked_fill(
            mask==0,-math.inf
        )

        attn = torch.softmax(scores,dim=-1)

        output = attn@V

        output = output.transpose(1,2).contiguous().view(B,S,D)

        return self.W_out(output)

def causal_mask(seq_len,scores):
    mask = torch.tril(
        torch.ones(seq_len,seq_len,dtype=torch.bool,device=seq_len.device)
    )
    scores = scores.masked_fill(
        mask==0,-math.inf
    )

    attn = torch.softmax(scores,dim=-1)

    return attn




def rope(x,cos,sin):

    x1 = x[...,0::2]
    x2 = x[...,1::2]

    out1 = x1*cos - x2*sin
    out2 = x1*sin + x2*cos

    out = torch.stack((out1,out2),dim=-1)

    return out.flatten(-2)

def get_rope_cos_sin(head_dim=None,Seqlengh=None,device=None,base=10000.0):

    freq = 1 / (
        base**(
        torch.arange(
            0,head_dim,2,device=device
        ).float()/head_dim
    )
    )

    position = torch.arange(Seqlengh).float().to(device)

    theta = torch.outer(position,freq)

    cos = torch.cos(theta).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(theta).unsqueeze(0).unsqueeze(0)

    return cos,sin


class LoraLinear(nn.Module):
    def __init__(
            self,input_size,output_size,r,alpha
    ):
        super().__init__()

        self.base = nn.Linear(input_size,output_size)
        self.r = r
        self.scaling = alpha/r
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = torch.randn((r,input_size))*0.01
        self.B = torch.zeros((output_size,r))

    def forward(self,x):

        base_out = self.base(x)

        lora_out = x@self.A.T@self.B.T

        return (base_out+lora_out*self.scaling)


def topk_sampling(logits,k):
    values,indices = torch.topk(logits,k)

    filtered = torch.full_like(logits,-math.inf)

    filtered.scatter_(-1,indices,values)

    probs = torch.softmax(logits,dim=-1)

    return torch.multinomial(probs,1)
















