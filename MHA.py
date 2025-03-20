class mha(nn):
  def __init__(self,headnum, hiddensize):
    self.q_proj = nn.linear(hiddensize, hiddensize)
    self.k_proj = nn.linear(hiddensize, hiddensize)
    self.v_proj = nn.linear(hiddensize, hiddensize)
    
    self.o_proj = nn.linear(hiddensize, hiddensize)

    self.head_num = headnum

  #x shape:(b,s,h)
  def forward(self, x, attention_mask):
    Q,K,V = self.q_proj(x), self.k_proj(x), self.v_proj(x) 
    # position encoding
    # Q = rope(Q) K =rope(K)
    
    Q = Q.view(b,s,head_num,_).permutue(1,2) #shape (b,num,s,h)
    K = K.view(b,s,head_num,_).permutue(1,2) #shape (b,num,s,h)
    V = V.view(b,s,head_num,_).permutue(1,2) #shape (b,num,s,h)

    weight = Q@K.permute(2,3)/sqrt(hiddensize/headnum)
    if not attention_mask:
      weight = weight.fill_mask(attention_mask==0, -1*e10)

    attention = nn.softmax(weight,dim =-1) @ V #(b,num,s,h)
    attention = attention.transpose(1,2).contiguous().view(b,s,-1)
    attention = nn.o_proj(attention)
    
