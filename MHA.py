class mha(nn):
  def __init__(self,headnum, hiddensize):
    self.q_proj = nn.linear(hiddensize, hiddensize)
    self.k_proj = nn.linear(hiddensize, hiddensize)
    self.v_proj = nn.linear(hiddensize, hiddensize)
    
    self.o_proj = nn.linear(hiddensize, hiddensize)

    self.head_num = headnum

  #x shape:(b,s,h)
  def forward(self, x):
    Q,K,V = self.q_proj(x), self.k_proj(x), self.v_proj(x) 
    Q = Q.view(b,s,head_num,_).permutue(1,2) #shapd
    Q = Q.view(b,s,
    
