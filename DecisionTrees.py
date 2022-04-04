import numpy as np

def entropy(S):
  p1 = np.sum(S)/len(S)
  p0 = 1-p1
  if p0==0 or p1==0:
    return 0
  return -p0*np.log2(p0) - p1*np.log2(p1)

S = [0,0,0,0,1,1,1,1]
print(entropy(S))
S = [0,0,0,0]
print(entropy(S))
S = [0,1,1,1,1]
print(entropy(S))
S = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
print(entropy(S))