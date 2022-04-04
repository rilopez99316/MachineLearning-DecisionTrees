import numpy as np
import matplotlib.pyplot as plt

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

def entropy_from_p1(p1):
    p0 = 1-p1
    if p0==0 or p1==0:
        return 0
    return -p0*np.log2(p0) - p1*np.log2(p1)

p1 = np.linspace(0,1,100) # Generates 100 values from 0 to 1, unifirmly spaced

ent = []
for p in p1:
  ent.append(entropy_from_p1(p))

plt.plot(p1,ent)
plt.xlabel('p(1)')
plt.ylabel('entropy')

G = entropy_from_p1(0.5) - (4/8)*entropy_from_p1(0.5) - (4/8)*entropy_from_p1(0.5) 
print(G)
G = entropy_from_p1(0.5) - (4/8)*entropy_from_p1(0.25) - (4/8)*entropy_from_p1(0.75)
print(G) 
G = entropy_from_p1(0.5) - (4/8)*entropy_from_p1(0.5) - (4/8)*entropy_from_p1(0.5) 
print(G)

G = entropy_from_p1(0.25) - (2/4)*entropy_from_p1(0) - (2/4)*entropy_from_p1(0.5) 
print(G)
G = entropy_from_p1(0.25) - (2/4)*entropy_from_p1(0) - (2/4)*entropy_from_p1(0.5) 
print(G)
