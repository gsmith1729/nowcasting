import numpy as np

for j in range(100):
    n=np.random.randint(10)+1
    factors=np.random.rand(n)
    phi=np.random.rand(n,n)
    evals, evecs = np.linalg.eig(phi)
    phi=phi/(np.absolute(max(evals))+0.5)
    L=np.random.rand(15,n)
    xs=[]
    fs=[]
    for t in range(3000):
        factors=np.matmul(phi,factors)+np.random.normal(size=n)
        x=np.matmul(L,factors)#+np.random.normal(size=15)
        xs.append(x)
        fs.append(factors)
    xs=np.asarray(xs)
    S=np.matmul(np.matrix.transpose(xs),xs)/len(xs)
    w ,v = np.linalg.eig(S)
    for i in range(len(w)):
        if np.abs(w[i])<1.1:
            print(i-n)
            break
