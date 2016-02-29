import knn
model = knn('datingTestSet')
model.train()
x=[1,1,1];k=3
res=model.test(x,k)


