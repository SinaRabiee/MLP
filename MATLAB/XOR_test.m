clear 
clc
X = [1 1 -1 -1; 1 -1 1 -1];
D_XOR = [-1 1 1 -1];
[W,V] = MLP(3,D_XOR,X,4,2,1);

Z = [1 1 -1 -1; 1 -1 1 -1; 1 1 1 1];
Y = cat(1,V*Z,ones(1,4))
W*Y


