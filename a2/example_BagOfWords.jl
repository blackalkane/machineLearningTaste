using JLD
using SparseArrays
using Printf

data = load("newsgroups.jld")
X = data["X"]
y = data["y"]
Xtest = data["Xtest"]
ytest = data["ytest"]
wordlist = data["wordlist"]
groupnames = data["groupnames"]

#=Q1 = wordlist[30]
@show Q1

Q2 = X[200, :]
@show Q2
@show wordlist[29] wordlist[100]

Q3 = y[200]
@show groupnames[4]=#
