########### Training data (rankings only, no dates):
con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/train_ratings_all.dat")
X.tr = read.table (con)
con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/train_y_rating.dat")
y.tr = read.table (con)


########### Divide training data into training and validation
n = dim(X.tr)[1]
n_train=c(1:50)
trtr = data.frame (X = X.tr[n_train,],y=y.tr[n_train,])

# Lasso
library(lasso2)
norm.vals = seq (10e-5, 0.25,by=0.001)

mods = l1ce(y~.+1,data=trtr, bound=norm.vals) # l1ce outputs a list of models
preds = sapply (mods, predict,newdata=trtr) # example of lapply!
resids = apply (preds, 2, "-", trtr$y)
RSSs = apply(resids^2, 2, sum)
lambdas = as.numeric(sapply (mods, "[",5))
plot (lambdas, RSSs, main=paste("Lasso RMSE Vs Lambda"),ylab="RMSE",xlab="lambda",log="x")
plot (norm.vals, RSSs, main=paste("Lasso RMSE Vs s"), ylab="RMSE",xlab="s")

# Ridge
library(MASS)
lambda.vals = exp(seq(-15,10,by=0.1))

mods = lm.ridge(y~.,data=trtr, lambda=lambda.vals)
preds = as.matrix(trtr[,1:99]) %*% t(coef(mods)[,-1]) +  rep(1,50) %o% coef(mods)[,1]
resids = matrix (data=trtr$y, nrow=dim(preds)[1], ncol=dim(preds)[2], byrow=F)-preds
RSSs = apply (resids^2, 2, sum)
plot (mods$lambda, RSSs, main=paste("Ridge RMSE Vs Lambda"),ylab="RMSE",xlab="lambda",log="x")
plot (apply(mods$coef^2, 2, sum), RSSs, main=paste("Ridge RMSE Vs s"), ylab="RMSE",xlab="s")

# Approximate the limiting solutions
library(pracma)

lambda.val = exp(-15)
norm.val = 0.15

# Lasso
mods = l1ce(y~.+1,data=trtr, bound=norm.val)
l1.norm.lasso = sum(abs(mods$coefficients[c(2:100)]))
l2.norm.lasso = sum(mods$coefficients[c(2:100)] ** 2)

# Ridge
mods = lm.ridge(y~.,data=trtr, lambda=lambda.val)
l1.norm.ridge = sum(abs(mods$coef))
l2.norm.ridge = sum(mods$coef ** 2)

cat("L1 Norm for Lasso", l1.norm.lasso)
cat("L1 Norm for Ridge", l1.norm.ridge)
cat("L2 Norm for Lasso", l2.norm.lasso)
cat("L2 Norm for Ridge", l2.norm.ridge)
