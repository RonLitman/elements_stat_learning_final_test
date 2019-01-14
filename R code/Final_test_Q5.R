########### Training data (rankings only, no dates):
con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/train_ratings_all.dat")
X.tr = read.table (con)
con = url("http://www.tau.ac.il/~saharon/StatsLearn2018/train_y_rating.dat")
y.tr = read.table (con)


########### Divide training data into training and validation
n = dim(X.tr)[1]
id_train = c(1:100)
n_train = 100
trtr = data.frame (X = X.tr[id_train,],y=y.tr[id_train,])

# Run SVM
# i
library(e1071)

eps = 0
ga = 0.0001
cost.vals = exp(seq(-15,10,by=0.1))
err.tr.rbf = NULL

for (c in cost.vals)
  {
  model = svm(y ~ ., data = trtr, type='eps-regression', epsilon=eps, cost=c, gamma=ga)
  pred = predict(model, trtr)
  err.tr.rbf = c(err.tr.rbf, mean(abs(pred-trtr$y)))
}

plot (cost.vals, err.tr.rbf, main="Support vector regression-RBF kernel Gamma=0.0001", xlab="Log Cost", ylab="mean absolute loss", type="l",log="x")

eps = 0
ga = 5.0
cost.vals = exp(seq(-15,10,by=0.1))
err.tr.rbf = NULL

for (c in cost.vals)
{
  model = svm(y ~ ., data = trtr, type='eps-regression', epsilon=eps, cost=c, gamma=ga)
  pred = predict(model, trtr)
  err.tr.rbf = c(err.tr.rbf, mean(abs(pred-trtr$y)))
}

plot (cost.vals, err.tr.rbf, main="Support vector regression-RBF kernel Gamma=5.0", xlab="Log Cost", ylab="mean absolute loss", type="l",log="x")

# iii
eps = 0
cost.val = exp(15)

ga = 0.0001
model_small = svm(y ~ ., data = trtr[id_train,], type='eps-regression', epsilon=eps, cost=cost.val, gamma=ga)

ga = 5.0
model_big = svm(y ~ ., data = trtr[id_train,], type='eps-regression', epsilon=eps, cost=cost.val, gamma=ga)

trtr[101,] = rep(5, each=100)
pred_small_ga = predict(model_small, trtr[101,])
pred_big_ga = predict(model_big, trtr[101,])

cat('Prediction for Small gamma is:', pred_small_ga, '\n')
cat('Prediction for Big gamma is:', pred_big_ga, '\n')
