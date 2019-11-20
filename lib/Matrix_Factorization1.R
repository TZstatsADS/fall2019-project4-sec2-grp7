#Define a function to calculate RMSE
RMSE <- function(rating, est_rating){
  sqr_err <- function(obs){
    sqr_error <- (obs[3] - est_rating[as.character(obs[2]), as.character(obs[1])])^2
    return(sqr_error)
  }
  return(sqrt(mean(apply(rating, 1, sqr_err))))  
}

h=matrix(0,nrow=610,ncol=9724)
rownames(h)=colnames(result$p)
colnmaes(h)=colnames(result$q)
for (i in 1: di,(ratings)[1]){
	h[as.character(ratings[i,1),
	as.character(ratings[i,2])]=1
}

#Stochastic Gradient Descent
# a function returns a list containing factorized matrices p and q, training and testing RMSEs.
gradesc <- function(f = 10, 
                    lrate = 0.01, lambda1=0.3, lambda2=0.4, max.iter, stopping.deriv = 0.01,
                    data, train, test){
  set.seed(0)
  #random assign value to matrix p and q
  p <- matrix(runif(f*U, -1, 1), ncol = U) 
  colnames(p) <- as.character(1:U)
  #p_sig<-sd(p)^2
  q <- matrix(runif(f*I, -1, 1), ncol = I)
  colnames(q) <- levels(as.factor(data$movieId))
  
  #q_sig<-sd(q)^2
  #r_ui_hat<-t(q) %*% p

  
  
  train_RMSE <- c()
  test_RMSE <- c()

  
  for(l in 1:max.iter){
    #sample_idx <- sample(1:nrow(train), nrow(train))
    #loop through each training case and perform update
    u <- as.character(train[,1])
    
    i <- as.character(train[,2])
    
    r_ui <- train[s,3] 
    e_ui <- ((r_ui - t(q[,i]) %*% p[,u])*h)
    
    grad_q <- e_ui %*% p[,u] - lambda1 * q[,i]
    if (all(abs(grad_q) > stopping.deriv, na.rm = T)){
    	q[,i] <- q[,i] + lrate * grad_q
    	}
    grad_p <-((   e_ui %*% q[,i]) - lambda2 * p    ) 
    if (all(abs(grad_p) > stopping.deriv, na.rm = T)){
    	p[,u] <- p[,u] + lrate * grad_p
    	}
    }
        	
      

    #print the values of training and testing RMSE
    if (l %% 10 == 0){
      cat("epoch:", l, "\t")
      est_rating <- t(q) %*% p
      rownames(est_rating) <- levels(as.factor(data$movieId))
      
      train_RMSE_cur <- RMSE(train, est_rating)
      cat("training RMSE:", train_RMSE_cur, "\t")
      train_RMSE <- c(train_RMSE, train_RMSE_cur)
      
      test_RMSE_cur <- RMSE(test, est_rating)
      cat("test RMSE:",test_RMSE_cur, "\n")
      test_RMSE <- c(test_RMSE, test_RMSE_cur)
    } 
  }
  
  return(list(p = p, q = q, train_RMSE = train_RMSE, test_RMSE = test_RMSE))
}
