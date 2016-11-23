from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

if sys.version_info.major == 3:
    raw_input = input


def mogEM(x, K, iters, randConst=1, minVary=0):
    """
    Fits a Mixture of K Diagonal Gaussians on x.

    Inputs:
      x: data with one data vector in each column.
      K: Number of Gaussians.
      iters: Number of EM iterations.
      randConst: scalar to control the initial mixing coefficients
      minVary: minimum variance of each Gaussian.

    Returns:
      p: probabilities of clusters (or mixing coefficients).
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.
      logLikelihood: log-likelihood of data after every iteration.
    """
    N, T = x.shape

    # Initialize the parameters
    p = randConst + np.random.rand(K, 1)
    p = p / np.sum(p)   # mixing coefficients
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)

    # Question 4.3: change the initializaiton with Kmeans here
    #--------------------  Add your code here --------------------
    kmean_iters = 5
    #mn = KMeans(x, K, kmean_iters)
    mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)

    #------------------- Answers ---------------------

    #------------------------------------------------------------
    vary = vr * np.ones((1, K)) * 2
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logLikelihood = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in xrange(iters):
        # Do the E step
        respTot = np.zeros((K, 1))
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        ivary = 1 / vary
        logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - \
            0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
        logPcAndx = np.zeros((K, T))
        for k in xrange(K):
            dis = (x - mu[:, k].reshape(-1, 1))**2
            logPcAndx[k, :] = logNorm[k] - 0.5 * \
                np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logLikelihood[i] = np.sum(np.log(Px) + mx)

        print 'Iter %d logLikelihood %.5f' % (i + 1, logLikelihood[i])

        # Plot log likelihood of data
        plt.figure(0)
        plt.clf()
        plt.plot(np.arange(i), logLikelihood[:i], 'r-')
        plt.title('Log-likelihood of data versus # iterations of EM')
        plt.xlabel('Iterations of EM')
        plt.ylabel('Log-likelihood')
        plt.draw()

        # Do the M step
        # update mixing coefficients
        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        p = respTot

        # update mean
        respX = np.zeros((N, K))
        for k in xrange(K):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)

        mu = respX / respTot.T

        # update variance
        respDist = np.zeros((N, K))
        for k in xrange(K):
            respDist[:, k] = np.mean(
                (x - mu[:, k].reshape(-1, 1))**2 * PcGivenx[k, :].reshape(1, -1), axis=1)

        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    return p, mu, vary, logLikelihood


def mogLogLikelihood(p, mu, vary, x):
    """ Computes log-likelihood of data under the specified MoG model

    Inputs:
      x: data with one data vector in each column.
      p: probabilities of clusters.
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.

    Returns:
      logLikelihood: log-likelihood of data after every iteration.
    """
    K = p.shape[0]
    N, T = x.shape
    ivary = 1 / vary
    logLikelihood = np.zeros(T)
    for t in xrange(T):
        # Compute log P(c)p(x|c) and then log p(x)
        logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
            - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
            - 0.5 * \
            np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)
                   ** 2, axis=0).reshape(-1, 1)

        mx = np.max(logPcAndx, axis=0)
        logLikelihood[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx

    return logLikelihood


def ShowMeans(means, number=0):
  """Show the cluster centers as images."""
  plt.figure(number)
  plt.clf()
  for i in xrange(means.shape[1]):
    plt.subplot(1, means.shape[1], i+1)
    plt.imshow(means[:, i].reshape(48, 48), cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')

def q2():
    # Question 4.2 and 4.3
    K = 7
    iters = 10
    minVary = 0.01
    randConst = 1

    # load data
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData(
        '../toronto_face.npz')

    # Train a MoG model with 7 components on all training data, i.e., inputs_train,
    # with both original initialization and kmeans initialization.
    #------------------- Add your code here ---------------------
    p, mu, vary, logLikelihood = mogEM(inputs_train, K, iters, randConst, minVary)
    #print "p:"
    #print p
    #ShowMeans(mu, number=0)
    #ShowMeans(vary, number=0)

    #------------------- Answers ---------------------


def q4():
    # Question 4.4
    iters = 10
    minVary = 0.01
    randConst = 1.0

    numComponents = np.array([7, 14, 21, 28, 35])
    T = numComponents.shape[0]
    errorTrain = np.zeros(T)
    errorTest = np.zeros(T)
    errorValidation = np.zeros(T)

    # extract data of class 1-Anger, 4-Happy
    dataQ4 = LoadDataQ4('../toronto_face.npz')
    # images
    x_train_anger = dataQ4['x_train_anger']
    x_train_happy = dataQ4['x_train_happy']
    x_train = np.concatenate([x_train_anger, x_train_happy], axis=1)
    x_valid = np.concatenate(
        [dataQ4['x_valid_anger'], dataQ4['x_valid_happy']], axis=1)
    x_test = np.concatenate(
        [dataQ4['x_test_anger'], dataQ4['x_test_happy']], axis=1)

    # label
    y_train = np.concatenate(
        [dataQ4['y_train_anger'], dataQ4['y_train_happy']])
    #print("x_train.shape: ", x_train.shape)
    #print("x_train_anger.shape: ", x_train_anger.shape)
    y_valid = np.concatenate(
        [dataQ4['y_valid_anger'], dataQ4['y_valid_happy']])
    y_test = np.concatenate([dataQ4['y_test_anger'], dataQ4['y_test_happy']])
    #print("y_test", y_test)

    # Hints: this is p(d), use it based on Bayes Theorem
    num_anger_train = x_train_anger.shape[1]
    num_happy_train = x_train_happy.shape[1]
    log_likelihood_class = np.log(
        [num_anger_train, num_happy_train]) - np.log(num_anger_train + num_happy_train)

    num_anger_val = dataQ4['x_valid_anger'].shape[1]
    num_happy_val = dataQ4['x_valid_happy'].shape[1]
    log_likelihood_class_val = np.log(
        [num_anger_val, num_happy_val]) - np.log(num_anger_val + num_happy_val)

    num_anger_test = dataQ4['x_test_anger'].shape[1]
    num_happy_test = dataQ4['x_test_happy'].shape[1]
    log_likelihood_class_test = np.log(
        [num_anger_test, num_happy_test]) - np.log(num_anger_test + num_happy_test)

    for t in xrange(T):
        K = numComponents[t]

        # Train a MoG model with K components
        # Hints: using (x_train_anger, x_train_happy) train 2 MoGs
        #-------------------- Add your code here ------------------------------
        p_anger, mu_anger, vary_anger, logLikelihood_anger = mogEM(x_train_anger, K, iters, randConst, minVary)
        p_happy, mu_happy, vary_happy, logLikelihood_happy = mogEM(x_train_happy, K, iters, randConst, minVary)
        #------------------- Answers ---------------------

        # Compute the probability P(d|x), classify examples, and compute error rate
        # Hints: using (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
        # to compute error rates, you may want to use mogLogLikelihood function
        #-------------------- Add your code here ------------------------------
        loganger_train = mogLogLikelihood(p_anger, mu_anger, vary_anger, x_train)
        loghappy_train = mogLogLikelihood(p_happy, mu_happy, vary_happy, x_train)
        errorTrain[t] = calErrorRate(loganger_train+log_likelihood_class[0], loghappy_train+log_likelihood_class[1], y_train)
       
        loganger_val = mogLogLikelihood(p_anger, mu_anger, vary_anger, x_valid)
        loghappy_val = mogLogLikelihood(p_happy, mu_happy, vary_happy, x_valid)
        errorValidation[t] = calErrorRate(loganger_val+log_likelihood_class_val[0], loghappy_val+log_likelihood_class_val[1], y_valid)
        
        loganger_test = mogLogLikelihood(p_anger, mu_anger, vary_anger, x_test)
        loghappy_test = mogLogLikelihood(p_happy, mu_happy, vary_happy, x_test)
        errorTest[t] = calErrorRate(loganger_test+log_likelihood_class_test[0], loghappy_test+log_likelihood_class_test[1], y_test)
      
        #------------------- Answers ---------------------

    # Plot the error rate
    plt.figure(0)
    plt.clf()
    #-------------------- Add your code here --------------------------------

    #------------------- Answers ---------------------
    # to be removed before release
    plt.plot(numComponents, errorTrain, 'r', label='Training')
    plt.plot(numComponents, errorValidation, 'g', label='Validation')
    plt.plot(numComponents, errorTest, 'b', label='Testing')
    plt.xlabel('Number of Mixture Components')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.draw()
    plt.pause(0.0001)

def calErrorRate(loganger, loghappy, y):
    T = y.shape[0]
    err_cnt = 0
    for i in xrange(T):
        if ((loganger[i] > loghappy[i]) and y[i] == 1) or ((loganger[i] < loghappy[i]) and y[i] == 0):
            err_cnt += 1
    return float(err_cnt)/float(T)


if __name__ == '__main__':
    #-------------------------------------------------------------------------
    # Note: Question 4.2 and 4.3 both need to call function q2
    # you need to comment function q4 below
    #q2()

    #-------------------------------------------------------------------------
    # Note: Question 4.4 both need to call function q4
    # you need to comment function q2 above
    q4()

    raw_input('Press Enter to continue.')
