import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors


def IBU(ymes,t0,Rin,n):    
    #This is the iterative Bayesian unfolding method.
    #Inputs: 
    # ymes: Measured data provided in a histogram with M bins (M,)
    # t0: Prior distribution (M,)
    # Rin: Detector resolution matrix. First coordinate is the measured value and second coordinate is the truth level. (M,M)
    # n:number of iterations.
    #Returns: unfolded predictions (M)

    tn = t0
    for _ in range(n):
        Rjitni = [np.array(Rin[:][i])*tn[i] for i in range(len(tn))]
        Pm_given_t = Rjitni / np.matmul(Rin,tn)
        tn = np.dot(Pm_given_t,ymes)
        pass
    return tn


def MADE(data_shape, cond_shape):
    # Density estimation with MADE.
    made = tfb.AutoregressiveNetwork(params=2, 
                                     hidden_units=[16,16], #To be changed when using bigger histograms
                                     event_shape=data_shape,
                                     activation='swish',
                                     conditional=True,
                                     conditional_event_shape=cond_shape,
                                    )
    distribution = tfd.TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[data_shape]),
        bijector=tfb.MaskedAutoregressiveFlow(made))

    # Construct and fit model.
    x_ = tfkl.Input(shape=(data_shape,), dtype=tf.float32)
    c_ = tfkl.Input(shape=(cond_shape,), dtype=tf.float32)
    log_prob_ = distribution.log_prob(x_, bijector_kwargs={'conditional_input': c_})
    model = tfk.Model([x_,c_], log_prob_)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2),loss=lambda _, log_prob: -log_prob)
    return model, distribution

    batch_size = 512

    # Train
    model.fit(x=[data,cond_train],
              y=np.zeros((len(data),0), dtype=np.float32),
              batch_size=batch_size,
              epochs=20,
              shuffle=True,
              verbose=True)
    # Predict the full background distribution
    samples = distribution.sample(data.shape, bijector_kwargs={'conditional_input': cond_eval})
    return samples 

def MLE(model,ymes,ndim):
    x = tf.Variable(ndim*[1.0/ndim])
    loss = lambda: -model.log_prob(x, bijector_kwargs={'conditional_input': ymes})
    losses = tfp.math.minimize(loss,
                               num_steps=200,
                               #convergence_criterion=(
                               #     tfp.optimizers.convergence_criteria.LossNotDecreasing(atol=0.001)),
                               trainable_variables=[x],
                               optimizer=tf.optimizers.Adam(learning_rate=0.01))
    return x


def NPU(ymes,Rin,N):
    #Inputs: 
    # ymes: Measured data provided in a histogram with M bins (M,)
    # Rin: Detector resolution matrix. First coordinate is the measured value and second coordinate is the truth level. (M,M)
    # N: Total number of observations
    #Returns samples from p(true|measured).  Would normally want the mode over true, which is equivalent to the MLE given p(true) is uniform.
    
    M = 10000 # a big number - should make bigger later
    nsample = 1000
    ts0 = np.random.uniform(0,1,M)*N
    ts = np.c_[ts0,N-ts0]
    
    ms = []
    for j in range(len(ts)):
        m_hold = [np.random.poisson(ts[j][i]) for i in range(len(ts[j]))] #stat fluctuations
        m_hold = np.random.multinomial(m_hold[0],Rin[:,0])+np.random.multinomial(m_hold[1],Rin[:,1]) #resolutions
        ms += [m_hold]
        pass
    ts = np.array(ts)
    ms = np.array(ms)
    
    n = len(ts)
    x = ms #conditional feature
    y = ts #learn p(y|x)
    nx = N 
    ny = N
    
    #Normalize the total number of events to make the NF easier to train
    x = x/float(nx)
    y = y/float(ny)
    
    model,dist = MADE(y.ndim,x.ndim)
    # Fit.
    batch_size = 100
    myhistory = model.fit([y,x], 
                          y=np.zeros((len(x),0), dtype=np.float32), #dummy labels
                          batch_size=batch_size,
                          epochs=100,
                          verbose = 0)
    
    #plt.plot(myhistory.history['loss'][10:-1])
    #plt.xlabel("epochs")
    #plt.ylabel("loss")

    #mle = MLE(dist,ymes/float(nx),y.shape[-1])
    #print(mle)
    mle = MLE(dist,ymes/float(nx),y.shape[-1]).numpy()    
    output = dist.sample(nsample, bijector_kwargs={'conditional_input': np.tile(ymes/float(nx),nsample).reshape([nsample,len(ymes)])}).numpy()    
    return output*ny,mle*ny