import numpy as np
import tensorflow as tf




def KL_kernel_Momentum(Momentum, Omega):
    Momentum = Momentum[:, np.newaxis]  # Reshape Momentum as column to allow broadcasting
    ker = Omega / (Omega**2 + Momentum**2)  # Element-wise division
    return ker / np.pi 

def KL_kernel_Position_Vacuum(Position, Omega):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    ker = np.exp(-Omega * np.abs(Position))
    return ker

def KL_kernel_Position_FiniteT(Position, Omega,T):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    ker = np.cosh(Omega * (1/2 - Position)/T) / np.sinh(Omega/2/T)
    return ker

def KL_kernel_Omega(KL,x,Omega,args=[]):
    return Omega * KL(x, Omega, *args)

def Di(KL, rhoi, delomega):
    # Ensure both tensors are of the same data type (float32)
    KL = tf.cast(KL, dtype=tf.float32)  # Cast KL to float32
    rhoi = tf.cast(rhoi, dtype=tf.float32)  # Cast rhoi to float32
    delomega = tf.cast(delomega, dtype=tf.float32)  # Cast delomega to float32
    
    # Ensure rhoi has the correct shape [500,1] for matrix multiplication
    rhoi = tf.reshape(rhoi, [-1, 1])  # Reshape to [500, 1]

    # Perform matrix multiplication
    dis = tf.matmul(KL, rhoi)  # Shape will be [25, 1]
    dis = tf.squeeze(dis, axis=-1)  # Remove the singleton dimension to get [25]
    
    dis = dis * delomega  # Multiply by delomega
    return dis

def generateNoisyCorrelator(correlator, x, noise_width):
    del_xi= x[1]-x[0] #spacing between x values
    custom_std = noise_width*(correlator*(x+1e-1))/del_xi  #gaussian width
    noisy_Dpi = correlator + np.random.normal(0,custom_std) #random instance of errors
    return noisy_Dpi,custom_std