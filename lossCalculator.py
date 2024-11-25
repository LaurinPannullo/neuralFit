import numpy as np
import tensorflow as tf
import correlator

def l2_regularization( weights):
    return tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in weights])


def smoothness_loss(rho):
    diff = rho[:, 1:] - rho[:, :-1]
    return tf.reduce_sum(tf.square(diff))
    
def custom_loss( y_pred, y_true,gauss_width):
    # Ensure both y_true and y_pred are of type float32
    gauss_width = tf.cast(gauss_width, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    chi_squared = tf.square((y_true - y_pred) / gauss_width)
    chi_squared = tf.reduce_mean(chi_squared)
   
    return chi_squared  # Chi-squared loss


def total_loss(y_pred, y_true=None, gauss_width=None, rho=None, model=None, lambda_s=None, lambda_l2=None):

    # Smoothness loss
    smooth_loss = smoothness_loss(rho)
    
    # L2 loss (regularization on the network weights)
    l2_loss = l2_regularization(model.trainable_weights)

    #main_loss
    main_loss = custom_loss(y_pred, y_true,gauss_width)
    
    # Total loss = main loss + smoothness regularizer + L2 regularization
    print("LOSS CALCULATOR",main_loss,smooth_loss,l2_loss)
    return main_loss + lambda_s * smooth_loss + (lambda_l2 * l2_loss)*0.5



class LossCalculator:
    def __init__(self, model=None,y_true=None,gauss_width=None,kernel=None,delomega=None,x=None,lambda_s_func=None,lambda_l2_func=None):
        self.model = model
        self.y_true = y_true
        self.x=x
        self.gauss_width = gauss_width

        if self.gauss_width is None:
            self.gauss_width = tf.constant(1.0, dtype=tf.float32)
        else:
            self.gauss_width = tf.cast(self.gauss_width, dtype=tf.float32)
        self.kernel = kernel
        self.delomega = delomega
        self.lambda_s_func = lambda_s_func
        self.lambda_l2_func = lambda_l2_func

    def get_lambda_s(self,epoch):
        return self.lambda_s_func(epoch)
    
    def get_lambda_l2(self,epoch):
        return self.lambda_l2_func(epoch)
    
    def l2_regularization(self):
        return self.l2_regularization(self.model.trainable_weights)
    
    def smoothness_loss(self,rho=None):
        if rho is None:
            rho=self.model(self.x)
        return smoothness_loss(rho)
    
    def custom_loss(self, y_pred,y_true=None):
        if y_true is None:
            y_true = self.y_true
        return custom_loss(y_pred,self.y_true,self.gauss_width)

    def total_loss(self,epoch,y_pred=None,rho=None,y_true=None):
        if rho is None:
            rho=self.model(self.x)
        if y_pred is None:
            y_pred = correlator.Di(self.kernel, rho, self.delomega)
        if y_true is None:
            y_true = self.y_true

        return total_loss(y_pred,y_true=y_true,gauss_width=self.gauss_width,rho=rho, model=self.model, lambda_s=self.get_lambda_s(epoch), lambda_l2=self.get_lambda_l2(epoch))
    

    