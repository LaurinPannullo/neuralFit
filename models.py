import tensorflow as tf
import keras
import numpy as np
import time


# Setting up the Neural Network
# NN style - Define the neural network architecture one node to 500 - base class
class SpectralNN(tf.keras.Model):
    def __init__(self, num_output_nodes, width=64, depth=3):
        super(SpectralNN, self).__init__()
        
        # Create hidden layers
        self.hidden_layers = []
        for _ in range(depth):
            self.hidden_layers.append(tf.keras.layers.Dense(width, activation='elu', use_bias=False))
        
        # Output layer (softplus activation to ensure positive definiteness)
        self.output_layer = tf.keras.layers.Dense(num_output_nodes, activation=tf.keras.activations.softplus, use_bias=False)  # Shape [500]
    
    def call(self, inputs):
        # Forward pass through hidden layers
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output layer (representing rho(omega))
        output = self.output_layer(x)  # Shape [500]
        return output


class SpectralNNP2P(tf.keras.Model):
    def __init__(self, num_output_nodes, width=64, depth=3, lambda_s=None, lambda_l2=None, kl_ker=None, del_omega=None, gauss_width=None):
        super(SpectralNNP2P, self).__init__()
        self.lambda_s = lambda_s
        self.lambda_l2 = lambda_l2
        self.kl_ker = kl_ker
        self.del_omega = del_omega
        self.gauss_width = gauss_width
        
        # Create hidden layers
        self.hidden_layers = []
        for _ in range(depth):
            self.hidden_layers.append(tf.keras.layers.Dense(width, activation='elu'))
        
        # Output layer that produces one output per frequency (500 outputs)
        self.output_layer = tf.keras.layers.Dense(1, activation='softplus')
    
    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
class networkTrainer:
    def __init__(self, model, optimizer, loss_calculator):
        self.model = model
        self.optimizer = optimizer
        self.loss_calculator = loss_calculator
    
    def train_step(self, epoch):
        with tf.GradientTape() as tape:
            total_loss_value,individual_losses = self.loss_calculator.total_loss(epoch)

        # Compute gradients and update weights
        gradients = tape.gradient(total_loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
    
        return total_loss_value,individual_losses
    
    def train(self, num_epochs,verbose=False,start_epoch=0):
        losses = []
        individual_losses_history = []
        net_num_epochs = num_epochs-start_epoch
        if verbose:
            print(f'Training for {net_num_epochs} epochs')
            start_time = time.time()
        for epoch in range(start_epoch,start_epoch+num_epochs):
            total_loss_value,individual_losses = self.train_step(epoch)
            losses.append(total_loss_value)
            individual_losses_history.append(individual_losses)
            if verbose and net_num_epochs>10 and epoch % (net_num_epochs//10) == 0:
                print(f'Epoch {epoch}, Loss: {total_loss_value}')
        if verbose:
            end_time = time.time()
            print(f'Training took {end_time-start_time:.2f} seconds')
        return losses,individual_losses_history