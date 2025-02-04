import tensorflow as tf
import keras
from dataclasses import dataclass, field

import numpy as np
from typing import List
import json
import argparse
import time
import pprint
import os



# Helper functions





# correlator classes and functions
# partially former content of correlator.py


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
    with np.errstate(divide='ignore'):
        ker = np.cosh(Omega * (Position-1/(2*T))) / np.sinh(Omega/2/T)

        # set all entries in ker to 1 where Position is modulo 1/T and the entry is nan, because of numerical instability for large Omega
        ker[np.isnan(ker) & (Position % (1/T) == 0)] = 1
        #set all other nan entries to 0
        ker[np.isnan(ker)] = 0

        # ker[(Position%1/T==0)]=1
        # ker[(Position[:,0]!=0)]=0
    return ker

def KL_kernel_Omega(KL,x,Omega,args=[]):
    ret=KL(x, Omega, *args)
    ret[:,Omega==0]=1
    ret=Omega * ret
    # set for all Omega=0 to 1
    if len(args)==0:
        ret[:,Omega==0]=0
    else:
        ret[:,Omega==0]=2*args[0]
    return ret
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



# neural network classes
# former content of models.py

class SpectralNN(tf.keras.Model):
    def __init__(self, num_output_nodes, width=[32]):
        super(SpectralNN, self).__init__()
        
        # Create hidden layers
        self.hidden_layers = []
        for w in width:
            self.hidden_layers.append(tf.keras.layers.Dense(w, activation='elu', use_bias=False))
        
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
    def __init__(self, num_output_nodes, width=[32]):
        super(SpectralNNP2P, self).__init__()

        # Create hidden layers
        self.hidden_layers = []
        for w in width:
            self.hidden_layers.append(tf.keras.layers.Dense(w, activation='elu'))
        
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

# loss calculator classes and functions
# partially former content of lossCalculator.py

def l2_regularization( weights):
    return tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in weights])

def smoothness_loss(rho):
    diff = rho[:, 1:] - rho[:, :-1]
    return tf.reduce_sum(tf.square(diff))
    
def custom_loss( y_pred, y_true,std):
    # Ensure both y_true and y_pred are of type float32
    std = tf.cast(std, dtype=tf.float32)
    std = std/std.numpy()[0]
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    chi_squared = tf.square((y_true - y_pred) / std)
    chi_squared = tf.reduce_mean(chi_squared)
   
    return chi_squared  # Chi-squared loss

def total_loss(y_pred, y_true=None, std=None, rho=None, model=None,
                lambda_s=None, lambda_l2=None):

    # Smoothness loss
    smooth_loss = smoothness_loss(rho)
    
    # L2 loss (regularization on the network weights)
    l2_loss = l2_regularization(model.trainable_weights)


    #main_loss
    main_loss = custom_loss(y_pred, y_true,std)
    
    # Total loss = main loss + smoothness regularizer + L2 regularization
    return main_loss + lambda_s * smooth_loss + (lambda_l2 * l2_loss)*0.5,[main_loss,smooth_loss,l2_loss]

class LossCalculator:
    def __init__(self, model=None,y_true=None,std=None,kernel=None,
                 delomega=None,
                 x=None,
                 lambda_s_func=lambda x:0.0,
                 lambda_l2_func=lambda x:0.0,):
        self.model = model
        self.y_true = y_true
        self.x=x
        self.std = std

        if self.std is None:
            self.std = tf.constant(1.0, dtype=tf.float32)
        else:
            self.std = tf.cast(self.std, dtype=tf.float32)
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
        return custom_loss(y_pred,self.y_true,self.std)

    def total_loss(self,epoch,y_pred=None,rho=None,y_true=None):
        if rho is None:
            rho=self.model(self.x)
        if y_pred is None:
            y_pred = Di(self.kernel, rho, self.delomega)
        if y_true is None:
            y_true = self.y_true

        return total_loss(y_pred,y_true=y_true,std=self.std,
                           rho=rho, model=self.model, lambda_s=self.get_lambda_s(epoch),
                             lambda_l2=self.get_lambda_l2(epoch))
    

# Interface and runner classes

@dataclass
class networkParameters:
    lambda_s: List = field(default_factory=list)
    lambda_l2: List = field(default_factory=list)
    epochs: List = field(default_factory=list)
    learning_rate: List = field(default_factory=list)
    width: List = field(default_factory=list)
    errorWeighting: bool = False
    networkStructure: str = "SpectralNN"

    def __post_init__(self):
        # Ensure lambda_s, lambda_l2, epochs, and learning_rate are lists of the same length
        if not (len(self.lambda_s) == len(self.lambda_l2) == len(self.epochs) == len(self.learning_rate)):
            raise ValueError("lambda_s, lambda_l2, epochs, and learning_rate must be lists of the same length.")
        
        # Ensure all entries in lambda_s, lambda_l2 and learning_rate are floats
        if not all(isinstance(item, float) for item in self.lambda_s):
            raise ValueError("All entries in lambda_s must be floats.")
        if not all(isinstance(item, float) for item in self.lambda_l2):
            raise ValueError("All entries in lambda_l2 must be floats.")
        if not all(isinstance(item, float) for item in self.learning_rate):
            raise ValueError("All entries in learning_rate must be floats.")
        
        # Ensure all entries in epochs are integers
        if not all(isinstance(item, int) for item in self.epochs):
            raise ValueError("All entries in epochs must be integers.")
        
        #Ensure that width is a list of integers 
        if not all(isinstance(item, int) for item in self.width):
            raise ValueError("All entries in width must be integers.")
        

   

class neuralFit:
    def __init__(self,networkParameters:networkParameters):
        self.lambda_s=networkParameters.lambda_s
        self.lambda_l2=networkParameters.lambda_l2
        self.epochs=networkParameters.epochs
        self.learning_rate=networkParameters.learning_rate
        self.width=networkParameters.width
        self.depth=len(self.width)
        self.errorWeighting=networkParameters.errorWeighting
        self.networkStructure=networkParameters.networkStructure


    def initKernel(self,extractedQuantity:str,finiteT_kernel:bool,Nt:int,x:np.ndarray,omega:np.ndarray):
        if extractedQuantity=="RhoOverOmega" and finiteT_kernel:
            kernel=KL_kernel_Omega(KL_kernel_Position_FiniteT,x,omega,args=(1/Nt,))
        elif extractedQuantity=="RhoOverOmega" and finiteT_kernel==False:
            kernel=KL_kernel_Omega(KL_kernel_Position_Vacuum,x,omega)
        elif extractedQuantity=="Rho" and finiteT_kernel:
            kernel=KL_kernel_Position_FiniteT(x,omega,1/Nt)
        elif extractedQuantity=="Rho" and finiteT_kernel==False:
            kernel=KL_kernel_Position_Vacuum(x,omega)
        else:
            raise ValueError("Invalid choice spectral function target")
        return kernel

    def fitCorrelator(self,x,error,correlator,finiteT_kernel,Nt,omega,extractedQuantity="RhoOverOmega",verbose=True):

        kernel=self.initKernel(extractedQuantity,finiteT_kernel,Nt,x,omega)
        del_omega=omega[1]-omega[0]
        # pprint.pprint(kernel[1])

        if self.errorWeighting:
            errorWeight=error
        else:
            errorWeight=np.ones(len(x))

        constant_input = tf.constant([[1.0]], dtype=tf.float32) #NN

        if self.networkStructure == "SpectralNN":
            model = SpectralNN(num_output_nodes=len(omega), width=self.width)
        elif self.networkStructure == "SpectralNNP2P":
            model = SpectralNNP2P(num_output_nodes=len(omega), width=self.width, lambda_s=self.lambda_s[0], lambda_l2=self.lambda_l2[0], kl_ker=kernel, del_omega=del_omega, gauss_width=errorWeight)
        else:
            raise ValueError("Invalid choice of network")
        target_output=correlator
        lossCalc=LossCalculator(model=model,
                                            y_true=target_output,
                                            kernel=kernel,
                                            delomega=del_omega,
                                            x=constant_input,
                                            std=errorWeight,
                                            lambda_s_func=lambda x:self.lambda_s[0],
                                            lambda_l2_func=lambda x:self.lambda_l2[0]
                                            )

        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate[0])
        trainer=networkTrainer(model,optimizer,lossCalc)
        # total_loss_history,loss_history=trainer.train(self.epochs[0],warmup=True,verbose=verbose)
        total_loss_history=[]
        loss_history=[]

        for lambda_s,lambda_l2,learning_rate,epochs in zip(self.lambda_s,self.lambda_l2,self.learning_rate,self.epochs):
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
            lossCalc.lambda_s_func=lambda x:lambda_s
            lossCalc.lambda_l2_func=lambda x:lambda_l2
            trainer.optimizer=optimizer
            total_loss_history_tmp,loss_history_tmp=trainer.train(epochs,verbose=verbose)
            total_loss_history.extend(total_loss_history_tmp)
            loss_history.extend(loss_history_tmp)
            if verbose:
                print("-"*40)


        spectralFunction=model(constant_input)
        return np.squeeze(spectralFunction)
   
class ParameterHandler:
    def __init__(self, paramsDefaultDict):
        self.allowed_params = paramsDefaultDict.keys()
        self.params = paramsDefaultDict

    def load_from_json(self, config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
        for name in self.allowed_params:
            if name in data:
                self.params[name] = data[name]

    def override_with_args(self, args):
        for name in self.allowed_params:
            val = getattr(args, name, None)
            if val is not None:
                self.params[name] = val

    def check_parameters(self):
        for name in self.allowed_params:
            if name == "outputFile" and self.params[name] is None:
              continue
            elif name not in self.params or self.params[name] is None:
                raise ValueError(f"Parameter '{name}' is not set.")

    def load_params(self, config_path,args):
        self.load_from_json(config_path)
        self.override_with_args(args)
        self.check_parameters()

    def get_params(self):
        return self.params
    
    def get_width(self):
        width = self.params["width"]
        return [width] if isinstance(width, int) else width
    
    def getNetworkParams(self):
        return networkParameters(
            lambda_s=self.params["lambda_s"],
            lambda_l2=self.params["lambda_l2"],
            epochs=self.params["epochs"],
            learning_rate=self.params["learning_rate"],
            width=self.get_width(),
            errorWeighting=self.params["errorWeighting"],
            networkStructure=self.params["networkStructure"]
        )
    
    def get_extractedQuantity(self):
        return self.params["extractedQuantity"]
    
    def get_correlator_file(self):
        return os.path.abspath(self.params["correlatorFile"])
    

    def get_verbose(self):
        return self.params["verbose"]

    def get_correlator_cols(self):
        correlator_cols = self.params["correlatorCols"]
        if isinstance(correlator_cols, list):
            return correlator_cols
        elif isinstance(correlator_cols,int):
            return [correlator_cols]
        elif isinstance(correlator_cols, str) and ':' in correlator_cols:
            start_str, end_str = correlator_cols.split(':')
            start = int(start_str) if start_str else None
            end = int(end_str) if end_str else None
            return list(range(start if start is not None else 0, end + 1 if end is not None else len(np.loadtxt(self.params["correlatorFile"], max_rows=1))))
        else:
            raise ValueError("correlator_cols must be a integer index or list of indices or a string with a range (e.g., '6:10', '6:', ':10', ':').")   
class FitRunner:
    def __init__(self, parameterHandler):
        self.parameterHandler=parameterHandler
        self.net_params = self.parameterHandler.getNetworkParams()

        self.x, self.mean, self.error, self.correlators = self.extractColumns(
            self.parameterHandler.get_correlator_file(),
            self.parameterHandler.get_params()["xCol"],
            self.parameterHandler.get_params()["meanCol"],
            self.parameterHandler.get_params()["errorCol"],
            self.parameterHandler.get_correlator_cols()
            )

        self.omega = np.linspace(
            self.parameterHandler.get_params()["omega_min"],
            self.parameterHandler.get_params()["omega_max"],
            self.parameterHandler.get_params()["omega_points"]
        )

        self.finiteT_kernel=self.parameterHandler.get_params()["FiniteT_kernel"]

        self.verbose = self.parameterHandler.get_verbose()

        self.multiFit = self.parameterHandler.get_params()["multiFit"]
        self.multiFitBootstrap_samples = self.parameterHandler.get_params()["multiFitBootstrap_samples"]

        self.extractedQuantity = self.parameterHandler.get_extractedQuantity()

        self.Nt = self.parameterHandler.get_params()["Nt"]
        if self.Nt==0:
            self.Nt=len(self.x)

        self.outputDir = os.path.abspath(self.parameterHandler.get_params()["outputDir"])
        if self.parameterHandler.get_params()["outputFile"] is None:
            self.outputFile = f"{self.extractedQuantity}_{os.path.basename(self.parameterHandler.get_correlator_file())}" 
        else:
            self.outputFile = os.path.basename(self.parameterHandler.get_params()["outputFile"])

        
    
    def extractColumns(self, file, x_col, mean_col, error_col, correlator_cols):
        data = np.loadtxt(file)
        x = data[:, x_col]
        mean = data[:, mean_col]
        error = data[:, error_col]
        correlator = data[:, correlator_cols]
        return x, mean, error, correlator

    def run_fits(self):
        fitter = neuralFit(self.net_params)
        results = []

        if self.correlators.ndim == 1:
            self.correlators = np.array([self.correlators])
        else:
            self.correlators = self.correlators.T
        

        n_correlators = self.correlators.shape[0]
        
        if self.multiFit:
            if self.multiFitBootstrap_samples == 0:
                start_time = time.time()
                print("="*40)
                print(f"Multifitting {len(self.correlators)} correlators")
                print("="*40)
                sf = fitter.fitCorrelator(
                    self.x,
                    self.error,
                    self.correlators,
                    self.finiteT_kernel,
                    self.Nt,
                    self.omega,
                    extractedQuantity=self.extractedQuantity,
                    verbose=self.verbose
                )                 
                if self.verbose:
                    print("="*40)
                    print(f"Training time: {time.time()-start_time:.2f} seconds")
                results.append(sf)
            else:
                for i in range(self.multiFitBootstrap_samples):
                    #pick n_correlators random correlators from self.correlators 
                    random_correlators = self.correlators[np.random.choice(n_correlators, n_correlators, replace=True)]
                    start_time = time.time()
                    print("="*40)
                    print(f"Multifitting {len(self.correlators)} correlators with bootstrap sample {i+1}/{self.multiFitBootstrap_samples}")
                    print("="*40)
                    sf = fitter.fitCorrelator(
                        self.x,
                        self.error,
                        random_correlators,
                        self.finiteT_kernel,
                        self.Nt,
                        self.omega,
                        extractedQuantity=self.extractedQuantity,
                        verbose=self.verbose
                    )                 
                    if self.verbose:
                        print("="*40)
                        print(f"Training time: {time.time()-start_time:.2f} seconds")
                    results.append(sf)
        else:
            for i,corr in enumerate(self.correlators):
                start_time = time.time()
                print("="*40)
                print(f"Fitting correlator {i+1}/{len(self.correlators)}")
                print("="*40)
                sf = fitter.fitCorrelator(
                    self.x,
                    self.error,
                    corr,
                    self.finiteT_kernel,
                    self.Nt,
                    self.omega,
                    extractedQuantity=self.extractedQuantity,
                    verbose=self.verbose
                )                 
                if self.verbose:
                    print("-"*40)
                    print(f"Training time: {time.time()-start_time:.2f} seconds")
                results.append(sf)
        return np.array(results)
    
    def calculate_mean_error(self, results):
        N=len(results)
        mean=np.mean(results,axis=0)
        error=np.sqrt((N-1)*np.sum((results-mean)**2,axis=0)/N)
        return mean,error
    
    def save_results(self, mean,error,results,extractedQuantity="RhoOverOmega"):
        header ="# Omega "+extractedQuantity+"_mean "+extractedQuantity+"_error"
        for i in range(len(results)):
            header += f" {extractedQuantity}_sample_{i}"
        writeData = np.column_stack((self.omega,mean,error,results.T))
        np.savetxt(os.path.join(self.outputDir,self.outputFile), writeData, header=header)

        if self.parameterHandler.get_params()["saveParams"]:
            self.save_params(self.parameterHandler.get_params(),os.path.join(self.outputDir,self.outputFile+".params"))
    
    def save_loss_history(self, loss_history, outputFile):
        return None
    
    def save_params(self, params, outputFile):
        with open(outputFile+'.json', 'w') as f:
            json.dump(params, f, indent=4)
    
def initializeArgumentParser(paramsDefaultDict):
    parser = argparse.ArgumentParser(
        prog="neuralFit",
        description="Fit spectral functions to provided correlators using a neural network."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )
    
    for name, default in paramsDefaultDict.items():
        typeArg=type(default)
        typeString=typeArg.__name__
        if typeArg==list:
            nargsArg='+'
            typeArg=type(default[0])
            typeString=f"List of {typeArg.__name__}"
        else:
            nargsArg=None
        parser.add_argument(
            f"--{name}",
            type=typeArg,
            nargs=nargsArg,
            # default=default,
            help=f"Value for parameter '{name}'"
        )
    return parser

def main(paramsDefaultDict):
    parser=initializeArgumentParser(paramsDefaultDict)
    args = parser.parse_args()

    parameterHandler = ParameterHandler(paramsDefaultDict)
    parameterHandler.load_params(args.config,args)

    if parameterHandler.get_verbose():
        print("*"*40)
        print("Running fits with the following parameters:")
        pprint.pprint(parameterHandler.get_params())

    fitRunner = FitRunner(parameterHandler)
    results = fitRunner.run_fits()
    mean,error = fitRunner.calculate_mean_error(results)
    fitRunner.save_results(mean,error,results)



paramsDefaultDict = {
    #NetworkParams
    "lambda_s": [1e-5],
    "lambda_l2": [1e-8],
    "epochs": [100],
    "learning_rate": [1e-4],
    "width": [32,32,32],
    "errorWeighting": True,
    "networkStructure": "SpectralNN",
    #Correlator/Rho params
    "omega_min": 0,
    "omega_max": 10,
    "omega_points": 500,
    "Nt": 0,
    "extractedQuantity": "RhoOverOmega",
    "FiniteT_kernel": True,
    "multiFit": False,
    "multiFitBootstrap_samples": 0,
    "seed": 1,
    "correlatorFile": None,
    "xCol": 0,
    "meanCol": 1,
    "errorCol": 2,
    "correlatorCols": "3:",
    #General Params
    "saveParams": False,
    "saveLossHistory": False,
    "verbose": False,
    "outputFile": None,
    "outputDir": ''

}


#TODOs
# X Multifit bootstrap
# X implement change of network architecture
# X choosing zeroT or FiniteT kernel
# - always fit mean and use this to give the mean column in the output file
# - implement both jackknife and bootstrap error estimation
# -- calculate error from samples
# X width as list for adaptive network width
# X check that training stage lists are of equal length
# X find better name for 'which' parameter
# X Nt as explicit parameter
# - make such that correlatorfile and outputfile can handle relative paths
# - way to save loss history
# X way to save parameters
# - check parameter handling and checking
# - implement error handling
# X merge verything into one file
# - make documentation
# -- file parameters can be relative, but relative to cwd
# -- also some part about tensorflow installation and python virtual environments
# ----------------
# - make momentum kernel accesible
# - implement multiprocessing
# - implement not only error weighting but correlator mean value weighting
if __name__ == "__main__":
    main(paramsDefaultDict)