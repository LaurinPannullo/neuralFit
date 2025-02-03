


import models
import lossCalculator
import correlator
import tensorflow as tf
from dataclasses import dataclass, field

import numpy as np
from typing import List
import json
import argparse

@dataclass
class networkParameters:
    lambda_s: List = field(default_factory=list)
    lambda_l2: List = field(default_factory=list)
    epochs: List = field(default_factory=list)
    learning_rate: List = field(default_factory=list)
    width: int = 0
    depth: int = 0
    errorWeighting: bool = False
   
   

class neuralFit:
    def __init__(self,networkParameters:networkParameters):
        self.lambda_s=networkParameters.lambda_s
        self.lambda_l2=networkParameters.lambda_l2
        self.epochs=networkParameters.epochs
        self.learning_rate=networkParameters.learning_rate
        self.width=networkParameters.width
        self.depth=networkParameters.depth
        self.errorWeighting=networkParameters.errorWeighting


    def initKernel(self,which:str,Nt:int,x:np.ndarray,omega:np.ndarray):
        if which=="RhoOverOmega" and Nt>0:
            kernel=correlator.KL_kernel_Omega(correlator.KL_kernel_Position_FiniteT,x,omega,args=(1/Nt,))
        elif which=="RhoOverOmega" and Nt==0:
            kernel=correlator.KL_kernel_Omega(correlator.KL_kernel_Position_Vacuum,x,omega)
        elif which=="Rho" and Nt>0:
            kernel=correlator.KL_kernel_Position_FiniteT(x,omega,1/Nt)
        elif which=="Rho" and Nt==0:
            kernel=correlator.KL_kernel_Position_Vacuum(x,omega)
        else:
            raise ValueError("Invalid choice spectral function target")
        return kernel

    def fitCorrelator(self,x,error,correlator,Nt,omega,which="RhoOverOmega",verbose=True):

        kernel=self.initKernel(which,Nt,x,omega)
        del_omega=omega[1]-omega[0]


        if self.errorWeighting:
            errorWeight=error
        else:
            errorWeight=np.ones(len(x))

        constant_input = tf.constant([[1.0]], dtype=tf.float32) #NN

        model = models.SpectralNN(num_output_nodes=len(omega), width=self.width, depth=self.depth)
        target_output=correlator
        lossCalc=lossCalculator.LossCalculator(model=model,
                                            y_true=target_output,
                                            kernel=kernel,
                                            delomega=del_omega,
                                            x=constant_input,
                                            std=errorWeight,
                                            lambda_s_func_warmup=lambda x:self.lambda_s[0],
                                            lambda_s_func=lambda x:self.lambda_s[1],
                                            lambda_l2_func_warmup=lambda x:self.lambda_l2[0],
                                            lambda_l2_func=lambda x:self.lambda_l2[1]
                                            )

        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate[0])
        trainer=models.networkTrainer(model,optimizer,lossCalc)
        total_loss_history,loss_history=trainer.train(self.epochs[0],warmup=True,verbose=verbose)

        for lambda_s,lambda_l2,learning_rate,epochs in zip(self.lambda_s[1:],self.lambda_l2[1:],self.learning_rate[1:],self.epochs[1:]):
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
            lossCalc.lambda_s_func=lambda x:lambda_s
            lossCalc.lambda_l2_func=lambda x:lambda_l2
            trainer.optimizer=optimizer
            total_loss_history_tmp,loss_history_tmp=trainer.train(epochs,warmup=False,verbose=verbose)
            total_loss_history.extend(total_loss_history_tmp)
            loss_history.extend(loss_history_tmp)

        spectralFunction=model(constant_input)
        return spectralFunction
   
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
              self.params[name] = f"{self.params['which']}_meanTraining_{self.params['correlatorFile']}"  
            elif name not in self.params or self.params[name] is None:
                raise ValueError(f"Parameter '{name}' is not set.")

    def load_params(self, config_path,args):
        self.load_from_json(config_path)
        self.override_with_args(args)
        self.check_parameters()

    def get_params(self):
        return self.params
    
    def getNetworkParams(self):
        return networkParameters(
            lambda_s=self.params["lambda_s"],
            lambda_l2=self.params["lambda_l2"],
            epochs=self.params["epochs"],
            learning_rate=self.params["learning_rate"],
            width=self.params["width"],
            depth=self.params["depth"],
            errorWeighting=self.params["errorWeighting"]
        )
    
    def get_correlator_file(self):
        return self.params["correlatorFile"]

    def get_verbose(self):
        return self.params["verbose"]

    def get_correlator_cols(self):
        correlator_cols = self.params["correlatorCols"]
        if isinstance(correlator_cols, list):
            return correlator_cols
        elif isinstance(correlator_cols, str) and ':' in correlator_cols:
            start_str, end_str = correlator_cols.split(':')
            start, end = int(start_str), int(end_str)
            return list(range(start, end+1))
        else:
            raise ValueError("correlator_cols must be a list of indices or a string with a range (e.g., '6:10').")
        

    
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

        self.Nt=len(self.x)
    
    def extractColumns(self, file, x_col, mean_col, error_col, correlator_cols):
        data = np.loadtxt(file)
        x = data[:, x_col]
        mean = data[:, mean_col]
        error = data[:, error_col]
        correlator = data[:, correlator_cols]
        return x, mean, error, correlator

    def run_fits(self, which="RhoOverOmega"):
        fitter = neuralFit(self.net_params)
        results = []

        if self.correlators.ndim == 1:
            self.correlators = [self.correlators]
        for corr in self.correlators:
            sf = fitter.fitCorrelator(
                self.x,
                self.error,
                corr,
                self.Nt,
                self.omega,
                which=which,
                verbose=self.parameterHandler.get_verbose()
            )
            results.append(sf.numpy())
        return results
    
    def calculate_mean_error(self, results):
        N=len(results)
        mean=np.mean(results,axis=0)
        error=np.sqrt((N-1)*np.sum((results-mean)**2,axis=0)/N)
        return mean,error
    
    def save_results(self, mean,error,results, outputFile,which="RhoOverOmega"):
        header ="# Omega "+which+"_mean "+which+"_error"
        for i in range(len(results)):
            header += f" {which}_sample_{i}"
        writeData = np.column_stack((self.omega,mean,error,results))
        np.savetxt(outputFile, writeData, header=header)

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
            nargsArg=1
        parser.add_argument(
            f"--{name}",
            type=typeArg,
            nargs=nargsArg,
            default=default,
            help=f"Value for parameter '{name}' of type {typeString} with default {default}"
        )
    return parser

def main(paramsDefaultDict):
    parser=initializeArgumentParser(paramsDefaultDict)
    args = parser.parse_args()

    parameterHandler = ParameterHandler(paramsDefaultDict)
    parameterHandler.load_params(args.config,args)

    fitRunner = FitRunner(parameterHandler)
    results = fitRunner.run_fits()
    mean,error = fitRunner.calculate_mean_error(results)
    fitRunner.save_results(mean,error,results,parameterHandler.get_params()["outputFile"])



paramsDefaultDict = {
    #NetworkParams
    "lambda_s": [1e-5],
    "lambda_l2": [1e-8],
    "epochs": [100],
    "learning_rate": [1e-4],
    "width": 32,
    "depth": 3,
    "errorWeighting": True,
    #Correlator/Rho params
    "omega_min": 0,
    "omega_max": 10,
    "omega_points": 500,
    "which": "RhoOverOmega",
    "correlatorFile": None,
    "xCol": 0,
    "meanCol": 1,
    "errorCol": 2,
    "correlatorCols": [3],
    #General Params
    "verbose": False,
    "outputFile": None
}

if __name__ == "__main__":
    main(paramsDefaultDict)