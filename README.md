# Usage Documentation

## Overview
This script fits correlator data using a neural network with configurable parameters. Below are the available parameters, their possible values, and how they can be passed in JSON or via command-line arguments.

### Possible Network Structures

### Training stages

### Correlator input file format

### Extracted quantities

### Error methods

### Output


## Parameters


| Parameter          | Default          | Type         | Possible Values | Purpose/Comment |
|--------------------|------------------|--------------|------------------------------------------------------|------|
| **lambda_s**       | `[1e-5]`           | List of floats        | List of any floats (e.g., `1e-5`)                             | Coupling for the smoothness loss contribution |
| **lambda_l2**      | `[1e-8]`           | List of floats        | List of any floats                                            | Coupling for the L2 loss contribution |
| **epochs**         | `[100]`            | List of integers      | List of positive integers                                          |
| **learning_rate**  | `[1e-4]`           | List of floats        | List of positive floats float                                            |
| **width**          | `[32, 32, 32]`   | List         | List of integers                                     |
| **errorWeighting** | `True`           | Boolean      | `True`, `False`                                      |
| **networkStructure** | `"SpectralNN"` | String       | `"SpectralNN"`, `"SpectralNNP2P"`        |
| **omega_min**      | `0`              | Float        | Any float                                            |
| **omega_max**      | `10`             | Float        | Any float                                            |
| **omega_points**   | `500`            | Integer      | Any integer                                          |
| **Nt**             | `0`              | Integer      | Any integer                                          |
| **extractedQuantity** | `"RhoOverOmega"` | String    | `"RhoOverOmega"`, `"Rho"`     |
| **FiniteT_kernel** | `True`           | Boolean      | `True`, `False`                                      |
| **multiFit**       | `False`          | Boolean      | `True`, `False`                                      |
| **correlatorFile** | `""`             | String       | Any string path                                      |
| **xCol**           | `0`              | Integer      | Any integer                                          |
| **meanCol**        | `1`              | Integer      | Any integer                                          |
| **errorCol**       | `2`              | Integer      | Any integer                                          |
| **correlatorCols** | `"3:"`           | String       | Any string range                                     |
| **errormethod**    | `"jackknife"`    | String       | `"jackknife"`, `"bootstrap"`                         |
| **saveParams**     | `False`          | Boolean      | `True`, `False`                                      |
| **saveLossHistory** | `False`         | Boolean      | `True`, `False`                                      |
| **verbose**        | `False`          | Boolean      | `True`, `False`                                      |
| **outputFile**     | `""`             | String       | Any string                                           |
| **outputDir**      | `''`             | String       | Any string                                           |

## Passing Parameters
### JSON
Create a JSON file (e.g., `params.json`):
```json
{
    "lambda_s": [1e-6,3.55323189e-05,3.55323189e-05],
    "lambda_l2": [1e-4,6.66754659e-08,6.66754659e-08],
    "epochs": [2000,90000,10000],
    "learning_rate": [1e-3,1e-4,1e-5],
    "errorWeighting": true,
    "networkStructure": "SpectralNN",
    "omega_min": 0,
    "omega_max": 10,
    "omega_points": 500,
    "Nt": 16,
    "extractedQuantity": "RhoOverOmega",
    "FiniteT_kernel": true,
    "multiFit": false,
    "correlatorFile": "correlator.txt",
    "xCol": 0,
    "meanCol": 1,
    "errorCol": 2,
    "correlatorCols": "",
    "errormethod": "jackknife",
    "saveParams": true,
    "saveLossHistory": true,
    "verbose": true,
    "outputFile": null,
    "outputDir": ""
}
```