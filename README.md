# EpiMod

EpiMod is a preliminary data-driven epidemiological modeling tool written in Python.

## Modeling

### Governing equations
We implement the [SEIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model)  which is a set of ordinary differential equations (ODEs) governing infectious diseases:

TODO: add equations, explain terms and variables.

Given an initial condition, we solve the ODEs using numerical integration. 

### Parameter estimation 
TODO: maximum a posteriori estimation, unknown parameters, known uncertain parameters, probability distribution assumptions, uncertainty quantification

### Future predictions
Once the parameters of the SEIR model are estimated based on the provided data, future predictions can be made (including uncertainty quantification) via the posterior predictive distrubtion.

## Quick Start
TODO: add step by step procedure to run a scenario.
