# introduction-data-science
This repository contains content from introduction to data science and ai module as part of WS23/24.

## How to use the project

Install the package with **python -m pip -e introduction-data-science** after you have cloned the repository.
This will install the idstools package which contains the relevant modules to fullfill the assignment.

You can either use the modules of the package standalone or go through **<config_name>.yaml** to configure the project.
This config file will be used by the wrapper function to execute the configured steps.

For command line execution you can use **idstools --help** to see your options.

## Build the documentation 

To build the documentation with sphinx execute **make html** in the docs directory.