#!/bin/bash

pip install virtualenv 
virtualenv my_env # create a virtual environment named my_env
source my_env/bin/activate # activate my_env
pip install -r requirements.txt
