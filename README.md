## torch2nki



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


HOW TO INSTALL DA NEUUUUUUUUURON 

1. get a ubuntu 
2. make sure apt is installed 
3. Run commands: 

sudo apt install -y python3.8-venv gcc-c++ 

python3.8 -m venv aws_neuron_venv_pytorch 

source aws_neuron_venv_pytorch/bin/activate 
python -m pip install -U pip 

python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

python -m pip install wget 
python -m pip install awscli 

python -m pip install neuronx-cc==2.* torch-neuronx torchvision


Debuggign: 
Verify after you do the source command that "which python" points to your venv" 


