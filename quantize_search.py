import logging
import torch

from bclef_DL import ClefUnifiedDL
from spikerplus import NetBuilder, VhdlGenerator
from spikerplus.vhdl import write_vhdl
from optuna_optimizer import Optimizer_Opt


logging.basicConfig(level=logging.INFO)

data_dir = "../Birds_Split"
snn_state_dict = "Trained/trained_state_dict.pt"
batch_size = 64

def main():
    
    data_loader = ClefUnifiedDL(data_dir=data_dir, encoding_type="sf", spiking_thresh=0.2)
    train_loader, test_loader = data_loader.load(batch_size=64)

    # Extract number of timesteps and inputs
    n_cycles = next(iter(train_loader))[0].shape[1]
    n_inputs = next(iter(train_loader))[0].shape[2]

    # Configure the SNN
    net_dict = {
        "n_cycles": n_cycles,
        "n_inputs": n_inputs,
        "layer_0": {
            "neuron_model": "lif",
            "n_neurons": 128,
            "beta": 0.9375,
            "learn_beta": False,
            "threshold": 1.,
            "learn_threshold": False,
            "reset_mechanism": "subtract"
        },
        "layer_1": {
            "neuron_model": "lif",
            "n_neurons": 9,
            "beta": 0.9375,
            "learn_beta": False,
            "threshold": 1.,
            "learn_threshold": False,
            "reset_mechanism": "none"
        }
    }

    # Search ranges for the optimizer (Quantization bit widths)
    optim_config = {

        "weights_bw"	: {
            "min"	: 4,
            "max"	: 10
        },

        "neurons_bw"	: {
            "min"	: 4,
            "max"	: 10
        },

        "fp_dec"	: {
            "min"	: 4,
            "max"	: 6
        }
    }

    net_builder = NetBuilder(net_dict)
    snn = net_builder.build()
    
    # Load pre-trained model weights
    state_dict = torch.load(snn_state_dict)
    snn.load_state_dict(state_dict)

    # Instantiate optimizer
    opt = Optimizer_Opt(snn, net_dict, optim_config)

    # Run grid search over provided quantization ranges
    opt.optimize(test_loader)

    # VHDL Generation
    # optim_config = {
	# "weights_bw"	: 4,
	# "neurons_bw"	: 6,
	# "fp_dec"		: 4
    # }
    # vhdl_generator = VhdlGenerator(snn, optim_config)
    # vhdl_snn = vhdl_generator.generate()
    # write_vhdl(vhdl_snn, output_dir="SpikerAudio")

if __name__ == "__main__":
    main()