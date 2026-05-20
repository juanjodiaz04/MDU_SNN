import logging
from bclef_DL import ClefUnifiedDL
from spikerplus import NetBuilder, Trainer, VhdlGenerator
from spikerplus.vhdl import write_vhdl


logging.basicConfig(level=logging.INFO)

data_dir = "../Birds_Balanced"
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

    bit_width_config = {
        "weights_bw": 8,
        "neurons_bw": 16,
        "fp_dec": 6
        }

    net_builder = NetBuilder(net_dict)
    snn = net_builder.build()
    trainer = Trainer(snn)

    trainer.train(train_loader,test_loader,n_epochs=20, store = True)

    # VHDL Generation
    # vhdl_generator = VhdlGenerator(snn, bit_width_config)
    # vhdl_snn = vhdl_generator.generate()
    # write_vhdl(vhdl_snn, output_dir="SpikerAudio")

if __name__ == "__main__":
    main()