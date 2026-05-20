import copy
import logging
import json
import torch
from tabulate import tabulate
import numpy as np
import optuna

from spikerplus.net_builder import SNN, NetBuilder
from spikerplus.trainer import Trainer


class Quantizer:

	def fixed_point(self, value, fp_dec, bitwidth):

		quant = value * 2**fp_dec

		return self.saturated_int(quant, bitwidth)

	def saturated_int(self, value, bitwidth):
		return self.saturate(self.to_int(value), bitwidth)

	def saturate(self, value, bitwidth):

		if type(value).__module__ == np.__name__ or \
		type(value).__module__ == torch.__name__:

			value[value > 2**(bitwidth-1)-1] = \
				2**(bitwidth-1)-1
			value[value < -2**(bitwidth-1)] = \
				-2**(bitwidth-1)

			return value.float()

		else:

			if value > 2**(bitwidth-1)-1:
				value = 2**(bitwidth-1)-1

			elif value < -2**(bitwidth-1):
				value = -2**(bitwidth-1)

			return float(value)

	def to_int(self, value):

		if type(value).__module__ == np.__name__:
			quant = value.astype(int).astype(float)

		elif type(value).__module__ == torch.__name__:
			quant = value.type(torch.int64).float()

		else:
			quant = float(int(value))

		return quant


class QuantSNN(SNN):

	def __init__(self, net_dict, neurons_bw):

		super().__init__(net_dict)

		self.neurons_bw = neurons_bw

		self.quantizer = Quantizer()


	def forward(self, input_spikes):

		self.reset()

		cur = {}

		if input_spikes.shape[0] != self.n_cycles:
			logging.warning("Input data have a time dimension different from "\
					"the network's number of steps. It's ok at this level, "\
					"but remember to use a suitable number of steps in the "\
					"vhdl generator")

		for step in range(input_spikes.shape[0]):

			first = True

			for layer in self.layers:

				idx = str(self.extract_index(layer))

				if "fc" in layer:

					if first:
						cur[layer] = self.layers[layer](input_spikes[step])
						first = False

					else:
						cur[layer] = self.layers[layer](self.spk[prev_layer])

				elif layer == "if" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[prev_layer], self.mem[layer])

				elif layer == "lif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[prev_layer], self.mem[layer])

				elif layer == "syn" + idx:
					self.spk[layer], self.syn[layer], self.mem[layer] = \
							self.layers[layer](cur[prev_layer], self.syn[layer],
							self.mem[layer])

				elif layer == "rif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[prev_layer], self.spk[layer], self.mem[layer])

				elif layer == "rlif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[prev_layer], self.spk[layer], self.mem[layer])

				elif layer == "rsyn" + idx:
					self.spk[layer], self.syn[layer], self.mem[layer] = \
							self.layers[layer](cur[prev_layer], self.spk[layer],
							self.syn[layer], self.mem[layer])

				prev_layer = layer

				self.quantize(layer)

				self.record(layer)

		self.stack_rec()

	def quantize(self, layer):

		if not "fc" in layer:
			self.mem[layer] = self.quantizer.saturated_int(
					self.mem[layer], self.neurons_bw)

			if "syn" in layer:
				self.syn[layer] = self.quantizer.saturated_int(
					self.syn[layer], self.neurons_bw)


class Optimizer_Opt(Trainer, NetBuilder):

    def __init__(self, net, net_dict, optim_config, readout_type="mem"):

        Trainer.__init__(self, net, readout_type)
        NetBuilder.__init__(self, net_dict)

        self.default_config = {
            "weights_bw"    : {"min" : 4, "max" : 8},
            "neurons_bw"    : {"min" : 4, "max" : 10},
            "fp_dec"        : {"min" : 2, "max" : 3}
        }

        self.allowed_keys = self.default_config.keys()
        self.quantizer = Quantizer()
        self.state_dict = net.state_dict()
        self.net_dict = self.parse_config(net_dict)
        self.optim_config = self.parse_opt_config(optim_config)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def parse_opt_config(self, optim_config):
        optim_dict = {}
        for key in optim_config:
            if key in self.allowed_keys:
                min_value = optim_config[key].get("min", self.default_config[key]["min"])
                max_value = optim_config[key].get("max", self.default_config[key]["max"])

                if not isinstance(min_value, int) or not isinstance(max_value, int):
                    raise ValueError("Range specifiers must be integers\n")

                optim_dict[key] = [i for i in range(min_value, max_value + 1)]

        log_message = "Optimizer configured: \n" + json.dumps(optim_dict, indent=4)
        logging.info(log_message)
        return optim_dict


    def optimize(self, dataloader, n_trials=30):

        study = optuna.create_study(
            directions=["maximize", "minimize", "minimize", "minimize"]
        )

        def objective(trial):
            fp_dec     = trial.suggest_int("fp_dec",      min(self.optim_config["fp_dec"]),      max(self.optim_config["fp_dec"]))
            w_bw       = trial.suggest_int("weights_bw",  min(self.optim_config["weights_bw"]),  max(self.optim_config["weights_bw"]))
            neuron_bw  = trial.suggest_int("neurons_bw",  min(self.optim_config["neurons_bw"]),  max(self.optim_config["neurons_bw"]))

            self.build_quant_snn(w_bw, neuron_bw, fp_dec)
            loss, acc = self.evaluate(dataloader)

            trial.set_user_attr("loss", loss)

            # Return all four objectives in the same order as `directions`.
            return acc, fp_dec, w_bw, neuron_bw

        study.optimize(objective, n_trials=n_trials)

        # Print all Pareto-optimal configurations.
        print(f"\nFound {len(study.best_trials)} optimal trade-off models on the Pareto Front:")
        for t in study.best_trials:
            print(
                f"Trial {t.number}: "
                f"Acc={t.values[0]*100:.2f}%, "
                f"FP={t.values[1]}b, "
                f"Weight={t.values[2]}b, "
                f"Neuron={t.values[3]}b"
            )

        headers = ["Trial", "FP Dec", "Neurons BW", "Weights BW", "Loss", "Accuracy"]

        # Build the results table from the completed trials.
        table = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                
                table.append([
                    str(t.number),
                    str(t.params["fp_dec"]),
                    str(t.params["neurons_bw"]),
                    str(t.params["weights_bw"]),
                    "{:.4f}".format(t.user_attrs.get("loss", 0.0)),
                    "{:.2f}%".format(t.values[0] * 100),
                ])

        table_str = "\n" + tabulate(table, headers=headers, tablefmt="grid")
        logging.info(table_str)

        # For a multi-objective study, pick the best trial from the
        # Pareto front by highest accuracy (values[0]) instead of using the
        # single-objective-only `study.best_trial` / `study.best_value`.
        best_trial = max(study.best_trials, key=lambda t: t.values[0])

        best_msg  = f"\n Best Trial Found: Trial {best_trial.number}\n"
        best_msg += f"Accuracy: {best_trial.values[0]*100:.2f}%\n"
        best_msg += f"Params: {json.dumps(best_trial.params, indent=4)}\n"
        logging.info(best_msg)

        return best_trial.params


    def build_quant_snn(self, weights_bw, neurons_bw, fp_dec):

        self.net = QuantSNN(self.net_dict, neurons_bw)

        # Deep copy of the original state dict to avoid modifying it during quantization
        quant_state_dict = copy.deepcopy(self.state_dict)

        for key in quant_state_dict.keys():
            if "weight" in key:
                quant_state_dict[key] = self.quantizer.fixed_point(
                        quant_state_dict[key], fp_dec, weights_bw)
            elif "threshold" in key:
                quant_state_dict[key] = self.quantizer.fixed_point(
                        quant_state_dict[key], fp_dec, neurons_bw)

        self.net.load_state_dict(quant_state_dict)
        self.net.to(self.device)

        #log_message  = "Network ready:\n"
        #log_message += "Fixed-point decimals: " + str(fp_dec) + "\n"
        #log_message += "Neurons bitwidth: "     + str(neurons_bw) + "\n"
        #log_message += "Weights bitwidth: "     + str(weights_bw) + "\n"
        #logging.info(log_message)