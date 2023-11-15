import collections
import itertools
import time

from mhcflurry import Class1NeuralNetwork
from mhcflurry.common import configure_tensorflow
from mhcflurry.custom_loss import get_loss

from mhcpred.data_dependent_weights_initialization import lsuv_init


class Class1BinaryNeuralNetwork(Class1NeuralNetwork):

    def __init__(self, **hyperparameters) -> None:
        super(Class1BinaryNeuralNetwork, self).__init__(**hyperparameters)

    @staticmethod
    def data_dependent_weights_initialization(network, x_dict=None, method="lsuv", verbose=1):
        """
        Data dependent weights initialization.

        Parameters
        ----------
        network : keras.Model
        x_dict : dict of string -> numpy.ndarray
            Training data as would be passed keras.Model.fit().
        method : string
            Initialization method. Currently only "lsuv" is supported.
        verbose : int
            Status updates printed to stdout if verbose > 0
        """
        if verbose:
            print("Performing data-dependent init: ", method)
        if method == "lsuv":
            assert x_dict is not None, "Data required for LSUV init"
            lsuv_init(network, x_dict, verbose=verbose > 0)
        else:
            raise RuntimeError("Unsupported init method: ", method)

    def fit_generator(
            self,
            generator,
            validation_peptide_encoding,
            validation_affinities,
            validation_allele_encoding=None,
            validation_inequalities=None,
            validation_output_indices=None,
            steps_per_epoch=10,
            epochs=1000,
            min_epochs=0,
            patience=10,
            min_delta=0.0,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0):
        """
        Fit using a generator. Does not support many of the features of fit(),
        such as random negative peptides.

        Fitting proceeds until early stopping is hit, using the peptides,
        affinities, etc. given by the parameters starting with "validation_".

        This is used for pre-training pan-allele models using data synthesized
        by the allele-specific models.

        Parameters
        ----------
        generator : generator yielding (alleles, peptides, affinities) tuples
            where alleles and peptides are lists of strings, and affinities
            is list of floats.
        validation_peptide_encoding : EncodableSequences
        validation_affinities : list of float
        validation_allele_encoding : AlleleEncoding
        validation_inequalities : list of string
        validation_output_indices : list of int
        steps_per_epoch : int
        epochs : int
        min_epochs : int
        patience : int
        min_delta : float
        verbose : int
        progress_callback : thunk
        progress_preamble : string
        progress_print_interval : float
        """
        configure_tensorflow()
        from tensorflow.keras import backend as K

        fit_info = collections.defaultdict(list)

        loss = get_loss(self.hyperparameters["loss"])

        (
            validation_allele_input,
            allele_representations,
        ) = self.allele_encoding_to_network_input(validation_allele_encoding)

        if self.network() is None:
            self._network = self.make_network(
                allele_representations=allele_representations,
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters)
            )
            if verbose > 0:
                self.network().summary()
        network = self.network()

        network.compile(loss=loss.loss, optimizer=self.hyperparameters["optimizer"])
        network.make_predict_function()
        self.set_allele_representations(allele_representations)

        if self.hyperparameters["learning_rate"] is not None:
            K.set_value(
                self.network().optimizer.lr, self.hyperparameters["learning_rate"]
            )
        fit_info["learning_rate"] = float(K.get_value(self.network().optimizer.lr))

        validation_x_dict = {
            "peptide": self.peptides_to_network_input(validation_peptide_encoding),
            "allele": validation_allele_input,
        }
        encode_y_kwargs = {}
        if validation_inequalities is not None:
            encode_y_kwargs["inequalities"] = validation_inequalities
        if validation_output_indices is not None:
            encode_y_kwargs["output_indices"] = validation_output_indices

        output = loss.encode_y(validation_affinities, **encode_y_kwargs)

        validation_y_dict = {
            "output": output,
        }

        mutable_generator_state = {
            "yielded_values": 0  # total number of data points yielded
        }

        def wrapped_generator():
            for alleles, peptides, affinities in generator:
                (allele_encoding_input, _) = self.allele_encoding_to_network_input(
                    alleles
                )
                x_dict = {
                    "peptide": self.peptides_to_network_input(peptides),
                    "allele": allele_encoding_input.values.reshape(-1, 1),
                }
                y_dict = {"output": affinities.reshape(-1, 1)}
                yield (x_dict, y_dict)
                mutable_generator_state["yielded_values"] += len(affinities)

        start = time.time()

        iterator = wrapped_generator()

        # Initialization required if a data_dependent_initialization_method
        # is set and this is our first time fitting (i.e. fit_info is empty).
        data_dependent_init = self.hyperparameters[
            "data_dependent_initialization_method"
        ]
        if data_dependent_init and not self.fit_info:
            first_chunk = next(iterator)
            self.data_dependent_weights_initialization(
                network,
                first_chunk[0],  # x_dict
                method=data_dependent_init,
                verbose=verbose,
            )
            iterator = itertools.chain([first_chunk], iterator)

        min_val_loss_iteration = None
        min_val_loss = None
        last_progress_print = 0
        epoch = 1
        while True:
            epoch_start_time = time.time()
            fit_history = network.fit(
                iterator,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=epoch - 1,
                epochs=epoch,
                use_multiprocessing=False,
                workers=0,
                validation_data=(validation_x_dict, validation_y_dict),
                verbose=verbose,
            )
            epoch_time = time.time() - epoch_start_time
            for key, value in fit_history.history.items():
                fit_info[key].extend(value)
            val_loss = fit_info["val_loss"][-1]

            if min_val_loss is None or val_loss < min_val_loss - min_delta:
                min_val_loss = val_loss
                min_val_loss_iteration = epoch

            patience_epoch_threshold = min(
                epochs, max(min_val_loss_iteration + patience, min_epochs)
            )

            progress_message = (
                    "epoch %3d/%3d [%0.2f sec.]: loss=%g val_loss=%g. Min val "
                    "loss %g at epoch %s. Cum. points: %d. Stop at epoch %d."
                    % (
                        epoch,
                        epochs,
                        epoch_time,
                        fit_info["loss"][-1],
                        val_loss,
                        min_val_loss,
                        min_val_loss_iteration,
                        mutable_generator_state["yielded_values"],
                        patience_epoch_threshold,
                    )
            ).strip()

            # Print progress no more often than once every few seconds.
            if progress_print_interval is not None and (
                    time.time() - last_progress_print > progress_print_interval
            ):
                print(progress_preamble, progress_message)
                last_progress_print = time.time()

            if progress_callback:
                progress_callback()

            if epoch >= patience_epoch_threshold:
                if progress_print_interval is not None:
                    print(progress_preamble, "STOPPING", progress_message)
                    break
            epoch += 1

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = mutable_generator_state["yielded_values"]
        self.fit_info.append(dict(fit_info))