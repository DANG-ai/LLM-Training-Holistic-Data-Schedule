from typing import List, Optional, Dict, Iterator
import numpy as np
import math
import random
import torch
import deepspeed
from megatron import print_rank_0, mpu
from megatron.utils import get_ltor_masks_and_position_ids, reduce_losses
from megatron.model.gpt2_model import cross_entropy

def _get_batch(neox_args, keys, data, datatype):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids

class WeightedRandomSampler(torch.utils.data.Sampler):
    datasets: Dict[str, torch.utils.data.Dataset]
    weights: List[float]

    def __init__(self, datasets, weights, num_samples=None, generator=None):
        self.datasets = datasets
        self.weights = weights
        if num_samples is not None:
            self.num_samples = num_samples
        else:
            self.num_samples = sum(len(dataset) for dataset in datasets.values())
        self.generator = generator
        self.epochs = {dataset_name: 0 for dataset_name in datasets.keys()}


        self._dataset_map = {}
        self._inverse_dataset_map = {}
        self._dataset_offsets = {}
        offset = 0
        for i, (dataset_name, dataset) in enumerate(datasets.items()):
            self._dataset_map[i] = dataset_name
            self._inverse_dataset_map[dataset_name] = i
            self._dataset_offsets[i] = offset
            offset += len(dataset)

        # self._dataset_map = {dataset_name: i for i, dataset_name in enumerate(datasets.keys())}
        # self._dataset_offsets = [dataset]
        self._dataset_options = range(len(datasets.keys()))
        self._reset(generator)
        self._all_done = False

    @property
    def all_done(self) -> bool:
        return self._all_done

    @all_done.setter
    def all_done(self, done):
        self._all_done = done

    @property
    def weights(self) -> List[float]:
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def iterate_epoch(self, dataset_name: str) -> None:
        self.epochs[dataset_name] += 1

    def _reset(self, generator):
        self._samplers = [iter(torch.utils.data.RandomSampler(dataset, generator=generator)) for dataset in self.datasets.values()]

    def _reset_single(self, dataset_name: str, generator):
        self._samplers[self._inverse_dataset_map[dataset_name]] = iter(torch.utils.data.RandomSampler(self.datasets[dataset_name], generator=generator))

    def _infinite_iterator(self):
        while True:

            # check for manual exit
            if self.all_done:
                break

            chosen_dataset = random.choices(self._dataset_options, weights=self._weights, k=1)[0]
            dataset_name = self._dataset_map[chosen_dataset]

            try:
                chosen_sample = next(self._samplers[chosen_dataset])
            except StopIteration:
                self.iterate_epoch(dataset_name)
                generator = torch.Generator()
                generator.manual_seed(self.generator.seed() + self.epochs[dataset_name])
                self._reset_single(dataset_name, self.generator)
                chosen_sample = next(self._samplers[chosen_dataset])
            yield chosen_sample + self._dataset_offsets[chosen_dataset]

    def __iter__(self) -> Iterator[int]:
        return self._infinite_iterator()

    def __len__(self):
        return self.num_samples

class DistributedWeightedRandomSampler(torch.utils.data.distributed.DistributedSampler):
    datasets: Dict[str, torch.utils.data.Dataset]
    weights: List[float]
    def __init__(self, datasets, weights, world_size: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if world_size is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= world_size or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval"
                             f"[0, {world_size - 1}]")
        # self.datasets = datasets
        self.weights = weights
        self.world_size = world_size
        self.rank = rank
        self.epochs = {dataset_name: 0 for dataset_name in datasets.keys()}
        self.drop_last = drop_last
        if self.drop_last:
            print_rank_0("WARNING: drop_last=True is not supported for DistributedWeightedRandomSampler, "
                           "and will be ignored.")
        self.shuffle = shuffle
        self.seed = seed

        # initialize mappings, samples, and offsets
        self.total_samples = 0
        self.local_samples = {}
        self._dataset_map = {}
        self._global_offsets = {}
        self._local_offsets = {}
        global_offset = 0
        dropped_samples = {}
        for i, (dataset_name, dataset) in enumerate(datasets.items()):
            self._dataset_map[i] = dataset_name
            self._global_offsets[dataset_name] = global_offset
            dataset_samples = len(dataset)
            
            if dataset_samples % self.world_size != 0:
                # Split to the nearest number of samples that is evenly divisible across all replicas
                self.local_samples[dataset_name] = math.ceil(
                    (dataset_samples - self.world_size) / self.world_size
                )
            else:
                self.local_samples[dataset_name] = math.ceil(len(dataset) / self.world_size)

            self.total_samples += self.local_samples[dataset_name]
            self._local_offsets[dataset_name] = self.local_samples[dataset_name] * self.rank
            global_offset += dataset_samples
            dropped_samples[dataset_name] = dataset_samples - self.local_samples[dataset_name] * self.world_size

        self.total_local_samples = sum(self.local_samples.values())

        print_rank_0(f"Rank: {self.rank} -- Datasampler World Size: {self.world_size}"
                    f" -- Total samples: {self.total_samples} -- Local samples: {self.local_samples}"
                    f" -- Global offsets: {self._global_offsets} -- Local offsets: {self._local_offsets}"
                    f" -- Dropped samples: {dropped_samples}"
                )

        self._dataset_options = range(len(datasets.keys()))
        self._reset()
        self._all_done = False

    @property
    def all_done(self) -> bool:
        return self._all_done

    @all_done.setter
    def all_done(self, done):
        self._all_done = done

    @property
    def weights(self) -> List[float]:
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def iterate_epoch(self, dataset_name: str) -> None:
        self.epochs[dataset_name] += 1

    def _reset(self):
        # initialize random number generators
        python_rng = np.random.default_rng(self.seed + self.rank)
        random.seed(python_rng.integers(0, 2**32 - 1))

        # Reset all samplers based on current state
        self._samplers = {}
        for i, (dataset_name, num_samples) in enumerate(self.local_samples.items()):
            indices = list(range(num_samples))
            dataset_seed = self.seed + i + self.rank + ((i+1)*self.rank)
            dataset_rng = np.random.default_rng(dataset_seed)
            g = torch.Generator()
            g.manual_seed(int(dataset_rng.integers(0, 2**32 - 1)))
            self._samplers[dataset_name] = iter(torch.utils.data.RandomSampler(indices, generator=g))

    def _reset_single(self, dataset_name: str, g: torch.Generator):
        indices = list(range(self.local_samples[dataset_name]))
        self._samplers[dataset_name] = iter(torch.utils.data.RandomSampler(indices, generator=g))

    def _infinite_iterator(self) -> Iterator[int]:
        while True:

            # check for manual exit
            if self.all_done:
                break

            chosen_dataset = random.choices(self._dataset_options, weights=self._weights, k=1)[0]
            dataset_name = self._dataset_map[chosen_dataset]

            try:
                chosen_sample = next(self._samplers[dataset_name])
            except StopIteration:
                self.iterate_epoch(dataset_name)
                generator = torch.Generator()
                generator.manual_seed(self.seed + self.epochs[dataset_name] + self.rank)
                self._reset_single(dataset_name, generator)
                chosen_sample = next(self._samplers[dataset_name])
            yield chosen_sample + self._global_offsets[dataset_name] + self._local_offsets[dataset_name]
        

    def __iter__(self) -> Iterator[int]:
        return self._infinite_iterator()

    def __len__(self) -> int:
        return self.total_local_samples


class BaseDataSamplingWeight:
    """
    Simple class that stores the weighting scheme for domains within the data.
    This class will maintain stationary weights, but can be subclassed to
    implement dynamic weighting schemes.
    """
    def __init__(self,
        weights: List[float] = None,
    ):
        self.weights = weights

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def __call__(self):
        return self.weights
    
    def update(self, iteration: int, **kwargs):
        pass

    def log(self):
        return {}
    
class DynamicDataSamplingWeight(BaseDataSamplingWeight):
    """
    Dynamic data sampling weighter that updates the weights based on the
    provided update method.
    """
    def __init__(self,
        dataset_names: List[str],
        weights: List[float] = None,
        warmup_steps: Optional[int] = 0,
        update_frequency: Optional[int] = 0,
        update_method: Optional[str] = None,
        internal_updates: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(weights=weights)
        assert(update_frequency > 0)
        self.dataset_names = dataset_names
        self.update_scheduler = UpdateScheduler(
            warmup_steps=warmup_steps,
            update_frequency=update_frequency,
        )
        self.update_method = update_method
        self.weight_updater = get_weight_updater(update_method, dataset_names, weights, **kwargs)
        self.internal_updates = internal_updates
        
    def update(self, iteration: int, **kwargs):
        if self.update_scheduler.requires_update(iteration):
            self.weights = self.weight_updater.update(iteration=iteration, **kwargs)
        elif self.internal_updates:
            self.weight_updater.internal_update(iteration=iteration, **kwargs)

    def group_update(self, iteration: int, **kwargs):
        if self.update_scheduler.requires_update(iteration):
            self.weights = self.weight_updater.group_update(iteration=iteration, **kwargs)
        elif self.internal_updates:
            self.weight_updater.internal_update(iteration=iteration, **kwargs)

    def log(self):
        logging_dict = {}
        for var in self.weight_updater.vars_to_log:
            logging_dict[var] = getattr(self.weight_updater, var)
        return logging_dict

class UpdateScheduler:
    def __init__(self, warmup_steps: int, update_frequency: int):
        self.warmup_steps = warmup_steps
        self.update_frequency = update_frequency

    def requires_update(self, iteration: int) -> bool:
        return iteration > self.warmup_steps and iteration % self.update_frequency == 0

class EMAWeightUpdater:
    """
    Implementation of the weight updater, using Exponential Moving Average (EMA).
    """
    def __init__(
        self,
        dataset_names: List[str],
        weights: List[float],
    ):
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = {name: weight for name, weight in zip(dataset_names, weights)}
        total_weights = np.sum(weights)
        self._probabilities = {name: weight/total_weights for name, weight in zip(dataset_names, weights)}
        self.smoothing_factor = 2
        self.distribution_function = None # can be None or "softmax"
        self.vars_to_log = ["_probabilities", "weights"]

    def update(self, dataset_name: str, reward: float, iteration: int) -> List[float]:
        self.weights[dataset_name] = (self.smoothing_factor / (1 + iteration)) * reward + (1 - self.smoothing_factor / (1 + iteration)) * self.weights[dataset_name]

        if self.distribution_function == "softmax":
            total_weights = np.sum([np.exp(weight) for weight in self.weights.values()])
            self._probabilities = {name: np.exp(weight)/total_weights for name, weight in self.weights.items()}

        else:
            total_weights = np.sum([weight for weight in self.weights.values()])
            self._probabilities = {name: weight/total_weights for name, weight in self.weights.items()}

        return list(self._probabilities.values())

class Exp3WeightUpdater:
    """
    Implementation of the weight updater, using Exponential-weight algorithm for Exploration and Exploitation (Exp3).
    """
    def __init__(
            self,
            dataset_names: List[str],
            weights: List[float],
            ):
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = weights
        self._cumulative_estimated_reward = {name: 0.0 for name in dataset_names}
        total_weights = np.sum(weights)
        self._probabilities = {name: weight/total_weights for name, weight in zip(dataset_names, weights)}
        self.eps = 1/self.num_datasets
        self.prev_eps = None
        self.vars_to_log = ["_probabilities", "_cumulative_estimated_reward"]

    def update(self, dataset_name: str, reward: float, iteration: int) -> List[float]:
        """
        Updates the weights based on the provided reward.
        """

        # print(f"Rank: {torch.distributed.get_rank()} -- dataset_name: {dataset_name} -- reward: {reward}"
        #       f" -- eps {self.eps} -- prev_eps {self.prev_eps}")

        # scale down reward so that cumulative reward doesn't explode
        reward = reward/10

        # update cumulative estimated reward
        self._cumulative_estimated_reward[dataset_name] += reward/self._probabilities[dataset_name]

        # print(f"Rank: {torch.distributed.get_rank()} -- cumulative_estimated_reward {self._cumulative_estimated_reward}")

        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_datasets, math.sqrt(math.log(self.num_datasets)/(self.num_datasets*iteration)))

        # print(f"Rank: {torch.distributed.get_rank()} -- eps {self.eps} -- prev_eps {self.prev_eps}")

        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._cumulative_estimated_reward.values()])
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards

        # print(f"Rank: {torch.distributed.get_rank()} -- total_estimated_rewards {total_estimated_rewards} -- scaling_factor {scaling_factor}")

        # update weights
        for name in self.dataset_names:
            self.weights[self.dataset_map[name]] = math.exp(self._cumulative_estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps

        # print(f"Rank: {torch.distributed.get_rank()} -- cumulative_estimated_reward {self._cumulative_estimated_reward}")

        # update probabilities
        total_weights = sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights

        # print(f"Rank: {torch.distributed.get_rank()} -- reward: {reward} -- eps {self.eps} -- scaling_factor {scaling_factor}"
        #       f" -- cumulative_estimated_reward {self._cumulative_estimated_reward} -- weights {self.weights} -- probabilities {self._probabilities}"
        #       f" -- total_weights {total_weights}"
        #       )

        return list(self._probabilities.values())


    def group_update(self, dataset_names: List[str], rewards: List, iteration: int):
        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_datasets, math.sqrt(math.log(self.num_datasets)/(self.num_datasets*iteration)))

        # update cumulative estimated reward
        for name, reward in zip(dataset_names, rewards):
            # scale down reward so that cumulative reward doesn't explode
            reward = reward/10
            self._cumulative_estimated_reward[name] += reward/self._probabilities[name]
        # print(f"Rank: {torch.distributed.get_rank()} -- cumulative_estimated_reward {self._cumulative_estimated_reward}")

        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._cumulative_estimated_reward.values()])
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards
        # print(f"Rank: {torch.distributed.get_rank()} -- total_estimated_rewards {total_estimated_rewards} -- scaling_factor {scaling_factor}")

        # update weights
        for name in self.dataset_names:
            self.weights[self.dataset_map[name]] = math.exp(self._cumulative_estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps
        # print(f"Rank: {torch.distributed.get_rank()} -- cumulative_estimated_reward {self._cumulative_estimated_reward}")

        # update probabilities
        total_weights = sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights

        # print(f"Rank: {torch.distributed.get_rank()} -- reward: {reward} -- eps {self.eps} -- scaling_factor {scaling_factor}"
        #       f" -- cumulative_estimated_reward {self._cumulative_estimated_reward} -- weights {self.weights} -- probabilities {self._probabilities}"
        #       f" -- total_weights {total_weights}"
        #       )

        return list(self._probabilities.values())
    
class SmoothedMeanWeightUpdater:
    def __init__(
            self,
            dataset_names: List[str],
            weights: List[float],
            smoothing_factor: float = 0.9,
            ):
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = weights
        self._estimated_reward = {name: 0.0 for name in dataset_names}
        total_weights = np.sum(weights)
        self._probabilities = {name: weight/total_weights for name, weight in zip(dataset_names, weights)}
        self.eps = 1/self.num_datasets
        self.prev_eps = None
        self.smoothing_factor = smoothing_factor
        self.vars_to_log = ["_probabilities", "_estimated_reward"]

    def update(self, dataset_name: str, reward: float, iteration: int) -> List[float]:
        """
        Updates the weights based on the provided reward.
        """

        # update cumulative estimated reward
        self._estimated_reward[dataset_name] = self.smoothing_factor*self._estimated_reward[dataset_name] + (1-self.smoothing_factor)*math.exp(reward)

        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_datasets, math.sqrt(math.log(self.num_datasets)/(self.num_datasets*iteration)))

        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._estimated_reward.values()])
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards

        # update weights
        for name in self.dataset_names:
            self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps

        # update probabilities
        total_weights = sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights

        return list(self._probabilities.values())
    
    def group_update(self, dataset_names: List[str], rewards: List, iteration: int):
        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_datasets, math.sqrt(math.log(self.num_datasets)/(self.num_datasets*iteration)))

        # update cumulative estimated reward
        for name, reward in zip(dataset_names, rewards):
            # smoothed mean
            # self._estimated_reward[name] = self.smoothing_factor*self._estimated_reward[name] + (1-self.smoothing_factor)*reward
            # smoothed exponentiated mean
            self._estimated_reward[name] = self.smoothing_factor*self._estimated_reward[name] + (1-self.smoothing_factor)*math.exp(reward)
        # print(f"Rank: {torch.distributed.get_rank()} -- estimated_reward {self._estimated_reward}")

        # calculate normalized scaling factor
        total_estimated_rewards = sum((r*self.prev_eps) for r in self._estimated_reward.values())
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards

        # update weights
        for name in self.dataset_names:
            # self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps
            self.weights[self.dataset_map[name]] = self._estimated_reward[name]*self.prev_eps*scaling_factor + self.eps

        # update probabilities
        total_weights = sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights

        return list(self._probabilities.values())

class NaiveValidationWeightUpdater:
    def __init__(
            self,
            dataset_names: List[str],
            weights: List[float],
            reward_dataloaders: List,
            neox_args,
            ):
        self.neox_args = neox_args
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = weights
        total_weights = np.sum(weights)
        self._probabilities = {name: weight/total_weights for name, weight in zip(dataset_names, weights)}
        self._rewards = {name: 0.0 for name in dataset_names}
        self._cumulative_estimated_reward = {name: 0.0 for name in dataset_names}
        # self._normalized_rewards = {name: 0.0 for name in dataset_names}
        self.reward_dataloaders = reward_dataloaders
        self.reward_data_iterators = {name: iter(dataloader) for name, dataloader in reward_dataloaders.items()}
        self.eps = 1/self.num_datasets
        self.prev_eps = None
        self.vars_to_log = ["_probabilities", "_cumulative_estimated_reward"]

    def update(self, iteration: int, model) -> List[float]:
        model.eval()
        keys = ["text"]
        datatype = torch.int64
        with torch.no_grad():
            for name, iterator in self.reward_data_iterators.items():
                try:
                    batch = next(iterator)
                except StopIteration:
                    # reset iterator
                    self.reward_data_iterators[name] = iter(self.reward_dataloaders[name])
                    batch = next(self.reward_data_iterators[name])
                

                tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
                    neox_args=self.neox_args,
                    keys=keys,
                    data=batch,
                    datatype=datatype,
                )

                outputs = model((tokens, position_ids, attention_mask))
                reward = cross_entropy(
                    outputs, (labels, loss_mask), _fp16=self.neox_args.fp16_lm_cross_entropy
                )
                self._rewards[name] = reduce_losses([reward]).mean().item()

                # When contiguous memory optimizations are enabled, the buffers
                # allocated by the optimizations are deallocated during backward pass
                # in the absence of backward pass the buffers should be reset after each
                # forward pass
                if self.neox_args.deepspeed and self.neox_args.deepspeed_activation_checkpointing:
                    deepspeed.checkpointing.reset()
        
        # update cumulative estimated (normalized) rewards
        max_reward = max(self._rewards.values())
        min_reward = min(self._rewards.values())
        for name in self.dataset_names:
            # self._normalized_rewards[name] = (self._rewards[name] - min_reward)/(max_reward - min_reward)
            self._cumulative_estimated_reward[name] += (self._rewards[name] - min_reward)/(max_reward - min_reward)

        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_datasets, math.sqrt(math.log(self.num_datasets)/(self.num_datasets*iteration)))

        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._cumulative_estimated_reward.values()])
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards

        # update probabilities
        # total_rewards = sum([math.exp(r) for r in self._normalized_rewards.values()])
        # for name in self.dataset_names:
        #     self._probabilities[name] = math.exp(self._normalized_rewards[name])/total_rewards
        # model.train()
        # return list(self._probabilities.values())

        # update weights
        for name in self.dataset_names:
            self.weights[self.dataset_map[name]] = math.exp(self._cumulative_estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps

        # update probabilities
        # ALON: total_weights should be summing up to 1, can possibly skip the following step
        total_weights = np.sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights

        model.train()
        return list(self._probabilities.values())

def get_weight_updater(update_method: str, dataset_names, weights, **kwargs):
    if update_method == "exp3":
        return Exp3WeightUpdater(dataset_names=dataset_names, weights=weights)
    elif update_method == "ema":
        return EMAWeightUpdater(dataset_names=dataset_names, weights=weights)
    elif update_method == "smoothed_mean":
        return SmoothedMeanWeightUpdater(dataset_names=dataset_names, weights=weights,
                                         smoothing_factor=kwargs["smoothing_factor"])
    elif update_method == "naive_validation":
        return NaiveValidationWeightUpdater(dataset_names=dataset_names, weights=weights,
                                            reward_dataloaders=kwargs["reward_dataloaders"],
                                            neox_args=kwargs["neox_args"])
                                                     

def get_data_sampling_weighter(
        dataset_names: List[str],
        weights: List[float],
        warmup_steps: Optional[int] = 0,
        update_frequency: Optional[int] = 0,
        update_method: Optional[str] = None,
        **kwargs):
    """
    Returns a data sampling weighter based on the provided arguments.
    """
    print_rank_0(f"Creating data sampling weighter with:\n\tdataset_names: {dataset_names}\n\tweights: {weights}\n\twarmup_steps: {warmup_steps}\n\tupdate_frequency: {update_frequency}\n\tupdate_method: {update_method}\n\tkwargs: {kwargs}")
    if update_method is None:
        assert(warmup_steps==0 and update_frequency==0), "Must provide update method if warmup steps or update frequency are provided."
        return BaseDataSamplingWeight(weights=weights)
    
    return DynamicDataSamplingWeight(
        dataset_names=dataset_names,
        weights=weights,
        warmup_steps=warmup_steps,
        update_frequency=update_frequency,
        update_method=update_method,
        **kwargs
    )