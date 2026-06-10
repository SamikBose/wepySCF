# Standard Library
import logging
from glob import glob

logger = logging.getLogger(__name__)
# Standard Library
import os
import os.path as osp
import pickle

# First Party Library
from wepy.reporter.reporter import Reporter


class WalkerPklReporter(Reporter):
    def __init__(
        self,
        save_dir="./",
        freq: int = 100,
        num_backups: int = 2,
        start_cycle: int | None = None,
    ) -> None:
        # the directory in which to save the pickles
        self.save_dir = save_dir
        # the frequency of cycles to backup the walkers as a pickle
        self.backup_freq = freq
        # the number of most recent walker pickles to keep, this will remove the rest
        self.num_backups = num_backups
        # start cycle index of the run that is being continued (corresponds to the pkl number)
        self.start_cycle_idx = max(0, start_cycle - 1) if start_cycle is not None else None

    def init(self, *args, **kwargs) -> None:
        # make sure the save_dir exists
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def report(self, cycle_idx: int | None = None, new_walkers=None, **kwargs) -> None:
        if cycle_idx is None:
            raise ValueError("WalkerPklReporter requires cycle_idx.")

        # use the correct starting cycle
        if self.start_cycle_idx is not None:
            cycle_idx += self.start_cycle_idx

        # total number of cycles completed
        n_cycles = cycle_idx + 1

        # if the cycle is on the frequency backup walkers to a pickle
        if n_cycles % self.backup_freq == 0:
            pkl_name = f"walkers_cycle_{cycle_idx}.pkl"
            pkl_path = osp.join(self.save_dir, pkl_name)

            with open(pkl_path, "wb") as wf:
                pickle.dump(new_walkers, wf)

            # remove old pickles if we have more than the num_backups
            if (cycle_idx // self.backup_freq) >= self.num_backups:
                old_idx = cycle_idx - self.num_backups * self.backup_freq
                old_pkl_fname = f"walkers_cycle_{old_idx}.pkl"
                old_pkl_path = osp.join(self.save_dir, old_pkl_fname)

                # prevent overwritting last run's final pickle
                if osp.exists(old_pkl_path) and old_idx != self.start_cycle_idx:
                    os.remove(old_pkl_path)
