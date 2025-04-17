import torch
from torch.utils.data import Dataset

class ShiftedTrajectoryData(Dataset):
    '''
    Class for dataset - turns 2D timeseries input into 3D array of shifted
    trajectories aligning with structure of loss functions
    '''

    def __init__(self, data, num_shifts, len_time):
        '''
        Reshapes data to match expected shape in the DeepKoopman model

        Parameters
        ----------
            data: torch.tensor 
                data (num_examples x num_variables)

            num_shifts: int
                number of time shifts used in loss evaluation. (max is 
                len_time - 1.)

            len_time: int
                length of time dimension for a single trajectory
        '''
        if data.ndim == 1:
            data = data.unsqueeze(1)

        self.num_shifts = num_shifts
        self.len_time = len_time
        self.n_states = data.shape[1]

        self.num_traj = data.shape[0] // len_time
        self.new_len_time = len_time - num_shifts
        self.total_samples = self.num_traj * self.new_len_time

        self.data_tensor = torch.zeros(num_shifts+1,
                                       self.total_samples,
                                       self.n_states)
        
        for j in range(num_shifts+1):
            for traj in range(self.num_traj):
                start = traj * len_time + j
                end = start + self.new_len_time
                idx = traj * self.new_len_time
                self.data_tensor[j, idx:idx+self.new_len_time] = data[start:end]

    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        return [self.data_tensor[j, idx] for j in range(self.num_shifts+1)]
