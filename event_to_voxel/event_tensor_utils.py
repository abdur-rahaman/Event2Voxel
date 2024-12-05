import torch
import numpy as np
from src.event_to_voxel.timers import CudaTimer, Timer

def events_to_voxel_grid(events, num_bins, width, height):
    """
    Convert a list of events into a voxel grid representation.

    :param events: A [N x 4] NumPy array where each row is [timestamp, x, y, polarity].
    :param num_bins: Number of temporal bins for the voxel grid.
    :param width: Width of the voxel grid.
    :param height: Height of the voxel grid.
    """
    assert events.shape[1] == 4
    assert num_bins > 0 and width > 0 and height > 0

    voxel_grid = np.zeros((num_bins, height, width), dtype=np.float32).ravel()

    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    # Normalize timestamps
    events[:, 0] = np.clip((num_bins - 1) * (events[:, 0] - first_stamp) / deltaT, 0, num_bins - 1)
    ts, xs, ys, pols = events[:, 0], events[:, 1].astype(int), events[:, 2].astype(int), events[:, 3]
    pols[pols == 0] = -1  # Convert polarity 0 to -1

    tis = np.floor(ts).astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = (
        (tis >= 0) & (tis < num_bins) &
        (xs >= 0) & (xs < width) &
        (ys >= 0) & (ys < height)
    )
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (
        (tis + 1 >= 0) & (tis + 1 < num_bins) &
        (xs >= 0) & (xs < width) &
        (ys >= 0) & (ys < height)
    )
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = voxel_grid.reshape((num_bins, height, width))
    return voxel_grid
