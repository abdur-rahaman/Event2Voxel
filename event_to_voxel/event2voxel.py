import os
import shutil
import numpy as np
from src.event_to_voxel.event_tensor_utils import events_to_voxel_grid

def convert_from_event_to_voxel(path: str):
    # Paths for input and output data
    data_path = os.path.join(path, 'event0')  # Directory for event data
    output_path = os.path.join(path, 'voxel0_orig')  # Directory for voxel data

    # Create the output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # Initialize timestamp list
    timestamp_list = []

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Handle timestamps file
            if file == 'timestamps.txt':
                shutil.copy(file_path, os.path.join(output_path, file))  # Copy timestamps
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    timestamp_list.append(line.strip().split(' '))  # Collect timestamps

            # Handle event `.npy` files
            elif file.endswith('.npy'):
                events = np.load(file_path)

                # Validate event data shape
                if len(events.shape) != 2 or events.shape[1] != 4:
                    raise ValueError(
                        f"Invalid shape for events file: {file_path}. Expected Nx4, got {events.shape}"
                    )

                # Extract event attributes
                t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

                # Convert timestamps from microseconds to seconds
                t = t / 1e6

                # Create single event array combining t, x, y, and p
                single_np_event = np.vstack((t, x, y, p)).T

                # Convert to voxel grid
                width, height, num_bins = 346, 260, 5
                voxel = events_to_voxel_grid(single_np_event, num_bins, width, height)

                # Save voxel grid
                new_file_name = file.replace('events', 'voxel').replace('.npy', '.npy')
                np.save(os.path.join(output_path, new_file_name), voxel)


def main():
    # Define base directory for event data
    base_path = '/mnt/sub_hdd/MVSEC/dataset/indoor_flying_1'

    # Call the conversion function
    convert_from_event_to_voxel(base_path)


if __name__ == '__main__':
    main()
