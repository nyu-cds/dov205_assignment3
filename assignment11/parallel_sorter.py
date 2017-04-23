"""
    Danny Vilela

    This program is a parallel sort implementation that utilizes
    a divide-and-conquer approach to the sorting algorithm. We use
    mergesort's merge phase to lead our re-combination of data.

    Run from the terminal as such:

        $ mpiexec -n NUMBER_OF_PROCESSES python parallel_sorter.py [DATA_SIZE]

    replace NUMBER_OF_PROCESSES with some integer value denoting the number
    of processes you would like to distribute work across. This program also
    takes an optional DATA_SIZE argument to specify the literal size of the random
    data generated that is to be sorted. If not provided, the program assumes
    a default data size of 10,000.
"""

from mpi4py import MPI
from sys import exit, argv
from numpy.random import randint
from numpy import array, array_split, sort

# Initialize MPI overhead.
communicator = MPI.COMM_WORLD
size, rank = communicator.Get_size(), communicator.Get_rank()


def main():

    # Only create original data if process ID is 0.
    if rank == 0:

        # Extract user-input data size or default to 10,000.
        data_size = get_opt_input(argv)
        
        # Generate 10,000 integers in the range [0, 10,000).
        data = generate_data(size=data_size)

        # Print our initial list to the user.
        print('Unsorted array: {}'.format(list(data)))

        # Split into :partitions partitions. We use :array_split() instead of :split()
        # because the former allows non-equal partition sizes.
        partitions = array_split(data, size)

    else:
        partitions = None

    # Scatter partitions across our processes.
    scattered_partitions = communicator.scatter(partitions, root=0)

    # Locally sort our own partition of the data.
    sorted_partitions = sort(scattered_partitions)

    # Gather the results from all processes into their results container (in process 0).
    container = communicator.gather(sorted_partitions, root=0)

    # If we are in process 0 -- hence, :container is a valid value.
    if container is not None:

        # Initialize final result container.
        merged = array([])

        # Merge our existing container with each incoming, sorted partition.
        for partition in container:
            merged = merge(merged, partition)
        
        # Print our results to the user.
        print('Sorted array: {}'.format(merged))


def generate_data(low=0, high=10000, size=10000):
    """Generate our dataset of :size elements in the range [:low, :high).

    :param low: lower bound of integer values for our data.
    :param high: upper bound of integer values for our data.
    :param size: number of elements to be included in data.
    """

    # Validate our data generation parameters are valid integers.
    try:
        low  = int(low)
        high = int(high)
        size = int(size)

    except (TypeError, ValueError):
        print(
            "Unable to cast data generation parameters (low={}, high={}, size={}) as integers.".format(low, high, size),
            "Setting to default low=0, high=100, size=100",
            sep=" "
        )
        return randint(0, 100, 100)
    
    # Handle case where our range is invalid
    if low >= high:
        low, high = high, low

    if size < 0:
        raise ValueError("Invalid size parameter {} must be greater than 0".format(size))

    return randint(low, high, size)


def merge(left, right):
    """Merge two sorted lists. Second phase of mergesort algorithm.

    Given two sorted lists (here, of numeric values), merge them
    into one, contiguous list. This is just the second phase of the
    merge sort algorithm, so we also inherit its stability.

    :param left: sorted list of numeric values.
    :param right: sorted list of numeric values.
    :return merged: sorted union of :left and :right.
    """
    
    if left is None or right is None:
        raise ValueError("Error: Parameters cannot be NoneType.")

    # Initialize each param list's length and starting index.
    left_length, left_index = len(left), 0
    right_length, right_index = len(right), 0

    # Initialize merged list to list of zeros of length (:left + :right).
    merged, merged_index = [0] * (left_length + right_length), 0

    # Standard merge phase: while we haven't finished processing :left
    # and haven't finished processing :right.
    while left_index < left_length and right_index < right_length:

        # If our current value at :left is less than :right, the next location
        # in :merged is the current value at :left. Update :left and :merged
        # indexes.
        if left[left_index] < right[right_index]:
            merged[merged_index] = left[left_index]
            left_index +=1
            merged_index += 1

        # Likewise for :right: update :merged at index :merged_index,
        # then increment :right_index and :merged_index.
        elif left[left_index] >= right[right_index]:
            merged[merged_index] = right[right_index]
            right_index += 1
            merged_index += 1

    # Finally, once one of the lists has been processed,
    # collect the rest from the remaining list. Note: the order
    # of these auxiliary loops does not matter -- one will always
    # be empty and not executed.
    while left_index < left_length:
        merged[merged_index] = left[left_index]
        left_index +=1
        merged_index += 1

    while right_index < right_length:
        merged[merged_index] = right[right_index]
        right_index += 1
        merged_index += 1

    return merged


def get_opt_input(args):
    """If user provides data array size, verify its validity.

    :param args: list of user-provided command-line arguments (sys.argv).
    :return cast: casted, verified value for an array's size.
    """

    # If no value provided, use default length of 10,000.
    if len(args) < 2:
        return 10000

    else:

        # Isolate size argument.
        size_arg = args[1]

        try:

            # If our integer cast of :size_arg is valid return that as array length.
            cast = int(size_arg)
            return cast

        # Inform user if we're unable to extract an integer from their input.
        except (ValueError, TypeError):
            print("Data size must be an integer -- `{}` is invalid. Returning default size of 10000.".format(size_arg))
            return 10000


if __name__ == '__main__':
    main()
