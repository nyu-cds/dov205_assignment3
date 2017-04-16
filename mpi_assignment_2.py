from mpi4py import MPI
from sys import exit
import numpy as np


# Initialize MPI overhead.
communicator = MPI.COMM_WORLD
size, rank = communicator.Get_size(), communicator.Get_rank()


def main():

    # Initialize containers for our signal and user input.
    signal = np.ones(1, dtype=int)

    # Only request input for process rank 0.
    if rank == 0:
        user_input = None

        # While our validator evaluates :user_input to anything but an integer, keep prompting for input.
        while type(user_input) is not int:
            user_input = receive_and_validate_input()

        # Apply first multiplication and pass it along.
        signal = np.multiply(signal, user_input)
        communicator.Send(signal, dest=rank+1)

        # ...Receive our transformed :signal instance from the last process and print.
        communicator.Recv(signal, source=size-1)
        print(signal[0])

    # For any rank aside from 0: receive, transform, send.
    else:

        # Receive :signal from previous process and apply multiplication
        communicator.Recv(signal, source=rank-1)
        signal = np.multiply(signal, rank)

        # If we are the last process, send to initial process 0. Otherwise, next process.
        communicator.Send(signal, dest=0 if (rank == size - 1) else rank + 1)


def receive_and_validate_input():
    """Wrapper function for core requirements: ask for input and validate that input.

    :return cast_response: either integer (if user input was valid) or None.
    """

    # Record user input.
    response = receive_input()

    # Attempt to cast input as int. If successful, return integer. Else `None`.
    cast_response = validate(response)
    return cast_response


def receive_input():
    """Prompt user for and record input. Handle exceptional cases.

    :return response: string representation of user's reponse to prompt.
    """

    try:
        response = input('Enter a value between 0 and 100:\n>>> ')
        return response

    except (KeyboardInterrupt, SystemExit, EOFError):
        print('Exiting..')
        exit(1)


def validate(response):
    """Given user input, determine whether it falls within our constraints.

    Note: this function returns a valid integer if and only if our user input
    can be represented as an int and is in the range [0, 100]. Otherwise,
    we return the value `None`.

    :return cast: an integer if :response is valid, otherwise `None`.
    """

    try:
        cast = int(response)
        return cast if (0 <= cast <= 100) else None

    except:
        return None


if __name__ == '__main__':
    main()
