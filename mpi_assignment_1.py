from mpi4py import MPI

# Initialize MPI overhead.
communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()


def main():

    # Define whether our rank is even. 
    is_even_rank = rank % 2
    
    # If so, print 'Hello' output. Otherwise, print 'Goodbye' output.
    if is_even_rank:
        print('Hello from process {}'.format(rank))

    else:
        print('Goodbye from process {}'.format(rank))


if __name__ == '__main__':
    main()
