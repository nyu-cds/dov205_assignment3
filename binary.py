from itertools import product, filterfalse


def zbits(n: int, k: int) -> set():
    """Generate :n-length binary sequences with :k zero-bits.

    :param n: the length of our desired output sequences.
    :param k: the number of zero-bits allowed in our output.
    :return as_strings: set of binary sequences as strings that fulfill :n and :k constraints.
    """
    
    # TODO can be optimized? No need to generate ALL pairs.
    # Generate all :n-length pairs, choosing from either 0 or 1.
    all_pairs = product([0, 1], repeat = n)

    # Calculate the difference beteween :n and :k to avoid re-computation.
    difference = n - k

    # Filter our initial space by removing products where k + number of zeros != n.
    filtered = filterfalse(lambda outcome: sum(outcome) != difference, all_pairs)

    # Join our valid binary sequence sets into their joined string representation.
    as_strings = {"".join(str(bit) for bit in outcome) for outcome in filtered}

    # Output: {"101...", "011...", ...}
    return as_strings


if __name__ == '__main__':
   
    # If run from terminal as `$ python binary.py`, run provided tests
    assert zbits(4, 3) == {'0100', '0001', '0010', '1000'}
    assert zbits(4, 1) == {'0111', '1011', '1101', '1110'}
    assert zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}
