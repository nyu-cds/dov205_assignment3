from pyspark import SparkContext
from operator import mul


def main(sc):

    # Generate RDD of values [1, 2, ..., 1000].
    nums = sc.parallelize(range(1, 1001))

    # Using 1 as default value, fold partitions using multiplication.
    product = nums.fold(1, mul)

    # Print, effectively, math.factorial(1000).
    print(product)


if __name__ == '__main__': 
    sc = SparkContext("local", "product")
    main(sc)
