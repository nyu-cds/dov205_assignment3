from pyspark import SparkContext
from operator import mul


def main(sc):

    nums = sc.parallelize(range(1, 1001))
    product = nums.fold(1, mul)
    print(product)


if __name__ == '__main__': 
    sc = SparkContext("local", "product")
    main(sc)
