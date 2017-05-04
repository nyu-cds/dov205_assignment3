from pyspark import SparkContext
from operator import add


def main(sc):

    # Generate RDD of values [1, 2, ..., 1000].
    nums = sc.parallelize(range(1, 1001))
    
    # Map all values to their square roots.
    roots = nums.map(lambda val: val ** 0.5)

    # Accumulate sum of roots, divide by size of :roots RDD.
    average = roots.fold(0, add) / roots.count()

    # Output average of our :roots RDD.
    print("Average: {}".format(average))


if __name__ == '__main__': 
    sc = SparkContext("local", "avg.of.roots")
    main(sc)
