from pyspark import SparkContext
from operator import add


def main(sc):

    nums = sc.parallelize(range(1, 1001))
    
    roots = nums.map(lambda val: (val ** 0.5))
    average = roots.fold(0, add) / roots.count()

    print("Average: {}".format(average))


if __name__ == '__main__': 
    sc = SparkContext("local", "avg.of.roots")
    main(sc)
