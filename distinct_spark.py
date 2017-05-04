from pyspark import SparkContext
from operator import add
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))


def main(sc):

    text = sc.textFile('pg2701.txt')
    
    count = text.flatMap(splitter) \
                .distinct() \
                .count()
    
    print(count)


if __name__ == '__main__': 
    sc = SparkContext("local", "wordcount")
    main(sc)
