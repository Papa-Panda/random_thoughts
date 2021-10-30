# Interested in why dictionary's accessing data has O(1) time complexity
# focus on example of python dict
# Python itself provides the hash implementation for str and tuple types. A quick look at the source should reveal the exact algorithm for those.
# refer to https://stackoverflow.com/questions/8997894/what-hash-algorithm-does-pythons-dictionary-mapping-use


def hash(tuple):
    mult = 1000003
    x = 0x345678
    for index, item in enumerate(tuple):
        x = ((x ^ hash(item)) * mult) & (1<<32)
        mult += (82520 + (len(tuple)-index)*2)
    return x + 97531
    
    
def hash(string):
    x = string[0] << 7
    for chr in string[1:]:
        x = ((1000003 * x) ^ chr) & (1<<32)
    return x

