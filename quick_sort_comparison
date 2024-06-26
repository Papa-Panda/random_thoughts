# https://www.zhihu.com/question/503261465/answer/2254675541
import random
import timeit
import time
from functools import wraps
scope = 10**3
data = []
for k in range(scope):
    data.append(random.randint(0, scope*3))

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args,  **kwargs)
        t1 = time.time()
        # print("Total time running %s: %s seconds" %(function, __name__, str(t1-t0)))
        print(__name__,str(t1-t0))
        return result
    return function_timer

@fn_timer
def Quicksort(lis):
    Quick(lis, 0 , len(lis)-1)

@fn_timer
def buildin_sort(lis):
    lis.sort()

def Quick(lis, first, last):
    if first<=last:
        split=Subsort(lis,first,last)
        Quick(lis,first, split-1)
        Quick(lis,split+1,last)
def Subsort(lis, first, last):
    left = first + 1
    right = last
    base = lis[first]
    down = False
    while not down:
        while left <= right and lis[left]<=base:
            left = left + 1
        while left <= right and lis[right]>=base:
            right = right -1
        if left > right:
            down = True
            base, lis[right] = lis[right], base
        else:
            lis[left], lis[right] = lis[right], lis[left]
        lis[first] = base
    return right
