import math
def cheapest(d, r):
    cheapD = math.inf
    cheapTotal = math.inf
    for i in range(len(d)):
        cheapD = min(cheapD, depart[i])
        cheapTotal = min(cheapTotal, r[i] + cheapD)
    return cheapTotal



depart = [2,2,3,5,7]
return_f = [2,6,4,7,3]

print(cheapest(depart, return_f))