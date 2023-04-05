import numpy as np

# c_n = list()
# for i in range(10):
#     c_n.append(list())

# print(len(c_n))

def flatten(lst):
    result = []
    for item in lst:
        if type(item) == list:
            result = result + flatten(item)
        else:
            result = result + [item]
    
    return result

a = list()
b = list()
c = list()

a.append(1)
a.append(2)

b.append(3)
b.append(4)

c.append(a)
c.append(b)

print(a)
print(b)
print(c)

c[1].append(5)

print(a)
print(b)
print(c)



print(flatten(c))