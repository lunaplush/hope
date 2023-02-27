import numpy as np
def almost_double_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        prod = 1
        for i in range(1, n+1, 2):
            prod = prod*i
    return prod


items = [('one', 'two'), ('three', 'four'), ('five', 'six'), ('string', 'a')]

sorted_items = sorted(items, key=lambda x: x[1][-1])

x = [1, 2, 3, 4, 5]
x[5::-2] = [-1, -3, -5]


def cumsum_and_erase(A, erase=1):
    B = []
    sum = 0
    for i in A:
        sum += i
        if sum!=erase:
            B.append(sum)
    return B

def process(sentences):
    result = []
    for s in sentences:
        answ = []
        for word in s.split():
            if word.isalpha():
                answ.append(word)
        result.append(" ".join(answ))
    return result



class Neuron:
    def __init__(self, w, f=lambda x: x):
        self.w = w
        self.func = f
        self.last_x = None

    def forward(self, x):
        self.last_x = x
        sum1 = 0
        for i in range(0, len(x)):
             sum1 += self.w[i]*x[i]
        return self.func(sum1)

    def backlog(self):
        return self.last_x

def func1(x):
    return x*x
a = Neuron((2, 3), func1 )

