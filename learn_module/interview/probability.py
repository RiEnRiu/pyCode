import numpy as np
import random

import sympy as sy

def simulation(times=1000):
    sum = float(0)
    for i in range(times):
        pre = 0
        now = 0
        while((pre==1 and now==1)==False):
            pre = now
            now = random.randint(0,1)
            sum += float(1)
    return sum/times


def limit_val(eps = 1e-6):
    a0=float(0)
    a1=float(0)
    a2=0.25
    sum_n_3 = 0
    ans = 0*a0+1*a1+2*a2
    i=3
    while(i*a2>eps):
        sum_n_3 += a0
        a0 = a1
        a1 = a2
        a2 = (1-sum_n_3)/8
        ans += i * a2
        i+=1
    return ans
        
def cal_real_prob():
    p0 = 0
    p1 = 0
    p2 = 0.25
    #p(n) = (1-sum(p(i), i=[0,n-3]))*0.125)
    #s(n) = 0.125-0.125 * s(n-3) + s(n-1)
    #ss(n) = ss(n-1) - 0.125 * ss(n-3); ss(n) = s(n) - 1
    #x**3 = x**2 - 0.125
    x = sy.Symbol('x')
    [x1,x2,x3] = sy.solve(x**3 - x**2 + 0.125,x)
    #ss(n) = A * x1**n + B * x2**n + C * x3**n
    #[[p(0)-1],[p(0)+p(1)-1],[p(0)+p(1)+p(2)-1]] = [[1,1,1],[x1,x2,x3],[x1*x1,x2*x2,x3*x3]]*[[A],[B],[C]]
    a = sy.Symbol('a')
    b = sy.Symbol('b')
    c = sy.Symbol('c')
    abc_ans = sy.solve([a+b+c-(p0-1),a*x1+b*x2+c*x3-(p0+p1-1),a*x1*x1+b*x2*x2+c*x3*x3-(p0+p1+p2-1)],[a,b,c])
    [A,B,C] = [abc_ans[a],abc_ans[b],abc_ans[c]]
    #p(n) = s(n) - s(n-1) = ss(n) - ss(n-1) = A*(x1-1)*x1**(n-1) + B*(x2-1)*x2**(n-1) + C*(x3-1)*x3**(n-1)
    #n*p(n) = (A*(x1-1)*n)*x1**(n-1) + (B*(x2-1)*n)*x2**(n-1) + (C*(x3-1)*n)*x3**(n-1)
    #[a1,a2,a3] = [A*(x1-1),B*(x2-1),C*(x3-1)]
    #[b1,b2,b3] = [0,0,0]
    #[q1,q2,q3] = [x1,x2,x3]
    #T(n) = sum(n*p(n)) = (t11*n+t12)*x1**n+t13 + (t21*n+t22)*x2**n+t23 + (t31*n+t32)*x3**n+t33
    #[[t11,t12,t13],[t21,t22,t23],[t31,t32,t33]] = \
    #       [[a1/(q1-1),(b1-a1/(q1-1))/(q1-1),-(b1-a1/(q1-1))/(q1-1)],\
    #        [a2/(q2-1),(b2-a2/(q2-1))/(q2-1),-(b2-a2/(q2-1))/(q2-1)],\
    #        [a3/(q3-1),(b3-a3/(q3-1))/(q3-1),-(b3-a3/(q3-1))/(q3-1)]]
    #lim(T(n)) = t13 + t23 + t33
    [a1,a2,a3] = [A*(x1-1),B*(x2-1),C*(x3-1)]
    [b1,b2,b3] = [0,0,0]
    [q1,q2,q3] = [x1,x2,x3]
    [[t11,t12,t13],[t21,t22,t23],[t31,t32,t33]] = \
           [[a1/(q1-1),(b1-a1/(q1-1))/(q1-1),-(b1-a1/(q1-1))/(q1-1)],\
            [a2/(q2-1),(b2-a2/(q2-1))/(q2-1),-(b2-a2/(q2-1))/(q2-1)],\
            [a3/(q3-1),(b3-a3/(q3-1))/(q3-1),-(b3-a3/(q3-1))/(q3-1)]]
    lim_Tn =  t13 + t23 + t33
    return lim_Tn

def use_mean(p,n):
    #Set E(n) is the answer
    #E(n-1) is the answer of the case n-1
    #1. when do E(n-1) times, we may finish in the next step with the probability of p
    #1. means:  do (E(n-1) + 1) times with the probability of p
    #2. when do E(n-1)+1 times, we may get the reverse side of the coin,then we must recount
    #2. means:  do (E(n-1) + 1 + E(n)) times with the probability of (1-p)
    #3. E(0) = 0, do not need any trys then we finish the experment
    #4. from 1.2. , we get E(n) = p*(E(n-1)+1) + (1-p)*(E(n-1)+1+E(n))
    #5. from 4. , we get E(n) = 1/p*E(n-1)+1/p
    #6. from 3.5. , we get E(n) = (1-p**n)/((p**n)*(1-p))
    return (1-p**n)/((p**n)*(1-p))

if __name__=='__main__':
    print(simulation(10000))
    print(limit_val(1e-12))
    print(cal_real_prob())
    print(use_mean(0.5,2))
