'''
Find out all the answers that meet the following 10 questions at the same time.
1.The answer to this question is:
(a)a; (b)b; (c)c; (d)d; 
2.The answer to the 5th question is:
(a)c; (b)d; (c)a; (d)b; 
3.Which of the following options has a different answer than the other three:
(a)3; (b)6; (c)2; (d)4; 
4.Which of the following options has the same answer:
(a)1,5; (b)2,7; (c)1,9; (d)6,10;
5.Which of the following options has the same answer as this one?:
(a)8; (b)4; (c)9 (d)7; 
6.Which two questions have the same answer as the 8th question?:
(a)2,4; (b)1,6; (c)3,10; (d)5,9; 
7.Which letter is the least in the correct answers to these ten questions:
(a)c; (b)b; (c)a; (d)d; 
8.Which of the following options has an answer that is not adjacent to the answer to the 1st question in alphabetical order:
(a)7; (b)5; (c)2; (d)10; 
9.It is known that "The answer to the 1st question is the same as the 6th" is contrary to "The answer to the X(st/nd/rd/th) question and the 5th". Then X is:
(a)6; (b)10; (c)2; (d)9; 
10.Among the correct answers to these ten questions,the maximum number of occurrences of the abcd letters subtract the minimum number is:
(a)3; (b)2; (c)4; (d)1; 
'''

def count(ans):
    r = {}
    n_abcd = {'a':0,'b':0,'c':0,'d':0}
    for x in ans:
        n_abcd[x] += 1
    r['n'] = n_abcd
    rmin = min(n_abcd.values())
    rmax = max(n_abcd.values())
    r['min'] = rmin
    r['max'] = rmax
    r['min_set'] = [i for i in n_abcd if n_abcd[i]==rmin]
    r['max_set'] = [i for i in n_abcd if n_abcd[i]==rmin]
    return r

def is_true_ans(ans):
    #q1
    if ans[0]=='a' and not(ans[0]=='a'):
        return False
    elif ans[0]=='b' and not(ans[0]=='b'):
        return False
    elif ans[0]=='c' and not(ans[0]=='c'):
        return False
    elif ans[0]=='d' and not(ans[0]=='d'):
        return False
    #q2
    if ans[1]=='a' and not(ans[4]=='c'):
        return False
    elif ans[1]=='b' and not(ans[4]=='d'):
        return False
    elif ans[1]=='c' and not(ans[4]=='a'):
        return False
    elif ans[1]=='d' and not(ans[4]=='b'):
        return False
    #q3
    if ans[2]=='a' and not(ans[2]!=ans[5] and ans[2]!=ans[1] and ans[2]!=ans[3]):
        return False
    elif ans[2]=='b' and not(ans[5]!=ans[2] and ans[5]!=ans[1] and ans[5]!=ans[3]):
        return False
    elif ans[2]=='c' and not(ans[1]!=ans[2] and ans[1]!=ans[5] and ans[1]!=ans[3]):
        return False
    elif ans[2]=='d' and not(ans[3]!=ans[2] and ans[3]!=ans[5] and ans[3]!=ans[1]):
        return False
    #q4
    if ans[3]=='a' and not(ans[0]==ans[4]):
        return False
    elif ans[3]=='b' and not(ans[1]==ans[6]):
        return False
    elif ans[3]=='c' and not(ans[0]==ans[8]):
        return False
    elif ans[3]=='d' and not(ans[5]==ans[9]):
        return False
    #q5
    if ans[4]=='a' and not(ans[4]==ans[7]):
        return False
    elif ans[4]=='b' and not(ans[4]==ans[3]):
        return False
    elif ans[4]=='c' and not(ans[4]==ans[8]):
        return False
    elif ans[4]=='d' and not(ans[4]==ans[6]):
        return False
    #q6
    if ans[5]=='a' and not((ans[7]==ans[1] and ans[7]==ans[3])):
        return False
    elif ans[5]=='b' and not((ans[7]==ans[0] and ans[7]==ans[5])):
        return False
    elif ans[5]=='c' and not((ans[7]==ans[2] and ans[7]==ans[9])):
        return False
    elif ans[5]=='d' and not((ans[7]==ans[4] and ans[7]==ans[8])):
        return False
    #q7
    r = count(ans)
    if ans[6]=='a' and not('c' in r['min_set']):
        return False
    elif ans[6]=='b' and not('b' in r['min_set']):
        return False
    elif ans[6]=='c' and not('a' in r['min_set']):
        return False
    elif ans[6]=='d' and not('d' in r['min_set']):
        return False
    #q8
    if ans[7]=='a' and not(abs(ord(ans[0])-ord(ans[6]))!=1):
        return False
    elif ans[7]=='b' and not(abs(ord(ans[0])-ord(ans[4]))!=1):
        return False
    elif ans[7]=='c' and not(abs(ord(ans[0])-ord(ans[1]))!=1):
        return False
    elif ans[7]=='d' and not(abs(ord(ans[0])-ord(ans[9]))!=1):
        return False
    #q9
    if ans[8]=='a' and not(((ans[0]==ans[5])^(ans[5]==ans[4]))):
        return False
    elif ans[8]=='b' and not(((ans[0]==ans[5])^(ans[9]==ans[4]))):
        return False
    elif ans[8]=='c' and not(((ans[0]==ans[5])^(ans[1]==ans[4]))):
        return False
    elif ans[8]=='d' and not(((ans[0]==ans[5])^(ans[8]==ans[4]))):
        return False
    #q10
    if ans[9]=='a' and not(r['max']-r['min']==3):
        return False
    elif ans[9]=='b' and not(r['max']-r['min']==2):
        return False
    elif ans[9]=='c' and not(r['max']-r['min']==4):
        return False
    elif ans[9]=='d' and not(r['max']-r['min']==1):
        return False

    return True
    

if __name__=='__main__':
    all_ans = [[]]
    ans_char = ['a','b','c','d']
    for i in range(10):
        all_ans = [x+[y] for x in all_ans for y in ans_char]

    ans_true = []
    for ans in all_ans:
        if is_true_ans(ans):
            ans_true.append(ans)
            print(ans)
    #print(ans_true)
