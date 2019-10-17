'''
Find out all the answers that meet the following 10 questions at the same time.
1.The answer to this question is:
(a)a; (b)b; (c)c; (d)d; 

1. Which question is the first question with answer (b)?:
(a)2; (b)3; (c)4; (d)5; (e)6
2. Which is the only two contiguous questions with the same answer?:
(a)2，3; (b)3，4; (c)4，5; (d)5，6; (e)6，7
3。本问题答案和哪一个问题的答案相同？
(a)1; (b)2; (c)4; (d)7; (e)6
4。答案是a的问题的个数是：
(a)0; (b)1; (c)2; (d)3; (e)4
5。本问题答案和哪一个问题的答案相同？
(a)10; (b)9; (c)8; (d)7; (e)6
6。答案是a的问题的个数和答案是什么的问题的个数相同？
a)b; (b)c; (c)d; (d)e; (e)以上都不是
7。按照字母顺序，本问题的答案和下一个问题的答案相差几个字母？
(a)4; (b)3; (c)2; (d)1; (e)0。(注：a和b相差一个字母)
8。答案是元音字母的问题的个数是：
(a)2; (b)3; (c)4; (d)5; (e)6。(注：a和e是元音字母)
9。答案是辅音字母的问题的个数是：
(a)一个质数; (b)一个阶乘数; (c)一个平方数; (d)一个立方数，(e)5的倍数
10。本问题的答案是：
(a)a; (b)b; (c)c; (d)d; (e)e。
'''

def count(ans):
    n_abcd = {'a':0,'b':0,'c':0,'d':0,'e':0,'vowel':0,'consomant':0}
    for x in ans:
        n_abcd[x] += 1
        if(x=='a' or x=='e'):
            n_abcd['vowel'] += 1
        else:
            n_abcd['consomant'] += 1
    return n_abcd

def is_true_ans(ans):
    #q1
    if ans[0]=='a' and not(ans[0]!='b' and ans[1]=='b'):
        return False
    elif ans[0]=='b' and not(ans[0]!='b' and ans[1]!='b' and ans[2]=='b'):
        return False
    elif ans[0]=='c' and not(ans[0]!='b' and ans[1]!='b' and ans[2]!='b' and ans[3]=='b'):
        return False
    elif ans[0]=='d' and not(ans[0]!='b' and ans[1]!='b' and ans[2]!='b' and ans[3]!='b' and ans[4]=='b'):
        return False
    elif ans[0]=='e' and not(ans[0]!='b' and ans[1]!='b' and ans[2]!='b' and ans[3]!='b' and ans[4]!='b' and ans[5]=='b'):
        return False
    #q2
    two_same = {(i,i+1):ans[i]==ans[i+1] for i in range(len(ans)-1)}
    if list(two_same.values()).count(True)!=1:
        return False
    elif ans[1]=='a' and not(two_same[(1,2)]):
        return False
    elif ans[1]=='b' and not(two_same[(2,3)]):
        return False
    elif ans[1]=='c' and not(two_same[(3,4)]):
        return False
    elif ans[1]=='d' and not(two_same[(4,5)]):
        return False
    elif ans[1]=='e' and not(two_same[(5,6)]):
        return False
    #q3
    if ans[2]=='a' and not(ans[0]=='a'):
        return False
    elif ans[2]=='b' and not(ans[1]=='b'):
        return False
    elif ans[2]=='c' and not(ans[3]=='c'):
        return False
    elif ans[2]=='d' and not(ans[6]=='d'):
        return False
    elif ans[2]=='e' and not(ans[5]=='e'):
        return False
    nCount = count(ans)
    #q4
    if ans[3]=='a' and not(nCount['a']==0):
        return False
    elif ans[3]=='b' and not(nCount['b']==1):
        return False
    elif ans[3]=='c' and not(nCount['c']==2):
        return False
    elif ans[3]=='d' and not(nCount['d']==3):
        return False
    elif ans[3]=='e' and not(nCount['e']==4):
        return False
    #q5
    if ans[4]=='a' and not(ans[4]==ans[9]):
        return False
    elif ans[4]=='b' and not(ans[4]==ans[8]):
        return False
    elif ans[4]=='c' and not(ans[4]==ans[7]):
        return False
    elif ans[4]=='d' and not(ans[4]==ans[6]):
        return False
    elif ans[4]=='e' and not(ans[4]==ans[5]):
        return False
    #q6
    if ans[5]=='a' and not(nCount['a']==nCount['b']):
        return False
    elif ans[5]=='b' and not(nCount['a']==nCount['c']):
        return False
    elif ans[5]=='c' and not(nCount['a']==nCount['d']):
        return False
    elif ans[5]=='d' and not(nCount['a']==nCount['e']):
        return False
    elif ans[5]=='e' and not(nCount['a']!=nCount['b'] and nCount['a']!=nCount['c'] and nCount['a']!=nCount['d'] and nCount['a']!=nCount['e']):
        return False
    #q7
    if ans[6]=='a' and not(abs(ord(ans[6])-ord(ans[7]))==4):
        return False
    elif ans[6]=='b' and not(abs(ord(ans[6])-ord(ans[7]))==3):
        return False
    elif ans[6]=='c' and not(abs(ord(ans[6])-ord(ans[7]))==2):
        return False
    elif ans[6]=='d' and not(abs(ord(ans[6])-ord(ans[7]))==1):
        return False
    elif ans[6]=='e' and not(abs(ord(ans[6])-ord(ans[7]))==0):
        return False
    #q8
    if ans[7]=='a' and not(nCount['vowel']==2):
        return False
    elif ans[7]=='b' and not(nCount['vowel']==3):
        return False
    elif ans[7]=='c' and not(nCount['vowel']==4):
        return False
    elif ans[7]=='d' and not(nCount['vowel']==5):
        return False
    elif ans[7]=='e' and not(nCount['vowel']==6):
        return False
    #q9
    r = nCount['consomant']
    if ans[8]=='a' and not(r==2 or r==3 or r==5 or r==7):
        return False
    elif ans[8]=='b' and not(r==1 or r==2 or r==6):
        return False
    elif ans[8]=='c' and not(r==0 or r==1 or r==4 or r==9):
        return False
    elif ans[8]=='d' and not(r==0 or r==1 or r==8):
        return False
    elif ans[8]=='e' and not(r==0 or r==5 or r==10):
        return False
    #q10
    if ans[9]=='a' and not(ans[9]=='a'):
        return False
    elif ans[9]=='b' and not(ans[9]=='b'):
        return False
    elif ans[9]=='c' and not(ans[9]=='c'):
        return False
    elif ans[9]=='d' and not(ans[9]=='d'):
        return False
    elif ans[9]=='e' and not(ans[9]=='e'):
        return False
    return True
    

if __name__=='__main__':
    all_ans = [[]]
    ans_char = ['a','b','c','d','e']
    for i in range(10):
        all_ans = [x+[y] for x in all_ans for y in ans_char]

    # print(all_ans[10086])
    # print(count(all_ans[10086]))
    # print(5**10)

    ans_true = []
    for ans in all_ans:
        if is_true_ans(ans):
            ans_true.append(ans)
            print(ans)
    #print(ans_true)
