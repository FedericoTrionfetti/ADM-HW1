# **Introduction**

"""Say "Hello, World!" With Python"""

print("Hello, World!")

"""Python If-Else"""

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n % 2 == 1:
    print ('Weird')
elif n<6 and n>1:
    print ('Not Weird')
elif n<21 and n>5:
    print ('Weird')
elif n>19:
    print ('Not Weird')

"""Arithmetic Operators"""

if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a+b)
print(a-b)
print(a*b)

"""Python: Division"""

if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)

"""Loops"""

if __name__ == '__main__':
    n = int(input())

for i in range (0,n):
    if i >= 0:
        print (i**2)

"""Write a function"""

def is_leap(year):
    leap = False

    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                leap = True
        else:
            leap = True
    return leap

"""Print Function"""

if __name__ == '__main__':
    n = int(input())

for i in range(1,n+1):
    print(i, end="")

"""# **Basic Data Types**

List Comprehensions
"""

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

l=[]
for i in range(0,x+1):
    for j in range (0,y+1):
        for k in range (0,z+1):
            if i+j+k != n:
                l.append([i,j,k])
print(l)

"""Find the Runner-Up Score!"""

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

arr=list(arr)
m=max(arr)
while m in arr:
    arr.remove(m)

ma=max(arr)
print(ma)

"""Nested Lists"""

if __name__ == '__main__':
    l=[]
    scores=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        scores.append(score)
        l.append([name, score])


    last=min(scores)
    while last in scores:
        scores.remove(last)
    second =min(scores)

    sls=[]
    for student in l:
        if student[1]== second:
            sls.append(student[0])

    sls.sort()

    for a in sls:
        print(a)

"""Finding the percentage"""

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

m= sum(student_marks[query_name])/len(student_marks[query_name])
print(f"{m:.2f}")

"""Lists"""

if __name__ == '__main__':
    N = int(input())
lst = []

for i in range(N):
        command =input().split()
        action = command[0]

        if action== "insert":
            lst.insert(int(command[1]), int(command[2]))
        elif action== "print":
            print(lst)
        elif action== "remove":
            lst.remove(int(command[1]))
        elif action== "append":
            lst.append(int(command[1]))
        elif action== "sort":
            lst.sort()
        elif action== "pop":
            lst.pop()
        elif action== "reverse":
            lst.reverse()

"""Tuples"""

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
t=tuple(integer_list)
print(hash(t))

"""# **Strings**

sWAP cASE
"""

def swap_case(s):
    swapped = []
    for char in s:
        if char.islower():
            swapped.append(char.upper())
        elif char.isupper():
            swapped.append(char.lower())
        else:
            swapped.append(char)
    return ''.join(swapped)

"""String Split and Join"""

def split_and_join(line):
    return "-".join(line.split())

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

"""What's Your Name?"""

def print_full_name(first, last):
    message = f"Hello {first} {last}! You just delved into python."
    print(message)

"""Mutations"""

def mutate_string(string,position,character):
    l=list(string)
    l[position]=character
    return"".join(l)

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

"""Find a string"""

def count_substring(string, sub_string):
    count = 0
    substring_length = len(sub_string)

    for i in range(len(string)-substring_length + 1):
        if string[i:i +substring_length] ==sub_string:
            count+= 1

    return count

"""String Validators"""

if __name__ == '__main__':
    s = input()

    flag_alnum=False
    flag_alpha=False
    flag_digit=False
    flag_lower=False
    flag_upper=False
    for i in s:
        if i.isalnum():
            flag_alnum=True
        if i.isalpha():
            flag_alpha=True
        if i.isdigit():
            flag_digit=True
        if i.islower():
            flag_lower=True
        if i.isupper():
            flag_upper=True

    print(flag_alnum)
    print(flag_alpha)
    print(flag_digit)
    print(flag_lower)
    print(flag_upper)

"""Text Alignment"""

thickness = int(input())
c = 'H'

for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))

for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))

"""Text Wrap"""

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

"""Designer Door Mat"""

N, M = map(int, input().split())

for i in range(1, N,2):
    print((".|." * i).center(M,'-'))

print("WELCOME".center(M, '-'))

for  i in range(N-2,-1,-2):
    print((".|."*i).center(M, '-'))

"""String Formatting"""

def print_formatted(number):
    width= len(bin(number))- 2
    for i in range(1, number+1):
        print(f"{i:>{width}d} {i:>{width}o} {i:>{width}X} {i:>{width}b}")

"""Alphabet Rangoli"""

def print_rangoli(size):
    alpha= 'abcdefghijklmnopqrstuvwxyz'
    L=[]
    for i in range(size):
        s= '-'.join(alpha[size-1:i:-1]+ alpha[i:size])
        L.append(s.center(4*size-3,'-'))
    print('\n'.join(L[:0:-1] + L))

"""Capitalize!"""

def solve(s):
    result = ''

    for i in range(len(s)):
        if i == 0 or s[i-1] == ' ':
            result += s[i].upper()
        else:
            result += s[i].lower()
    return result

"""The Minion Game"""

def minion_game(string):
    vowels='AEIOU'
    kevin_score=0
    stuart_score= 0
    n= len(string)

    for i in range(n):
        if string[i] in vowels:
            kevin_score+=n-i
        else:
            stuart_score+=n-i

    if kevin_score > stuart_score:
        print(f"Kevin {kevin_score}")
    elif stuart_score > kevin_score:
        print(f"Stuart {stuart_score}")
    else:
        print("Draw")

"""Merge the Tools!"""

def merge_the_tools(string, k):
       for i in range(0, len(string), k):
        t= string[i:i+k]
        u = ""
        for char in t:
            if char not in u:
                u+= char
        print(u)

"""# **Sets**

Introduction to Sets
"""

def average(array):
    distinct_heights = set(array)
    return round(sum(distinct_heights) / len(distinct_heights), 3)

"""No Idea!"""

n,m=map(int,input().split())
array=list(map(int,input().split()))
A=set(map(int,input().split()))
B=set(map(int,input().split()))

happiness=0
for element in array:
    if element in A: happiness+=1
    elif element in B: happiness-=1
print(happiness)

"""Symmetric Difference"""

a_size= int(input())
a= set(map(int, input().split()))
b_size= int(input())
b = set(map(int,input().split()))

symmetric_difference=sorted(a.symmetric_difference(b))
for num in symmetric_difference:
    print(num)

"""Set .add()"""

n = int(input())
stamps = set()
for _ in range(n):
    country = input()
    stamps.add(country)
print(len(stamps))

"""Set .discard(), .remove() & .pop()"""

n=int(input())
nums=set(map(int,input().split()))
commands=int(input())
for _ in range(commands):
    cmd=list(input().split())
    if len(cmd)==1:
        nums.pop()
    else:
        val=int(cmd[1])
        op=cmd[0]
        if op=="discard":
            nums.discard(val)
        else:
            nums.remove(val)
print(sum(nums))

"""Set .union() Operation"""

n=int(input())
english_subscribers = set(map(int, input().split()))
m=int(input())
french_subscribers= set(map(int, input().split()))
total_subscribers =english_subscribers.union(french_subscribers)
print(len(total_subscribers))

"""Set .intersection() Operation"""

n=int(input())
english_subscribers =set(map(int, input().split()))
m=int(input())
french_subscribers= set(map(int, input().split()))
both_subscribers= english_subscribers.intersection(french_subscribers)
print(len(both_subscribers))

"""Set .difference() Operation"""

n=int(input())
english_subscribers =set(map(int, input().split()))
m=int(input())
french_subscribers= set(map(int, input().split()))
only_english_subscribers= english_subscribers.difference(french_subscribers)
print(len(only_english_subscribers))

"""Set .symmetric_difference() Operation"""

n=int(input())
english_subscribers =set(map(int, input().split()))
m=int(input())
french_subscribers= set(map(int, input().split()))
either_subscribers =english_subscribers.symmetric_difference(french_subscribers)
print(len(either_subscribers))

"""Set Mutations"""

n=int(input())
s=set(map(int, input().split()))
m=int(input())

for _ in range(m):
    operation, _ =input().split()
    other_set=set(map(int, input().split()))
    if operation== 'update':
        s.update(other_set)
    elif operation== 'intersection_update':
        s.intersection_update(other_set)
    elif operation=='difference_update':
        s.difference_update(other_set)
    elif operation=='symmetric_difference_update':
        s.symmetric_difference_update(other_set)

print(sum(s))

"""The Capitan's Room"""

from collections import Counter

k=int(input())
room_numbers=list(map(int, input().split()))

room_count=Counter(room_numbers)

for room, count in room_count.items():
    if count == 1:
        captain_room=room
        break

print(captain_room)

"""Check Subset"""

t=int(input())

for _ in range(t):
    n=int(input())
    set_a=set(map(int, input().split()))
    m=int(input())
    set_b=set(map(int, input().split()))

    print(set_a.issubset(set_b))

"""Check Strict Superset"""

set_a=set(map(int, input().split()))
n=int(input())
is_strict_superset=True

for _ in range(n):
    set_b=set(map(int, input().split()))

    if not (set_a > set_b):
        is_strict_superset=False
        break

print(is_strict_superset)

"""# **Collections**

collections.Counter()
"""

from collections import Counter

num_shoes=int(input().strip())
shoe_sizes=list(map(int, input().strip().split()))
num_customers=int(input().strip())

available_sizes=Counter(shoe_sizes)

total_earnings=0

for _ in range(num_customers):
    size,price = map(int, input().strip().split())
    if available_sizes[size] > 0:
        total_earnings+= price
        available_sizes[size]-=1

print(total_earnings)

"""DefaultDict Tutorial"""

from collections import defaultdict

d=defaultdict(list)
n,m=map(int,input().split())
for i in range(n):
    w=input()
    d[w].append(str(i+1))
for _ in range(m):
    w=input()
    print(" ".join(d[w])or -1)

"""Collections.namedtuple()"""

from collections import namedtuple

n=int(input())
cols=",".join(input().split())
Student=namedtuple("Student",cols)
records=[]
for _ in range(n):
    entry=input().split()
    student=Student._make(entry)
    records.append(student)
average=sum(float(s.MARKS)for s in records)/n
print(f"{average:.2f}")

"""Collections.OrderedDict()"""

from collections import OrderedDict

n=int(input())
order=OrderedDict()

for _ in range(n):
    line =input().strip().split()
    item_name= ' '.join(line[:-1])
    price = int(line[-1])

    if item_name in order:
        order[item_name]+=price
    else:
        order[item_name]=price

for item_name, net_price in order.items():
    print(item_name, net_price)

"""Word Order"""

from collections import Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    pass

words=[]
n=int(input())
for _ in range(n):
    words.append(input().strip())
word_count=OrderedCounter(words)
print(len(word_count))
for word in word_count:
    print(word_count[word],end=" ")

"""Collections.deque()"""

from collections import deque

d=deque()
n=int(input())

for _ in range(n):
    operation=input().strip().split()
    method=operation[0]

    if method== 'append':
        d.append(int(operation[1]))
    elif method== 'appendleft':
        d.appendleft(int(operation[1]))
    elif method== 'pop':
        d.pop()
    elif method== 'popleft':
        d.popleft()

print(" ".join(map(str, d)))

"""Company Logo"""

#!/bin/python3

import math
import os
import random
import re
import sys

from collections import Counter

s = input().strip()
char_count = Counter(s)
sorted_chars = sorted(char_count.items(), key=lambda x: (-x[1], x[0]))
top_three = sorted_chars[:3]

for char, count in top_three:
    print(f"{char} {count}")

"""Piling Up!"""

test_cases=int(input())
for _ in range(test_cases):
    n=int(input())
    cubes=list(map(int, input().split()))

    left, right= 0, n - 1
    last=float('inf')
    possible=True

    while left<= right:
        if cubes[left] > last and cubes[right] > last:
            possible= False
            break

        if cubes[left]<= last and (cubes[right] > last or cubes[left] >= cubes[right]):
            last= cubes[left]
            left+=1
        else:
            last = cubes[right]
            right-=1

    print("Yes" if possible else "No")

"""# **Date and Time**

Calendar Module
"""

import calendar
input_date=input().strip()
month,day,year=map(int,input_date.split())
day_name=calendar.day_name[calendar.weekday(year,month,day)].upper()
print(day_name)

"""Time Delta"""

#!/bin/python3

import math
import os
import random
import re
import sys
import datetime


def time_delta(t1, t2):
    format_string = "%a %d %b %Y %H:%M:%S %z"
    time1 = datetime.datetime.strptime(t1, format_string)
    time2 = datetime.datetime.strptime(t2, format_string)
    return str(int(abs((time1 - time2).total_seconds())))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

"""# **Exceptions**

Exceptions
"""

tc = int(input())
for i in range(tc):
    a, b = input().split()

    if b == '0':
        print("Error Code: integer division or modulo by zero")
    elif not (a[0] == '-' and a[1:].isdigit()) and not a.isdigit():
        print("Error Code: invalid literal for int() with base 10: '{}'".format(a))
    elif not (b[0] == '-' and b[1:].isdigit()) and not b.isdigit():
        print("Error Code: invalid literal for int() with base 10: '{}'".format(b))
    else:
        print(int(a) // int(b))

"""# **Built-ins**

Zipped!
"""

N,X=map(int,input().split())
score_list=[]
for _ in range(X):
    score_list.append(list(map(float,input().split())))
for i in zip(*score_list):
    print(sum(i)/len(i))

"""Athlete Sort

"""

#!/bin/python3

import math
import os
import random
import re
import sys


if __name__=="__main__":
    n,m=map(int,input().split())
    matrix=[]
    for _ in range(n):
        matrix.append(list(map(int,input().rstrip().split())))
    k=int(input())
    for row in sorted(matrix,key=lambda x:x[k]):
        print(*row)

"""ginortS"""

s=input()
lower=[]
upper=[]
odd_digits=[]
even_digits=[]

for char in s:
    if char.islower():
        lower.append(char)
    elif char.isupper():
        upper.append(char)
    elif char.isdigit():
        if int(char) % 2 !=0:
            odd_digits.append(char)
        else:
            even_digits.append(char)

result = ''.join(sorted(lower) +sorted(upper) +sorted(odd_digits)+sorted(even_digits))
print(result)

"""# **Python Functionals**

Map and Lambda Function
"""

cube = lambda x: x ** 3

def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib[:n]

"""# **Regex and Parsing challenges**

Detect Floating Point Number
"""

from re import compile

pattern=compile("^[-+]?\d*\.\d+$")
for _ in range(int(input())):
    print(bool(pattern.match(input())))

"""Re.split()"""

regex_pattern = r"[,.]"	# Do not delete 'r'.

"""Group(), Groups() & Groupdict()"""

import re

s=input()
result=re.search(r"([A-Za-z0-9])\1",s)
if result is None:
    print(-1)
else:
    print(result[1])

"""Re.findall() & Re.finditer()"""

import re

s=input()
vowel_pairs=re.findall(
    r"(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([AEIOUaeiou]{2,})(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])",
    s,
)
if vowel_pairs:
    for pair in vowel_pairs:
        print(pair)
else:
    print(-1)

"""Re.start() & Re.end()"""

import re

s = input().strip()
k = input().strip()
l= len(s)
rd=False
for i in range(l):
    tr=re.match(k, s[i:])
    if tr:
        start_index= i+tr.start()
        end_index=i+tr.end()- 1
        print((start_index, end_index))
        rd=True
if rd==False:
    print("(-1, -1)")

"""Regex Substitution"""

import re
n=int(input())
for _ in range(n):
    line=input()
    line=re.sub(r"(?<= )&&(?= )","and",line)
    line=re.sub(r"(?<= )\|\|(?= )","or",line)
    print(line)

"""Validating Roman Numerals"""

regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

"""Validating phone numbers"""

from re import compile, match

n=int(input())
for _ in range(n):
    phone=input()
    condition=compile(r"^[7-9]\d{9}$")
    if bool(match(condition, phone)):
        print("YES")
    else:
        print("NO")

"""Validating and Parsing Email Addresses"""

import email.utils
import re

n=int(input())
for _ in range(n):
    email_input=input()
    parsed_email=email.utils.parseaddr(email_input)[1].strip()
    is_valid=bool(
        re.match(
            r"(^[A-Za-z][A-Za-z0-9\._-]+)@([A-Za-z]+)\.([A-Za-z]{1,3})$", parsed_email
        )
    )
    if is_valid:
        print(email_input)

"""Hex Color Code"""

import re

n=int(input())
for _ in range(n):
    color_string=input()
    matches=re.findall(r"(#[0-9A-Fa-f]{3}|#[0-9A-Fa-f]{6})(?:[;,.)]{1})", color_string)
    for color in matches:
        if color:
            print(color)

"""HTML Parser - Part 1"""

from html.parser import HTMLParser

class CustomHTMLParser(HTMLParser):
    def handle_attr(self, attributes):
        for attr_tuple in attributes:
            print("->", attr_tuple[0], ">", attr_tuple[1])

    def handle_starttag(self, tag, attributes):
        print("Start :", tag)
        self.handle_attr(attributes)

    def handle_endtag(self, tag):
        print("End   :", tag)

    def handle_startendtag(self, tag, attributes):
        print("Empty :", tag)
        self.handle_attr(attributes)
parser=CustomHTMLParser()
n=int(input())
s="".join(input() for _ in range(n))
parser.feed(s)

"""HTML Parser - Part 2"""

from html.parser import HTMLParser

class CustomHTMLParser(HTMLParser):
    def handle_comment(self, data):
        line_count=len(data.split("\n"))
        if line_count > 1:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        if data.strip():
            print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

parser=CustomHTMLParser()

n=int(input())
html_string="".join(input().rstrip()+"\n" for _ in range(n))
parser.feed(html_string)
parser.close()

"""Detect HTML Tags, Attributes and Attribute Values"""

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

html = ""
for _ in range(int(input())):
    html+=input().rstrip() + '\n'

parser=MyHTMLParser()
parser.feed(html)

"""Validating UID"""

import re

n=int(input())
upper_check=r".*([A-Z].*){2,}"
digit_check=r".*([0-9].*){3,}"
alphanumeric_and_length_check=r"([A-Za-z0-9]){10}$"
repeat_check=r".*(.).*\1"

for _ in range(n):
    uid=input().strip()
    upper_check_result=bool(re.match(upper_check, uid))
    digit_check_result=bool(re.match(digit_check, uid))
    alphanumeric_and_length_check_result=bool(re.match(alphanumeric_and_length_check, uid))
    repeat_check_result=bool(re.match(repeat_check, uid))

    if (upper_check_result and digit_check_result and alphanumeric_and_length_check_result and not repeat_check_result):
        print("Valid")
    else:
        print("Invalid")

"""Validating Credit Card Numbers"""

import re

n=int(input())
for _ in range(n):
    credit=input().strip()
    credit_no_hyphen=credit.replace("-", "")
    valid=True
    length_16=bool(re.match(r"^[4-6]\d{15}$", credit))
    length_19=bool(re.match(r"^[4-6]\d{3}-\d{4}-\d{4}-\d{4}$", credit))
    consecutive=bool(re.findall(r"(?=(\d)\1\1\1)", credit_no_hyphen))

    if length_16 or length_19:
        if consecutive:
            valid=False
    else:
        valid=False

    if valid:
        print("Valid")
    else:
        print("Invalid")

"""Validating Postal Codes"""

regex_integer_in_range = r"^(100000|[1-9]\d{5}|[1-9]\d{0,5}|0{6})$" 	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(?=(\d)(?=\d\1))"	# Do not delete 'r'.

"""Matrix Script"""

#!/bin/python3

import math
import os
import random
import re
import sys


n, m=map(int, input().split())
char_array=[""] * (n * m)

for i in range(n):
    line=input()
    for j in range(m):
        char_array[i + (j * n)]=line[j]

decoded_string="".join(char_array)
final_decoded_string=re.sub(r"(?<=[A-Za-z0-9])([ !@#$%&]+)(?=[A-Za-z0-9])", " ", decoded_string)

print(final_decoded_string)

"""# **XML**

XML 1 - Find the Score
"""

def get_attr_number(node):
 score=len(node.attrib)
 for child in node:
  score+=get_attr_number(child)
 return score

"""XML2 - Find the Maximum Depth"""

maxdepth=0
def depth(elem,level):
 global maxdepth
 level+=1
 if level>maxdepth:maxdepth=level
 for child in elem:
  depth(child,level)

"""# **Closures and Decorations**

Standardize Mobile Number Using Decorators
"""

def wrapper(f):
    def fun(l):
        formatted_numbers = ['+91 {} {}'.format(n[-10:-5], n[-5:]) for n in l]
        return f(formatted_numbers)
    return fun

"""Decorators 2 - Name Directory"""

def person_lister(f):
    def inner(people):
        sorted_people = sorted(people, key=lambda x: int(x[2]))
        return [f(person) for person in sorted_people]
    return inner

"""# **Numpy**

Arrays
"""

def arrays(arr):
    np_array = numpy.array(arr, float)
    return np_array[::-1]

"""Shape and Reshape"""

import numpy
arr=input().strip().split()
arr=numpy.array(arr,int)
arr=arr.reshape(3,3)
print(arr)

"""Transpose and Flatten"""

import numpy

n, m=map(int, input().split())
array=[]

for _ in range(n):
    row=list(map(int, input().split()))
    array.append(row)

np_array=numpy.array(array)
print(numpy.transpose(np_array))
print(np_array.flatten())

"""Concatenate"""

import numpy

n, m, p=map(int, input().split())

array1=[]
array2=[]

for _ in range(n):
    temp=list(map(int, input().split()))
    array1.append(temp)

for _ in range(m):
    temp=list(map(int, input().split()))
    array2.append(temp)

np_array1=numpy.array(array1)
np_array2=numpy.array(array2)

print(numpy.concatenate((np_array1, np_array2), axis=0))

"""Zeros and Ones"""

import numpy

shape=tuple(map(int,input().split()))

print(numpy.zeros(shape,dtype=int))
print(numpy.ones(shape,dtype=int))

"""Eye and Identity"""

import numpy
numpy.set_printoptions(legacy='1.13')

n, m = map(int, input().split())
print(numpy.eye(n, m))

"""Array Mathematics"""

import numpy

n, m=map(int, input().split())
array1=[]
array2=[]

for _ in range(n):
    temp=list(map(int, input().split()))
    array1.append(temp)

for _ in range(n):
    temp=list(map(int, input().split()))
    array2.append(temp)

np_array1=numpy.array(array1)
np_array2=numpy.array(array2)

print(np_array1 + np_array2)
print(np_array1 - np_array2)
print(np_array1 * np_array2)
print(np_array1 // np_array2)
print(np_array1 % np_array2)
print(np_array1**np_array2)

"""Floor, Ceil and Rint"""

import numpy

numpy.set_printoptions(legacy='1.13')

a=numpy.array(input().split(),float)
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))

"""Sum and Prod"""

import numpy

n, m=map(int, input().split())
array=[]

for _ in range(n):
    t=list(map(int, input().split()))
    array.append(t)

np_array=numpy.array(array)
print(numpy.max(numpy.min(np_array, axis=1)))

"""Min and Max"""

import numpy


n,m=map(int,input().split())
a=numpy.array([input().split() for _ in range(n)],int)
min_values=numpy.min(a,axis=1)
print(numpy.max(min_values))

"""Mean, Var, and Std"""

import numpy

order=input().split(" ")
arr=[]
for i in range(int(order[0])):
    row=list(map(int,input().split(" ")))
    arr.append(row)
Array=numpy.array(arr)

print(numpy.mean(Array,axis=1))
print(numpy.var(Array,axis=0))
print(round(numpy.std(Array,axis=None),11))

"""Dot and Cross"""

import numpy


n=int(input())
A=[]
for _ in range(n):
    row=list(map(int,input().split(" ")))
    A.append(row)
B=[]
for _ in range(n):
    row=list(map(int,input().split(" ")))
    B.append(row)

MatrixA=numpy.array(A)
MatrixB=numpy.array(B)

print(numpy.dot(MatrixA,MatrixB))

"""Inner and Outer"""

import numpy


A=numpy.array(list(map(int,input().split())))
B=numpy.array(list(map(int,input().split())))

print(numpy.inner(A,B))
print(numpy.outer(A,B))

"""Polynomials"""

import numpy


coefficients=list(map(float,input().split()))
x=float(input())

print(numpy.polyval(coefficients,x))

"""Linear Algebra"""

import numpy


n=int(input())
matrix=[]
for _ in range(n):
    row=list(map(float,input().split()))
    matrix.append(row)

determinant=numpy.linalg.det(matrix)
print(round(determinant,2))
