{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMCO/0PoO//uKEXQJy6wGEA",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/FedericoTrionfetti/ADM-HW1/blob/main/scripts.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Introduction**"
      ],
      "metadata": {
        "id": "OvKurm4FfXWz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Say \"Hello, World!\" With Python"
      ],
      "metadata": {
        "id": "iTrZKqTUflel"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SkFI_P0Ge6P5",
        "outputId": "e0ef3b44-6e97-47b6-92b1-de1649e1b67a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Hello, World!\n"
          ]
        }
      ],
      "source": [
        "print(\"Hello, World!\")\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Python If-Else"
      ],
      "metadata": {
        "id": "4WLe94ZIfndP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    n = int(input().strip())\n",
        "if n % 2 == 1:\n",
        "    print ('Weird')\n",
        "elif n<6 and n>1:\n",
        "    print ('Not Weird')\n",
        "elif n<21 and n>5:\n",
        "    print ('Weird')\n",
        "elif n>19:\n",
        "    print ('Not Weird')\n"
      ],
      "metadata": {
        "id": "e-1fJ3ZifreF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Arithmetic Operators"
      ],
      "metadata": {
        "id": "oWiTeP64frUX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    a = int(input())\n",
        "    b = int(input())\n",
        "print(a+b)\n",
        "print(a-b)\n",
        "print(a*b)\n"
      ],
      "metadata": {
        "id": "uxBtQcwkfrNP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Python: Division"
      ],
      "metadata": {
        "id": "I-FAfpq9frD5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    a = int(input())\n",
        "    b = int(input())\n",
        "print(a//b)\n",
        "print(a/b)\n"
      ],
      "metadata": {
        "id": "QNRsjiS3fq4m"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Loops"
      ],
      "metadata": {
        "id": "xXtulg_efqwG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    n = int(input())\n",
        "\n",
        "for i in range (0,n):\n",
        "    if i >= 0:\n",
        "        print (i**2)\n",
        ""
      ],
      "metadata": {
        "id": "kxXCu9aJfqoq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Write a function"
      ],
      "metadata": {
        "id": "DIzWuxwKfqg7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def is_leap(year):\n",
        "    leap = False\n",
        "\n",
        "    if year % 4 == 0:\n",
        "        if year % 100 == 0:\n",
        "            if year % 400 == 0:\n",
        "                leap = True\n",
        "        else:\n",
        "            leap = True\n",
        "    return leap"
      ],
      "metadata": {
        "id": "80PvVeEffqX4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Print Function"
      ],
      "metadata": {
        "id": "TYb9LH_sfqPy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    n = int(input())\n",
        "\n",
        "for i in range(1,n+1):\n",
        "    print(i, end=\"\")\n"
      ],
      "metadata": {
        "id": "Gjtcu3nIfqID"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Basic Data Types**"
      ],
      "metadata": {
        "id": "t0Dot3lofp_m"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "List Comprehensions"
      ],
      "metadata": {
        "id": "9r6egAGEfpuV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    x = int(input())\n",
        "    y = int(input())\n",
        "    z = int(input())\n",
        "    n = int(input())\n",
        "\n",
        "l=[]\n",
        "for i in range(0,x+1):\n",
        "    for j in range (0,y+1):\n",
        "        for k in range (0,z+1):\n",
        "            if i+j+k != n:\n",
        "                l.append([i,j,k])\n",
        "print(l)"
      ],
      "metadata": {
        "id": "9fzmQCzTfplK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Find the Runner-Up Score!"
      ],
      "metadata": {
        "id": "MbYv_7iIfpaz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    n = int(input())\n",
        "    arr = map(int, input().split())\n",
        "\n",
        "arr=list(arr)\n",
        "m=max(arr)\n",
        "while m in arr:\n",
        "    arr.remove(m)\n",
        "\n",
        "ma=max(arr)\n",
        "print(ma)"
      ],
      "metadata": {
        "id": "9tRk8vPXfpP9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Nested Lists"
      ],
      "metadata": {
        "id": "QpEdqIZVfpCe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    l=[]\n",
        "    scores=[]\n",
        "    for _ in range(int(input())):\n",
        "        name = input()\n",
        "        score = float(input())\n",
        "        scores.append(score)\n",
        "        l.append([name, score])\n",
        "\n",
        "\n",
        "    last=min(scores)\n",
        "    while last in scores:\n",
        "        scores.remove(last)\n",
        "    second =min(scores)\n",
        "\n",
        "    sls=[]\n",
        "    for student in l:\n",
        "        if student[1]== second:\n",
        "            sls.append(student[0])\n",
        "\n",
        "    sls.sort()\n",
        "\n",
        "    for a in sls:\n",
        "        print(a)\n"
      ],
      "metadata": {
        "id": "dl-ctfDGfnKB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Finding the percentage"
      ],
      "metadata": {
        "id": "ctdhSvgxfm1t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    n = int(input())\n",
        "    student_marks = {}\n",
        "    for _ in range(n):\n",
        "        name, *line = input().split()\n",
        "        scores = list(map(float, line))\n",
        "        student_marks[name] = scores\n",
        "    query_name = input()\n",
        "\n",
        "m= sum(student_marks[query_name])/len(student_marks[query_name])\n",
        "print(f\"{m:.2f}\")"
      ],
      "metadata": {
        "id": "yIsQ1J9sfvHX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Lists"
      ],
      "metadata": {
        "id": "NLhPtForKxi8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    N = int(input())\n",
        "lst = []\n",
        "\n",
        "for i in range(N):\n",
        "        command =input().split()\n",
        "        action = command[0]\n",
        "\n",
        "        if action== \"insert\":\n",
        "            lst.insert(int(command[1]), int(command[2]))\n",
        "        elif action== \"print\":\n",
        "            print(lst)\n",
        "        elif action== \"remove\":\n",
        "            lst.remove(int(command[1]))\n",
        "        elif action== \"append\":\n",
        "            lst.append(int(command[1]))\n",
        "        elif action== \"sort\":\n",
        "            lst.sort()\n",
        "        elif action== \"pop\":\n",
        "            lst.pop()\n",
        "        elif action== \"reverse\":\n",
        "            lst.reverse()"
      ],
      "metadata": {
        "id": "2H7U3CAXKxYX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Tuples"
      ],
      "metadata": {
        "id": "30PcEPL-KxOT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    n = int(input())\n",
        "    integer_list = map(int, input().split())\n",
        "t=tuple(integer_list)\n",
        "print(hash(t))\n"
      ],
      "metadata": {
        "id": "6SwCqG4pKxFB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Strings**"
      ],
      "metadata": {
        "id": "U1Zfwnq-Kw5v"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "sWAP cASE"
      ],
      "metadata": {
        "id": "rpXWVwXULlmD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def swap_case(s):\n",
        "    swapped = []\n",
        "    for char in s:\n",
        "        if char.islower():\n",
        "            swapped.append(char.upper())\n",
        "        elif char.isupper():\n",
        "            swapped.append(char.lower())\n",
        "        else:\n",
        "            swapped.append(char)\n",
        "    return ''.join(swapped)\n"
      ],
      "metadata": {
        "id": "ewmR_WOYKwyr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "String Split and Join"
      ],
      "metadata": {
        "id": "WWlEKnQPKwl8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def split_and_join(line):\n",
        "    return \"-\".join(line.split())\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    line = input()\n",
        "    result = split_and_join(line)\n",
        "    print(result)\n"
      ],
      "metadata": {
        "id": "cX86XdvJKweX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "What's Your Name?"
      ],
      "metadata": {
        "id": "vIOEDXGbKwGL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def print_full_name(first, last):\n",
        "    message = f\"Hello {first} {last}! You just delved into python.\"\n",
        "    print(message)"
      ],
      "metadata": {
        "id": "iWM4NmDRKv-O"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Mutations"
      ],
      "metadata": {
        "id": "HnY7Qb_CKv1K"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def mutate_string(string,position,character):\n",
        "    l=list(string)\n",
        "    l[position]=character\n",
        "    return\"\".join(l)\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    s = input()\n",
        "    i, c = input().split()\n",
        "    s_new = mutate_string(s, int(i), c)\n",
        "    print(s_new)\n"
      ],
      "metadata": {
        "id": "kJ1MlWIsKvt6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Find a string"
      ],
      "metadata": {
        "id": "6fz-bE4BKvjb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def count_substring(string, sub_string):\n",
        "    count = 0\n",
        "    substring_length = len(sub_string)\n",
        "\n",
        "    for i in range(len(string)-substring_length + 1):\n",
        "        if string[i:i +substring_length] ==sub_string:\n",
        "            count+= 1\n",
        "\n",
        "    return count\n",
        "\n"
      ],
      "metadata": {
        "id": "QqS08S8zKvbK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "String Validators"
      ],
      "metadata": {
        "id": "ZVgXnjVnKvPo"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == '__main__':\n",
        "    s = input()\n",
        "\n",
        "    flag_alnum=False\n",
        "    flag_alpha=False\n",
        "    flag_digit=False\n",
        "    flag_lower=False\n",
        "    flag_upper=False\n",
        "    for i in s:\n",
        "        if i.isalnum():\n",
        "            flag_alnum=True\n",
        "        if i.isalpha():\n",
        "            flag_alpha=True\n",
        "        if i.isdigit():\n",
        "            flag_digit=True\n",
        "        if i.islower():\n",
        "            flag_lower=True\n",
        "        if i.isupper():\n",
        "            flag_upper=True\n",
        "\n",
        "    print(flag_alnum)\n",
        "    print(flag_alpha)\n",
        "    print(flag_digit)\n",
        "    print(flag_lower)\n",
        "    print(flag_upper)\n"
      ],
      "metadata": {
        "id": "wsaOZplYL2N4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Text Alignment"
      ],
      "metadata": {
        "id": "Vb-cWNUVL2yV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "thickness = int(input())\n",
        "c = 'H'\n",
        "\n",
        "for i in range(thickness):\n",
        "    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))\n",
        "\n",
        "for i in range(thickness + 1):\n",
        "    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))\n",
        "\n",
        "for i in range((thickness + 1) // 2):\n",
        "    print((c * thickness * 5).center(thickness * 6))\n",
        "\n",
        "for i in range(thickness + 1):\n",
        "    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))\n",
        "\n",
        "for i in range(thickness):\n",
        "    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))"
      ],
      "metadata": {
        "id": "Gglp89YeL2po"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Text Wrap"
      ],
      "metadata": {
        "id": "nhj3kgwJL7lR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def wrap(string, max_width):\n",
        "    return textwrap.fill(string, max_width)\n"
      ],
      "metadata": {
        "id": "UWVkOhR4L7es"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Designer Door Mat"
      ],
      "metadata": {
        "id": "NlJ1jpbFL7VN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "N, M = map(int, input().split())\n",
        "\n",
        "for i in range(1, N,2):\n",
        "    print((\".|.\" * i).center(M,'-'))\n",
        "\n",
        "print(\"WELCOME\".center(M, '-'))\n",
        "\n",
        "for  i in range(N-2,-1,-2):\n",
        "    print((\".|.\"*i).center(M, '-'))\n"
      ],
      "metadata": {
        "id": "BZCfyfahL7Nn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "String Formatting"
      ],
      "metadata": {
        "id": "7UB2d_95L7EL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def print_formatted(number):\n",
        "    width= len(bin(number))- 2\n",
        "    for i in range(1, number+1):\n",
        "        print(f\"{i:>{width}d} {i:>{width}o} {i:>{width}X} {i:>{width}b}\")\n"
      ],
      "metadata": {
        "id": "oSRoLVEsL68u"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Alphabet Rangoli"
      ],
      "metadata": {
        "id": "ss17PP4TL6zL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def print_rangoli(size):\n",
        "    alpha= 'abcdefghijklmnopqrstuvwxyz'\n",
        "    L=[]\n",
        "    for i in range(size):\n",
        "        s= '-'.join(alpha[size-1:i:-1]+ alpha[i:size])\n",
        "        L.append(s.center(4*size-3,'-'))\n",
        "    print('\\n'.join(L[:0:-1] + L))\n",
        "\n"
      ],
      "metadata": {
        "id": "RSdnFq_cL6r0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Capitalize!"
      ],
      "metadata": {
        "id": "YnoVL9cdL6id"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def solve(s):\n",
        "    result = ''\n",
        "\n",
        "    for i in range(len(s)):\n",
        "        if i == 0 or s[i-1] == ' ':\n",
        "            result += s[i].upper()\n",
        "        else:\n",
        "            result += s[i].lower()\n",
        "    return result\n"
      ],
      "metadata": {
        "id": "zSFAkNdQL6Xw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "The Minion Game"
      ],
      "metadata": {
        "id": "mZN43I-zL56R"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def minion_game(string):\n",
        "    vowels='AEIOU'\n",
        "    kevin_score=0\n",
        "    stuart_score= 0\n",
        "    n= len(string)\n",
        "\n",
        "    for i in range(n):\n",
        "        if string[i] in vowels:\n",
        "            kevin_score+=n-i\n",
        "        else:\n",
        "            stuart_score+=n-i\n",
        "\n",
        "    if kevin_score > stuart_score:\n",
        "        print(f\"Kevin {kevin_score}\")\n",
        "    elif stuart_score > kevin_score:\n",
        "        print(f\"Stuart {stuart_score}\")\n",
        "    else:\n",
        "        print(\"Draw\")"
      ],
      "metadata": {
        "id": "RUtRyIOZL5Zs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Merge the Tools!"
      ],
      "metadata": {
        "id": "1QzGc1gtL2er"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def merge_the_tools(string, k):\n",
        "       for i in range(0, len(string), k):\n",
        "        t= string[i:i+k]\n",
        "        u = \"\"\n",
        "        for char in t:\n",
        "            if char not in u:\n",
        "                u+= char\n",
        "        print(u)"
      ],
      "metadata": {
        "id": "QPG4KM1OMSMG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Sets**"
      ],
      "metadata": {
        "id": "2qDMH6nBMSUV"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Introduction to Sets"
      ],
      "metadata": {
        "id": "snumVmIWOfVu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def average(array):\n",
        "    distinct_heights = set(array)\n",
        "    return round(sum(distinct_heights) / len(distinct_heights), 3)"
      ],
      "metadata": {
        "id": "yrJfBmBnOfHj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "No Idea!"
      ],
      "metadata": {
        "id": "az3V6BaNOe6C"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "n,m=map(int,input().split())\n",
        "array=list(map(int,input().split()))\n",
        "A=set(map(int,input().split()))\n",
        "B=set(map(int,input().split()))\n",
        "\n",
        "happiness=0\n",
        "for element in array:\n",
        "    if element in A: happiness+=1\n",
        "    elif element in B: happiness-=1\n",
        "print(happiness)\n"
      ],
      "metadata": {
        "id": "psZn_bduOetx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Symmetric Difference"
      ],
      "metadata": {
        "id": "IDnbXxFBOegQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "a_size= int(input())\n",
        "a= set(map(int, input().split()))\n",
        "b_size= int(input())\n",
        "b = set(map(int,input().split()))\n",
        "\n",
        "symmetric_difference=sorted(a.symmetric_difference(b))\n",
        "for num in symmetric_difference:\n",
        "    print(num)"
      ],
      "metadata": {
        "id": "0p0N-F7MOeTS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Set .add()"
      ],
      "metadata": {
        "id": "CEFTF77oOeGw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "n = int(input())\n",
        "stamps = set()\n",
        "for _ in range(n):\n",
        "    country = input()\n",
        "    stamps.add(country)\n",
        "print(len(stamps))\n"
      ],
      "metadata": {
        "id": "wAxJ_70gOd6e"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Set .discard(), .remove() & .pop()"
      ],
      "metadata": {
        "id": "PRXOJDwGOdsc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "n=int(input())\n",
        "nums=set(map(int,input().split()))\n",
        "commands=int(input())\n",
        "for _ in range(commands):\n",
        "    cmd=list(input().split())\n",
        "    if len(cmd)==1:\n",
        "        nums.pop()\n",
        "    else:\n",
        "        val=int(cmd[1])\n",
        "        op=cmd[0]\n",
        "        if op==\"discard\":\n",
        "            nums.discard(val)\n",
        "        else:\n",
        "            nums.remove(val)\n",
        "print(sum(nums))"
      ],
      "metadata": {
        "id": "QW-eVuE4OdgL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Set .union() Operation"
      ],
      "metadata": {
        "id": "He_Uwh2sOdR3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "n=int(input())\n",
        "english_subscribers = set(map(int, input().split()))\n",
        "m=int(input())\n",
        "french_subscribers= set(map(int, input().split()))\n",
        "total_subscribers =english_subscribers.union(french_subscribers)\n",
        "print(len(total_subscribers))\n"
      ],
      "metadata": {
        "id": "UgnTeKyXOdEE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Set .intersection() Operation"
      ],
      "metadata": {
        "id": "k-nmR4NxOcg-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "n=int(input())\n",
        "english_subscribers =set(map(int, input().split()))\n",
        "m=int(input())\n",
        "french_subscribers= set(map(int, input().split()))\n",
        "both_subscribers= english_subscribers.intersection(french_subscribers)\n",
        "print(len(both_subscribers))\n"
      ],
      "metadata": {
        "id": "PSkkGHijOcTg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Set .difference() Operation"
      ],
      "metadata": {
        "id": "I19oGGNvOcFE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "n=int(input())\n",
        "english_subscribers =set(map(int, input().split()))\n",
        "m=int(input())\n",
        "french_subscribers= set(map(int, input().split()))\n",
        "only_english_subscribers= english_subscribers.difference(french_subscribers)\n",
        "print(len(only_english_subscribers))"
      ],
      "metadata": {
        "id": "zR7rs3BPOb4F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Set .symmetric_difference() Operation"
      ],
      "metadata": {
        "id": "hh4PczQFObqu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "n=int(input())\n",
        "english_subscribers =set(map(int, input().split()))\n",
        "m=int(input())\n",
        "french_subscribers= set(map(int, input().split()))\n",
        "either_subscribers =english_subscribers.symmetric_difference(french_subscribers)\n",
        "print(len(either_subscribers))\n"
      ],
      "metadata": {
        "id": "1-4iWsmOOba8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Set Mutations"
      ],
      "metadata": {
        "id": "i6Sn1xGbObRK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "n=int(input())\n",
        "s=set(map(int, input().split()))\n",
        "m=int(input())\n",
        "\n",
        "for _ in range(m):\n",
        "    operation, _ =input().split()\n",
        "    other_set=set(map(int, input().split()))\n",
        "    if operation== 'update':\n",
        "        s.update(other_set)\n",
        "    elif operation== 'intersection_update':\n",
        "        s.intersection_update(other_set)\n",
        "    elif operation=='difference_update':\n",
        "        s.difference_update(other_set)\n",
        "    elif operation=='symmetric_difference_update':\n",
        "        s.symmetric_difference_update(other_set)\n",
        "\n",
        "print(sum(s))"
      ],
      "metadata": {
        "id": "DzHNYS6QObKf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "The Capitan's Room"
      ],
      "metadata": {
        "id": "uReV9BBrObBC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter\n",
        "\n",
        "k=int(input())\n",
        "room_numbers=list(map(int, input().split()))\n",
        "\n",
        "room_count=Counter(room_numbers)\n",
        "\n",
        "for room, count in room_count.items():\n",
        "    if count == 1:\n",
        "        captain_room=room\n",
        "        break\n",
        "\n",
        "print(captain_room)\n"
      ],
      "metadata": {
        "id": "8G5OqvjgOay0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Check Subset"
      ],
      "metadata": {
        "id": "RTF_L42MOapR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "t=int(input())\n",
        "\n",
        "for _ in range(t):\n",
        "    n=int(input())\n",
        "    set_a=set(map(int, input().split()))\n",
        "    m=int(input())\n",
        "    set_b=set(map(int, input().split()))\n",
        "\n",
        "    print(set_a.issubset(set_b))"
      ],
      "metadata": {
        "id": "pwdHzEEIOaiE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Check Strict Superset"
      ],
      "metadata": {
        "id": "c5ypl9hQOaWM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "set_a=set(map(int, input().split()))\n",
        "n=int(input())\n",
        "is_strict_superset=True\n",
        "\n",
        "for _ in range(n):\n",
        "    set_b=set(map(int, input().split()))\n",
        "\n",
        "    if not (set_a > set_b):\n",
        "        is_strict_superset=False\n",
        "        break\n",
        "\n",
        "print(is_strict_superset)\n"
      ],
      "metadata": {
        "id": "PpKDrVW5OZ50"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Collections**"
      ],
      "metadata": {
        "id": "dh-Aln11OW8S"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "collections.Counter()"
      ],
      "metadata": {
        "id": "xVYHilhHR7TA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter\n",
        "\n",
        "num_shoes=int(input().strip())\n",
        "shoe_sizes=list(map(int, input().strip().split()))\n",
        "num_customers=int(input().strip())\n",
        "\n",
        "available_sizes=Counter(shoe_sizes)\n",
        "\n",
        "total_earnings=0\n",
        "\n",
        "for _ in range(num_customers):\n",
        "    size,price = map(int, input().strip().split())\n",
        "    if available_sizes[size] > 0:\n",
        "        total_earnings+= price\n",
        "        available_sizes[size]-=1\n",
        "\n",
        "print(total_earnings)\n"
      ],
      "metadata": {
        "id": "Hl_gytuJR7Lw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "DefaultDict Tutorial"
      ],
      "metadata": {
        "id": "u1Im_yEyR7BK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import defaultdict\n",
        "\n",
        "d=defaultdict(list)\n",
        "n,m=map(int,input().split())\n",
        "for i in range(n):\n",
        "    w=input()\n",
        "    d[w].append(str(i+1))\n",
        "for _ in range(m):\n",
        "    w=input()\n",
        "    print(\" \".join(d[w])or -1)"
      ],
      "metadata": {
        "id": "DngRHgu7R659"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Collections.namedtuple()"
      ],
      "metadata": {
        "id": "Q72UE71vR6v1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import namedtuple\n",
        "\n",
        "n=int(input())\n",
        "cols=\",\".join(input().split())\n",
        "Student=namedtuple(\"Student\",cols)\n",
        "records=[]\n",
        "for _ in range(n):\n",
        "    entry=input().split()\n",
        "    student=Student._make(entry)\n",
        "    records.append(student)\n",
        "average=sum(float(s.MARKS)for s in records)/n\n",
        "print(f\"{average:.2f}\")\n"
      ],
      "metadata": {
        "id": "MlmD1Wg9R6pH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Collections.OrderedDict()"
      ],
      "metadata": {
        "id": "0GoPw98vR6g-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import OrderedDict\n",
        "\n",
        "n=int(input())\n",
        "order=OrderedDict()\n",
        "\n",
        "for _ in range(n):\n",
        "    line =input().strip().split()\n",
        "    item_name= ' '.join(line[:-1])\n",
        "    price = int(line[-1])\n",
        "\n",
        "    if item_name in order:\n",
        "        order[item_name]+=price\n",
        "    else:\n",
        "        order[item_name]=price\n",
        "\n",
        "for item_name, net_price in order.items():\n",
        "    print(item_name, net_price)"
      ],
      "metadata": {
        "id": "nky8gjUiR6Y8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Word Order"
      ],
      "metadata": {
        "id": "hd_1diVfR6QV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter, OrderedDict\n",
        "\n",
        "class OrderedCounter(Counter, OrderedDict):\n",
        "    pass\n",
        "\n",
        "words=[]\n",
        "n=int(input())\n",
        "for _ in range(n):\n",
        "    words.append(input().strip())\n",
        "word_count=OrderedCounter(words)\n",
        "print(len(word_count))\n",
        "for word in word_count:\n",
        "    print(word_count[word],end=\" \")"
      ],
      "metadata": {
        "id": "KlJaf0nIR6H9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Collections.deque()"
      ],
      "metadata": {
        "id": "92Z-6a_gR590"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import deque\n",
        "\n",
        "d=deque()\n",
        "n=int(input())\n",
        "\n",
        "for _ in range(n):\n",
        "    operation=input().strip().split()\n",
        "    method=operation[0]\n",
        "\n",
        "    if method== 'append':\n",
        "        d.append(int(operation[1]))\n",
        "    elif method== 'appendleft':\n",
        "        d.appendleft(int(operation[1]))\n",
        "    elif method== 'pop':\n",
        "        d.pop()\n",
        "    elif method== 'popleft':\n",
        "        d.popleft()\n",
        "\n",
        "print(\" \".join(map(str, d)))"
      ],
      "metadata": {
        "id": "mFIM9EvaR509"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Company Logo"
      ],
      "metadata": {
        "id": "dV5HCCnKR5q2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "from collections import Counter\n",
        "\n",
        "s = input().strip()\n",
        "char_count = Counter(s)\n",
        "sorted_chars = sorted(char_count.items(), key=lambda x: (-x[1], x[0]))\n",
        "top_three = sorted_chars[:3]\n",
        "\n",
        "for char, count in top_three:\n",
        "    print(f\"{char} {count}\")"
      ],
      "metadata": {
        "id": "faCuHofNR5jH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Piling Up!"
      ],
      "metadata": {
        "id": "Iq2dBsMkR5Y8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "test_cases=int(input())\n",
        "for _ in range(test_cases):\n",
        "    n=int(input())\n",
        "    cubes=list(map(int, input().split()))\n",
        "\n",
        "    left, right= 0, n - 1\n",
        "    last=float('inf')\n",
        "    possible=True\n",
        "\n",
        "    while left<= right:\n",
        "        if cubes[left] > last and cubes[right] > last:\n",
        "            possible= False\n",
        "            break\n",
        "\n",
        "        if cubes[left]<= last and (cubes[right] > last or cubes[left] >= cubes[right]):\n",
        "            last= cubes[left]\n",
        "            left+=1\n",
        "        else:\n",
        "            last = cubes[right]\n",
        "            right-=1\n",
        "\n",
        "    print(\"Yes\" if possible else \"No\")"
      ],
      "metadata": {
        "id": "nkAmkj52R5Pq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Date and Time**"
      ],
      "metadata": {
        "id": "Pw23uQA5R3xg"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Calendar Module"
      ],
      "metadata": {
        "id": "0L2aEH1dS6z4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import calendar\n",
        "input_date=input().strip()\n",
        "month,day,year=map(int,input_date.split())\n",
        "day_name=calendar.day_name[calendar.weekday(year,month,day)].upper()\n",
        "print(day_name)\n"
      ],
      "metadata": {
        "id": "Y075Kg7YS6qr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Time Delta"
      ],
      "metadata": {
        "id": "ijZILrIeS6gL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "import datetime\n",
        "\n",
        "\n",
        "def time_delta(t1, t2):\n",
        "    format_string = \"%a %d %b %Y %H:%M:%S %z\"\n",
        "    time1 = datetime.datetime.strptime(t1, format_string)\n",
        "    time2 = datetime.datetime.strptime(t2, format_string)\n",
        "    return str(int(abs((time1 - time2).total_seconds())))\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    fptr = open(os.environ['OUTPUT_PATH'], 'w')\n",
        "\n",
        "    t = int(input())\n",
        "\n",
        "    for t_itr in range(t):\n",
        "        t1 = input()\n",
        "\n",
        "        t2 = input()\n",
        "\n",
        "        delta = time_delta(t1, t2)\n",
        "\n",
        "        fptr.write(delta + '\\n')\n",
        "\n",
        "    fptr.close()"
      ],
      "metadata": {
        "id": "pwX2o4g0S6W8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Exceptions**"
      ],
      "metadata": {
        "id": "j1pltpyLS5K1"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Exceptions"
      ],
      "metadata": {
        "id": "bKSNWC7NS-rn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "tc = int(input())\n",
        "for i in range(tc):\n",
        "    a, b = input().split()\n",
        "\n",
        "    if b == '0':\n",
        "        print(\"Error Code: integer division or modulo by zero\")\n",
        "    elif not (a[0] == '-' and a[1:].isdigit()) and not a.isdigit():\n",
        "        print(\"Error Code: invalid literal for int() with base 10: '{}'\".format(a))\n",
        "    elif not (b[0] == '-' and b[1:].isdigit()) and not b.isdigit():\n",
        "        print(\"Error Code: invalid literal for int() with base 10: '{}'\".format(b))\n",
        "    else:\n",
        "        print(int(a) // int(b))"
      ],
      "metadata": {
        "id": "58oz3RmmTjHa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Built-ins**"
      ],
      "metadata": {
        "id": "bR8jTrC9S_Li"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Zipped!\n"
      ],
      "metadata": {
        "id": "vu6tpMnqTK3K"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "N,X=map(int,input().split())\n",
        "score_list=[]\n",
        "for _ in range(X):\n",
        "    score_list.append(list(map(float,input().split())))\n",
        "for i in zip(*score_list):\n",
        "    print(sum(i)/len(i))\n"
      ],
      "metadata": {
        "id": "x73CFA4STKu4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Athlete Sort\n"
      ],
      "metadata": {
        "id": "iEB7AyqwTKli"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "\n",
        "if __name__==\"__main__\":\n",
        "    n,m=map(int,input().split())\n",
        "    matrix=[]\n",
        "    for _ in range(n):\n",
        "        matrix.append(list(map(int,input().rstrip().split())))\n",
        "    k=int(input())\n",
        "    for row in sorted(matrix,key=lambda x:x[k]):\n",
        "        print(*row)"
      ],
      "metadata": {
        "id": "SiidSdk7TKcg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "ginortS"
      ],
      "metadata": {
        "id": "N_bp7pBbTKS_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "s=input()\n",
        "lower=[]\n",
        "upper=[]\n",
        "odd_digits=[]\n",
        "even_digits=[]\n",
        "\n",
        "for char in s:\n",
        "    if char.islower():\n",
        "        lower.append(char)\n",
        "    elif char.isupper():\n",
        "        upper.append(char)\n",
        "    elif char.isdigit():\n",
        "        if int(char) % 2 !=0:\n",
        "            odd_digits.append(char)\n",
        "        else:\n",
        "            even_digits.append(char)\n",
        "\n",
        "result = ''.join(sorted(lower) +sorted(upper) +sorted(odd_digits)+sorted(even_digits))\n",
        "print(result)"
      ],
      "metadata": {
        "id": "yb1sLFTvTKIs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Python Functionals**"
      ],
      "metadata": {
        "id": "uzvqBt2PTJn1"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Map and Lambda Function\n"
      ],
      "metadata": {
        "id": "q1iqjkhATRQW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "cube = lambda x: x ** 3\n",
        "\n",
        "def fibonacci(n):\n",
        "    fib = [0, 1]\n",
        "    for i in range(2, n):\n",
        "        fib.append(fib[-1] + fib[-2])\n",
        "    return fib[:n]\n"
      ],
      "metadata": {
        "id": "MrPj41LQTQxK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Regex and Parsing challenges**"
      ],
      "metadata": {
        "id": "hnsb7mS8TRj_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Detect Floating Point Number"
      ],
      "metadata": {
        "id": "KAmwXnj3UnlC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from re import compile\n",
        "\n",
        "pattern=compile(\"^[-+]?\\d*\\.\\d+$\")\n",
        "for _ in range(int(input())):\n",
        "    print(bool(pattern.match(input())))\n"
      ],
      "metadata": {
        "id": "RzJ7l-rXUnaM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Re.split()"
      ],
      "metadata": {
        "id": "Gi3lxiqVUnSL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "regex_pattern = r\"[,.]\"\t# Do not delete 'r'."
      ],
      "metadata": {
        "id": "AboOuU_sUnKn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Group(), Groups() & Groupdict()"
      ],
      "metadata": {
        "id": "PRFKCeyJUnB7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "\n",
        "s=input()\n",
        "result=re.search(r\"([A-Za-z0-9])\\1\",s)\n",
        "if result is None:\n",
        "    print(-1)\n",
        "else:\n",
        "    print(result[1])"
      ],
      "metadata": {
        "id": "mP0siY9fUm54"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Re.findall() & Re.finditer()"
      ],
      "metadata": {
        "id": "ytnx8UaSUmxH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "\n",
        "s=input()\n",
        "vowel_pairs=re.findall(\n",
        "    r\"(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([AEIOUaeiou]{2,})(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])\",\n",
        "    s,\n",
        ")\n",
        "if vowel_pairs:\n",
        "    for pair in vowel_pairs:\n",
        "        print(pair)\n",
        "else:\n",
        "    print(-1)\n"
      ],
      "metadata": {
        "id": "eRt-dx9uUmpw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Re.start() & Re.end()"
      ],
      "metadata": {
        "id": "gScylYANUmhn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "\n",
        "s = input().strip()\n",
        "k = input().strip()\n",
        "l= len(s)\n",
        "rd=False\n",
        "for i in range(l):\n",
        "    tr=re.match(k, s[i:])\n",
        "    if tr:\n",
        "        start_index= i+tr.start()\n",
        "        end_index=i+tr.end()- 1\n",
        "        print((start_index, end_index))\n",
        "        rd=True\n",
        "if rd==False:\n",
        "    print(\"(-1, -1)\")\n"
      ],
      "metadata": {
        "id": "M8RM2bcUUmaS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Regex Substitution"
      ],
      "metadata": {
        "id": "TlurtPodUmS5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "n=int(input())\n",
        "for _ in range(n):\n",
        "    line=input()\n",
        "    line=re.sub(r\"(?<= )&&(?= )\",\"and\",line)\n",
        "    line=re.sub(r\"(?<= )\\|\\|(?= )\",\"or\",line)\n",
        "    print(line)\n"
      ],
      "metadata": {
        "id": "UwfPkt6NUmLP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Validating Roman Numerals"
      ],
      "metadata": {
        "id": "qc0dD4JfUmEG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "regex_pattern = r\"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$\""
      ],
      "metadata": {
        "id": "zBr-isD8Ul7K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Validating phone numbers"
      ],
      "metadata": {
        "id": "SsmJavI5Ulyu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from re import compile, match\n",
        "\n",
        "n=int(input())\n",
        "for _ in range(n):\n",
        "    phone=input()\n",
        "    condition=compile(r\"^[7-9]\\d{9}$\")\n",
        "    if bool(match(condition, phone)):\n",
        "        print(\"YES\")\n",
        "    else:\n",
        "        print(\"NO\")"
      ],
      "metadata": {
        "id": "jlLWWI3YUlrM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Validating and Parsing Email Addresses"
      ],
      "metadata": {
        "id": "3CxkLPhgUli8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import email.utils\n",
        "import re\n",
        "\n",
        "n=int(input())\n",
        "for _ in range(n):\n",
        "    email_input=input()\n",
        "    parsed_email=email.utils.parseaddr(email_input)[1].strip()\n",
        "    is_valid=bool(\n",
        "        re.match(\n",
        "            r\"(^[A-Za-z][A-Za-z0-9\\._-]+)@([A-Za-z]+)\\.([A-Za-z]{1,3})$\", parsed_email\n",
        "        )\n",
        "    )\n",
        "    if is_valid:\n",
        "        print(email_input)"
      ],
      "metadata": {
        "id": "g5Kla6JLUlbF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Hex Color Code"
      ],
      "metadata": {
        "id": "cHN_-kzPUlQl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "\n",
        "n=int(input())\n",
        "for _ in range(n):\n",
        "    color_string=input()\n",
        "    matches=re.findall(r\"(#[0-9A-Fa-f]{3}|#[0-9A-Fa-f]{6})(?:[;,.)]{1})\", color_string)\n",
        "    for color in matches:\n",
        "        if color:\n",
        "            print(color)\n"
      ],
      "metadata": {
        "id": "Pju14Eh-UlJF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "HTML Parser - Part 1"
      ],
      "metadata": {
        "id": "V--DsP6JUlBE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from html.parser import HTMLParser\n",
        "\n",
        "class CustomHTMLParser(HTMLParser):\n",
        "    def handle_attr(self, attributes):\n",
        "        for attr_tuple in attributes:\n",
        "            print(\"->\", attr_tuple[0], \">\", attr_tuple[1])\n",
        "\n",
        "    def handle_starttag(self, tag, attributes):\n",
        "        print(\"Start :\", tag)\n",
        "        self.handle_attr(attributes)\n",
        "\n",
        "    def handle_endtag(self, tag):\n",
        "        print(\"End   :\", tag)\n",
        "\n",
        "    def handle_startendtag(self, tag, attributes):\n",
        "        print(\"Empty :\", tag)\n",
        "        self.handle_attr(attributes)\n",
        "parser=CustomHTMLParser()\n",
        "n=int(input())\n",
        "s=\"\".join(input() for _ in range(n))\n",
        "parser.feed(s)"
      ],
      "metadata": {
        "id": "LUweJZHUUk4_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "HTML Parser - Part 2"
      ],
      "metadata": {
        "id": "GUny-0BhUkvX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from html.parser import HTMLParser\n",
        "\n",
        "class CustomHTMLParser(HTMLParser):\n",
        "    def handle_comment(self, data):\n",
        "        line_count=len(data.split(\"\\n\"))\n",
        "        if line_count > 1:\n",
        "            print(\">>> Multi-line Comment\")\n",
        "        else:\n",
        "            print(\">>> Single-line Comment\")\n",
        "        if data.strip():\n",
        "            print(data)\n",
        "\n",
        "    def handle_data(self, data):\n",
        "        if data.strip():\n",
        "            print(\">>> Data\")\n",
        "            print(data)\n",
        "\n",
        "parser=CustomHTMLParser()\n",
        "\n",
        "n=int(input())\n",
        "html_string=\"\".join(input().rstrip()+\"\\n\" for _ in range(n))\n",
        "parser.feed(html_string)\n",
        "parser.close()\n"
      ],
      "metadata": {
        "id": "F6zSeSLgUkoT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Detect HTML Tags, Attributes and Attribute Values"
      ],
      "metadata": {
        "id": "VEJfq6opUkgE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from html.parser import HTMLParser\n",
        "\n",
        "class MyHTMLParser(HTMLParser):\n",
        "    def handle_starttag(self, tag, attrs):\n",
        "        print(tag)\n",
        "        for attr in attrs:\n",
        "            print(f\"-> {attr[0]} > {attr[1]}\")\n",
        "\n",
        "html = \"\"\n",
        "for _ in range(int(input())):\n",
        "    html+=input().rstrip() + '\\n'\n",
        "\n",
        "parser=MyHTMLParser()\n",
        "parser.feed(html)"
      ],
      "metadata": {
        "id": "rPfic-S4UkZP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Validating UID"
      ],
      "metadata": {
        "id": "-KrDyhmuUkRF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "\n",
        "n=int(input())\n",
        "upper_check=r\".*([A-Z].*){2,}\"\n",
        "digit_check=r\".*([0-9].*){3,}\"\n",
        "alphanumeric_and_length_check=r\"([A-Za-z0-9]){10}$\"\n",
        "repeat_check=r\".*(.).*\\1\"\n",
        "\n",
        "for _ in range(n):\n",
        "    uid=input().strip()\n",
        "    upper_check_result=bool(re.match(upper_check, uid))\n",
        "    digit_check_result=bool(re.match(digit_check, uid))\n",
        "    alphanumeric_and_length_check_result=bool(re.match(alphanumeric_and_length_check, uid))\n",
        "    repeat_check_result=bool(re.match(repeat_check, uid))\n",
        "\n",
        "    if (upper_check_result and digit_check_result and alphanumeric_and_length_check_result and not repeat_check_result):\n",
        "        print(\"Valid\")\n",
        "    else:\n",
        "        print(\"Invalid\")"
      ],
      "metadata": {
        "id": "jbjgr-0DUkJG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Validating Credit Card Numbers"
      ],
      "metadata": {
        "id": "j4ZzN6BeUkBU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "\n",
        "n=int(input())\n",
        "for _ in range(n):\n",
        "    credit=input().strip()\n",
        "    credit_no_hyphen=credit.replace(\"-\", \"\")\n",
        "    valid=True\n",
        "    length_16=bool(re.match(r\"^[4-6]\\d{15}$\", credit))\n",
        "    length_19=bool(re.match(r\"^[4-6]\\d{3}-\\d{4}-\\d{4}-\\d{4}$\", credit))\n",
        "    consecutive=bool(re.findall(r\"(?=(\\d)\\1\\1\\1)\", credit_no_hyphen))\n",
        "\n",
        "    if length_16 or length_19:\n",
        "        if consecutive:\n",
        "            valid=False\n",
        "    else:\n",
        "        valid=False\n",
        "\n",
        "    if valid:\n",
        "        print(\"Valid\")\n",
        "    else:\n",
        "        print(\"Invalid\")"
      ],
      "metadata": {
        "id": "tgItKfDdUj5_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Validating Postal Codes"
      ],
      "metadata": {
        "id": "zj4o1XCxUjza"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "regex_integer_in_range = r\"^(100000|[1-9]\\d{5}|[1-9]\\d{0,5}|0{6})$\" \t# Do not delete 'r'.\n",
        "regex_alternating_repetitive_digit_pair = r\"(?=(\\d)(?=\\d\\1))\"\t# Do not delete 'r'."
      ],
      "metadata": {
        "id": "Ve7s8QmiUjsH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Matrix Script"
      ],
      "metadata": {
        "id": "evl7xXjvUjkR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "\n",
        "n, m=map(int, input().split())\n",
        "char_array=[\"\"] * (n * m)\n",
        "\n",
        "for i in range(n):\n",
        "    line=input()\n",
        "    for j in range(m):\n",
        "        char_array[i + (j * n)]=line[j]\n",
        "\n",
        "decoded_string=\"\".join(char_array)\n",
        "final_decoded_string=re.sub(r\"(?<=[A-Za-z0-9])([ !@#$%&]+)(?=[A-Za-z0-9])\", \" \", decoded_string)\n",
        "\n",
        "print(final_decoded_string)"
      ],
      "metadata": {
        "id": "_5Gqi6DLUjZ8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **XML**"
      ],
      "metadata": {
        "id": "ezgJnxWWUcw4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "XML 1 - Find the Score"
      ],
      "metadata": {
        "id": "CVIbhB3oX0_b"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def get_attr_number(node):\n",
        " score=len(node.attrib)\n",
        " for child in node:\n",
        "  score+=get_attr_number(child)\n",
        " return score"
      ],
      "metadata": {
        "id": "H5mhRzsxX01_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "XML2 - Find the Maximum Depth"
      ],
      "metadata": {
        "id": "M_Td39-tX0nM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "maxdepth=0\n",
        "def depth(elem,level):\n",
        " global maxdepth\n",
        " level+=1\n",
        " if level>maxdepth:maxdepth=level\n",
        " for child in elem:\n",
        "  depth(child,level)"
      ],
      "metadata": {
        "id": "R98uUjuXX0ZA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Closures and Decorations**"
      ],
      "metadata": {
        "id": "_wgEYt0iYD7X"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Standardize Mobile Number Using Decorators"
      ],
      "metadata": {
        "id": "wwRfSTP_YDWK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def wrapper(f):\n",
        "    def fun(l):\n",
        "        formatted_numbers = ['+91 {} {}'.format(n[-10:-5], n[-5:]) for n in l]\n",
        "        return f(formatted_numbers)\n",
        "    return fun"
      ],
      "metadata": {
        "id": "dwchmjADYDN3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Decorators 2 - Name Directory"
      ],
      "metadata": {
        "id": "XkcC8ZL9YDFP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def person_lister(f):\n",
        "    def inner(people):\n",
        "        sorted_people = sorted(people, key=lambda x: int(x[2]))\n",
        "        return [f(person) for person in sorted_people]\n",
        "    return inner"
      ],
      "metadata": {
        "id": "mny_mJNSYXrl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Numpy**"
      ],
      "metadata": {
        "id": "tmF1DWykYCfd"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Arrays"
      ],
      "metadata": {
        "id": "LjPF8tZjYz11"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def arrays(arr):\n",
        "    np_array = numpy.array(arr, float)\n",
        "    return np_array[::-1]\n"
      ],
      "metadata": {
        "id": "j74QNBUyYzuD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Shape and Reshape"
      ],
      "metadata": {
        "id": "joLGTno_Yzk9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "arr=input().strip().split()\n",
        "arr=numpy.array(arr,int)\n",
        "arr=arr.reshape(3,3)\n",
        "print(arr)"
      ],
      "metadata": {
        "id": "j8MYBQywYzd7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Transpose and Flatten"
      ],
      "metadata": {
        "id": "jUtFMdzkYzVe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "n, m=map(int, input().split())\n",
        "array=[]\n",
        "\n",
        "for _ in range(n):\n",
        "    row=list(map(int, input().split()))\n",
        "    array.append(row)\n",
        "\n",
        "np_array=numpy.array(array)\n",
        "print(numpy.transpose(np_array))\n",
        "print(np_array.flatten())\n"
      ],
      "metadata": {
        "id": "8I_50r3CYzO4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Concatenate"
      ],
      "metadata": {
        "id": "HrqhiWkBYzDx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "n, m, p=map(int, input().split())\n",
        "\n",
        "array1=[]\n",
        "array2=[]\n",
        "\n",
        "for _ in range(n):\n",
        "    temp=list(map(int, input().split()))\n",
        "    array1.append(temp)\n",
        "\n",
        "for _ in range(m):\n",
        "    temp=list(map(int, input().split()))\n",
        "    array2.append(temp)\n",
        "\n",
        "np_array1=numpy.array(array1)\n",
        "np_array2=numpy.array(array2)\n",
        "\n",
        "print(numpy.concatenate((np_array1, np_array2), axis=0))"
      ],
      "metadata": {
        "id": "n6_2pzOaYy9B"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Zeros and Ones"
      ],
      "metadata": {
        "id": "HBXqocHMYyzv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "shape=tuple(map(int,input().split()))\n",
        "\n",
        "print(numpy.zeros(shape,dtype=int))\n",
        "print(numpy.ones(shape,dtype=int))\n"
      ],
      "metadata": {
        "id": "OTK0OE2BYysJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Eye and Identity"
      ],
      "metadata": {
        "id": "BMH0BWIwYyiS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "numpy.set_printoptions(legacy='1.13')\n",
        "\n",
        "n, m = map(int, input().split())\n",
        "print(numpy.eye(n, m))\n",
        "\n"
      ],
      "metadata": {
        "id": "0iku_FQZYyb5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Array Mathematics"
      ],
      "metadata": {
        "id": "A96yrEQYYySk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "n, m=map(int, input().split())\n",
        "array1=[]\n",
        "array2=[]\n",
        "\n",
        "for _ in range(n):\n",
        "    temp=list(map(int, input().split()))\n",
        "    array1.append(temp)\n",
        "\n",
        "for _ in range(n):\n",
        "    temp=list(map(int, input().split()))\n",
        "    array2.append(temp)\n",
        "\n",
        "np_array1=numpy.array(array1)\n",
        "np_array2=numpy.array(array2)\n",
        "\n",
        "print(np_array1 + np_array2)\n",
        "print(np_array1 - np_array2)\n",
        "print(np_array1 * np_array2)\n",
        "print(np_array1 // np_array2)\n",
        "print(np_array1 % np_array2)\n",
        "print(np_array1**np_array2)\n",
        "\n"
      ],
      "metadata": {
        "id": "AAl7qQw7YyLF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Floor, Ceil and Rint"
      ],
      "metadata": {
        "id": "yxsyzYRPYyB3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "numpy.set_printoptions(legacy='1.13')\n",
        "\n",
        "a=numpy.array(input().split(),float)\n",
        "print(numpy.floor(a))\n",
        "print(numpy.ceil(a))\n",
        "print(numpy.rint(a))\n"
      ],
      "metadata": {
        "id": "Vt5b_jlGYx58"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Sum and Prod"
      ],
      "metadata": {
        "id": "8JY8CYprYxxN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "n, m=map(int, input().split())\n",
        "array=[]\n",
        "\n",
        "for _ in range(n):\n",
        "    t=list(map(int, input().split()))\n",
        "    array.append(t)\n",
        "\n",
        "np_array=numpy.array(array)\n",
        "print(numpy.max(numpy.min(np_array, axis=1)))\n"
      ],
      "metadata": {
        "id": "pzcNZRyBYxpV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Min and Max"
      ],
      "metadata": {
        "id": "6JGUuQBlYxfk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "\n",
        "n,m=map(int,input().split())\n",
        "a=numpy.array([input().split() for _ in range(n)],int)\n",
        "min_values=numpy.min(a,axis=1)\n",
        "print(numpy.max(min_values))\n"
      ],
      "metadata": {
        "id": "3OQkwLEZYxWM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Mean, Var, and Std"
      ],
      "metadata": {
        "id": "OtwIP-zLYxL8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "order=input().split(\" \")\n",
        "arr=[]\n",
        "for i in range(int(order[0])):\n",
        "    row=list(map(int,input().split(\" \")))\n",
        "    arr.append(row)\n",
        "Array=numpy.array(arr)\n",
        "\n",
        "print(numpy.mean(Array,axis=1))\n",
        "print(numpy.var(Array,axis=0))\n",
        "print(round(numpy.std(Array,axis=None),11))\n"
      ],
      "metadata": {
        "id": "C2F_XZEwYxEW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Dot and Cross"
      ],
      "metadata": {
        "id": "PnBI3kkoYw6S"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "\n",
        "n=int(input())\n",
        "A=[]\n",
        "for _ in range(n):\n",
        "    row=list(map(int,input().split(\" \")))\n",
        "    A.append(row)\n",
        "B=[]\n",
        "for _ in range(n):\n",
        "    row=list(map(int,input().split(\" \")))\n",
        "    B.append(row)\n",
        "\n",
        "MatrixA=numpy.array(A)\n",
        "MatrixB=numpy.array(B)\n",
        "\n",
        "print(numpy.dot(MatrixA,MatrixB))\n"
      ],
      "metadata": {
        "id": "pMMjSTkrYwzL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Inner and Outer"
      ],
      "metadata": {
        "id": "pgC3IQZdYwoO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "\n",
        "A=numpy.array(list(map(int,input().split())))\n",
        "B=numpy.array(list(map(int,input().split())))\n",
        "\n",
        "print(numpy.inner(A,B))\n",
        "print(numpy.outer(A,B))\n"
      ],
      "metadata": {
        "id": "qqYTqqN-Ywg7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Polynomials"
      ],
      "metadata": {
        "id": "P27DldabYwWp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "\n",
        "coefficients=list(map(float,input().split()))\n",
        "x=float(input())\n",
        "\n",
        "print(numpy.polyval(coefficients,x))\n"
      ],
      "metadata": {
        "id": "YnAoqWZKYwPi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Linear Algebra"
      ],
      "metadata": {
        "id": "6fgd7_n4YwFV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy\n",
        "\n",
        "\n",
        "n=int(input())\n",
        "matrix=[]\n",
        "for _ in range(n):\n",
        "    row=list(map(float,input().split()))\n",
        "    matrix.append(row)\n",
        "\n",
        "determinant=numpy.linalg.det(matrix)\n",
        "print(round(determinant,2))\n"
      ],
      "metadata": {
        "id": "1L8GUe2XYv7W"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# ***`Problem 2`***"
      ],
      "metadata": {
        "id": "i-bwssPH8eFI"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Birthday Cake Candles"
      ],
      "metadata": {
        "id": "-LVQTzK380_1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "#\n",
        "# Complete the 'birthdayCakeCandles' function below.\n",
        "#\n",
        "# The function is expected to return an INTEGER.\n",
        "# The function accepts INTEGER_ARRAY candles as parameter.\n",
        "#\n",
        "\n",
        "def birthdayCakeCandles(candles):\n",
        "    M = max(candles)\n",
        "    return candles.count(M)\n",
        "\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    fptr = open(os.environ['OUTPUT_PATH'], 'w')\n",
        "\n",
        "    candles_count = int(input().strip())\n",
        "\n",
        "    candles = list(map(int, input().rstrip().split()))\n",
        "\n",
        "    result = birthdayCakeCandles(candles)\n",
        "\n",
        "    fptr.write(str(result) + '\\n')\n",
        "\n",
        "    fptr.close()"
      ],
      "metadata": {
        "id": "U0-XWDCT8lao"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Number Line Jumps"
      ],
      "metadata": {
        "id": "kk_l_Jyl81_3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "#\n",
        "# Complete the 'kangaroo' function below.\n",
        "#\n",
        "# The function is expected to return a STRING.\n",
        "# The function accepts following parameters:\n",
        "#  1. INTEGER x1\n",
        "#  2. INTEGER v1\n",
        "#  3. INTEGER x2\n",
        "#  4. INTEGER v2\n",
        "#\n",
        "\n",
        "def kangaroo(x1, v1, x2, v2):\n",
        "    for i in range (1,10001):\n",
        "        if x1+v1*i==x2+v2*i:\n",
        "            return 'YES'\n",
        "    return 'NO'\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    fptr = open(os.environ['OUTPUT_PATH'], 'w')\n",
        "\n",
        "    first_multiple_input = input().rstrip().split()\n",
        "\n",
        "    x1 = int(first_multiple_input[0])\n",
        "\n",
        "    v1 = int(first_multiple_input[1])\n",
        "\n",
        "    x2 = int(first_multiple_input[2])\n",
        "\n",
        "    v2 = int(first_multiple_input[3])\n",
        "\n",
        "    result = kangaroo(x1, v1, x2, v2)\n",
        "\n",
        "    fptr.write(result + '\\n')\n",
        "\n",
        "    fptr.close()\n"
      ],
      "metadata": {
        "id": "ECZeG8EB8zhr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Viral Advertising"
      ],
      "metadata": {
        "id": "df_K9kSZ844E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "#\n",
        "# Complete the 'viralAdvertising' function below.\n",
        "#\n",
        "# The function is expected to return an INTEGER.\n",
        "# The function accepts INTEGER n as parameter.\n",
        "#\n",
        "\n",
        "def viralAdvertising(n):\n",
        "    shared = 5\n",
        "    cumulative_likes = 0\n",
        "\n",
        "    for _ in range(n):\n",
        "        liked = shared//2\n",
        "        cumulative_likes+=liked\n",
        "        shared = liked *3\n",
        "\n",
        "    return cumulative_likes\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    fptr = open(os.environ['OUTPUT_PATH'], 'w')\n",
        "\n",
        "    n = int(input().strip())\n",
        "\n",
        "    result = viralAdvertising(n)\n",
        "\n",
        "    fptr.write(str(result) + '\\n')\n",
        "\n",
        "    fptr.close()"
      ],
      "metadata": {
        "id": "DrnNFeTA84vM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Recursive Digit Sum"
      ],
      "metadata": {
        "id": "OReVFpR883p6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "#\n",
        "# Complete the 'superDigit' function below.\n",
        "#\n",
        "# The function is expected to return an INTEGER.\n",
        "# The function accepts following parameters:\n",
        "#  1. STRING n\n",
        "#  2. INTEGER k\n",
        "#\n",
        "\n",
        "def superDigit(n, k):\n",
        "    s=0\n",
        "    for char in n:\n",
        "        s+=int(char)\n",
        "    s=s*k\n",
        "    while s>=10:\n",
        "        current_sum=0\n",
        "        for digit in str(s):\n",
        "            current_sum+=int(digit)\n",
        "        s=current_sum\n",
        "\n",
        "    return s\n",
        "\n",
        "\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    fptr = open(os.environ['OUTPUT_PATH'], 'w')\n",
        "\n",
        "    first_multiple_input = input().rstrip().split()\n",
        "\n",
        "    n = first_multiple_input[0]\n",
        "\n",
        "    k = int(first_multiple_input[1])\n",
        "\n",
        "    result = superDigit(n, k)\n",
        "\n",
        "    fptr.write(str(result) + '\\n')\n",
        "\n",
        "    fptr.close()"
      ],
      "metadata": {
        "id": "ts22z72c83go"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Insertion Sort - Part 1"
      ],
      "metadata": {
        "id": "tZLg5hHS83W7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "#\n",
        "# Complete the 'insertionSort1' function below.\n",
        "#\n",
        "# The function accepts following parameters:\n",
        "#  1. INTEGER n\n",
        "#  2. INTEGER_ARRAY arr\n",
        "#\n",
        "\n",
        "def insertionSort1(n, arr):\n",
        "    m=arr[n-1]\n",
        "\n",
        "    for i in range(n-1):\n",
        "        if arr[n-2-i]> m:\n",
        "            arr[n-1-i]=arr[n-2-i]\n",
        "            print(*arr)\n",
        "        else:\n",
        "            arr[n-1-i]=m\n",
        "            print(*arr)\n",
        "            return\n",
        "\n",
        "    arr[0] =m\n",
        "    print(*arr)\n",
        "\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    n = int(input().strip())\n",
        "\n",
        "    arr = list(map(int, input().rstrip().split()))\n",
        "\n",
        "    insertionSort1(n, arr)"
      ],
      "metadata": {
        "id": "dpnwe2MN83LW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Insertion Sort - Part 2"
      ],
      "metadata": {
        "id": "nIojlQpv82yq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!/bin/python3\n",
        "\n",
        "import math\n",
        "import os\n",
        "import random\n",
        "import re\n",
        "import sys\n",
        "\n",
        "#\n",
        "# Complete the 'insertionSort2' function below.\n",
        "#\n",
        "# The function accepts following parameters:\n",
        "#  1. INTEGER n\n",
        "#  2. INTEGER_ARRAY arr\n",
        "#\n",
        "\n",
        "def insertionSort2(n, arr):\n",
        "     for i in range(1,n):\n",
        "        key=arr[i]\n",
        "        j=i -1\n",
        "        while j>=0 and arr[j]> key:\n",
        "            arr[j +1]=arr[j]\n",
        "            j-=1\n",
        "\n",
        "        arr[j+1]=key\n",
        "        print(*arr)\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    n = int(input().strip())\n",
        "\n",
        "    arr = list(map(int, input().rstrip().split()))\n",
        "\n",
        "    insertionSort2(n, arr)"
      ],
      "metadata": {
        "id": "Oigrjt1o9OHe"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}