# 2
def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        p1 = p2 = 0
        while p1 < len(word) and p2 < len(abbr):
            if abbr[p2].isdigit():
                if abbr[p2] == '0': # leading zeros are invalid
                    return False
                shift = 0
                while p2 < len(abbr) and abbr[p2].isdigit():
                    shift = (shift*10)+int(abbr[p2])
                    p2 += 1
                p1 += shift
            else:
                if word[p1] != abbr[p2]:
                    return False
                p1 += 1
                p2 += 1
        return p1 == len(word) and p2 == len(abbr)
# 有p1走完但p2没走完的情况吗 (有比如说p2有个很大的数字，p1直接超过length)

# 3
def validWordAbbreviation(self, word: str, abbr: str) -> bool:
    i, j = 0, 0
    while i < len(word) and j < len(abbr):
        if abbr[j].isdigit():
            cur = 0
            while j < len(abbr) and abbr[j].isdigit():
                if cur == 0 and abbr[j] == '0':
                    return False
                cur = cur * 10 + int(abbr[j])
                j += 1
            i += cur
        else:
            if word[i] != abbr[j]:
                return False
            i += 1
            j += 1
    return i == len(word) and j == len(abbr)


# 1
class Solution:
    def validWordAbbreviation(self, word, abbr):
        i, j, m, prev = len(word), len(abbr), 1, None
        
        while i > 0 and j > 0:
            c1, c2 = word[i-1], abbr[j-1]
            if c1 == c2:
                i -= 1
                j -= 1
                m = 1
                if prev == 0: return False
            elif c2.isnumeric():
                i -= int(c2)*m
                j -= 1
                m *= 10
                prev = int(c2)
            else: return False
        
        return i == j == 0