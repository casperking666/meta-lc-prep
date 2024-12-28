class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        # if len(word) == int(abbr):
        #     return True

        wPtr, aPtr = 0, 0
        while aPtr < len(abbr):
            num = -1
            if abbr[aPtr].isdigit():
                if abbr[aPtr] == "1" and aPtr + 1 < len(abbr) and abbr[aPtr+1].isdigit():
                    if aPtr + 2 < len(abbr) and abbr[aPtr+2].isdigit():
                        num = int(abbr[aPtr:aPtr+3])
                        aPtr += 2
                    else:
                        num = int(abbr[aPtr:aPtr+2])
                        aPtr += 1
                elif abbr[aPtr] == '0':
                    return False
                else:
                    num = int(abbr[aPtr])
            else:
                if wPtr >= len(word) or word[wPtr] != abbr[aPtr]:
                    return False
            
            print(num)
            print(f"{wPtr}\n")

            if num == -1:
                wPtr += 1
            elif num != -1 and num <= len(word) - wPtr:
                wPtr += num
            else:
                return False
            aPtr += 1

        print(wPtr)
        # “hi” / "1" output true, expected false
        return True if wPtr == len(word) else False
        