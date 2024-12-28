from collections import Counter
from collections import deque
import heapq


# questions I can't solve in second attempt 162, 50

# binary search insight:
# The distinction comes down to what we're doing at mid:
# When high = mid:
# The midpoint (mid) is still a valid candidate and is included in the next iteration's search space. We're not eliminating mid, just narrowing the range.
# When high = mid - 1:
# The midpoint (mid) is excluded from further consideration, as we know it cannot be the solution.


# 1249. Minimum Remove to Make Valid Parentheses
def minRemoveToMakeValid(self, s: str) -> str:
    s = list(s)
    stack = []
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                s[i] = ''
    while stack:
        s[stack.pop()] = ''
    return ''.join(s)


# 314. Binary Tree Vertical Order Traversal
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = deque([(root, 0)])
        dic = defaultdict(list)
        maxCol = minCol = 0

        while queue:
            for i in range(len(queue)):
                node, col = queue.popleft()
                dic[col].append(node.val)
                if node.left:
                    queue.append((node.left, col - 1))
                    minCol = min(minCol, col - 1)
                if node.right:
                    queue.append((node.right, col + 1))
                    maxCol = max(maxCol, col + 1)
        return [dic[key] for key in range(minCol, maxCol + 1)]

# 227. Basic Calculator II
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        pre_op = "+"
        pre_val = 0
        s += "+"
        for char in s:
            if char == " ":
                continue
            elif char.isdigit():
                pre_val = pre_val * 10 + int(char)
            elif char in "+-*/":
                if pre_op == "+":
                    stack.append(int(pre_val))
                elif pre_op == "-":
                    stack.append(-int(pre_val))
                elif pre_op == "*":
                    val = stack.pop()
                    stack.append(val * int(pre_val))
                else:
                    val = stack.pop()
                    stack.append(int(val / int(pre_val)))
                pre_op = char
                pre_val = 0

        return sum(stack)            


# 680. Valid Palindrome II
# two pointers without recursion
class Solution:
    def validPalindrome(self, s: str) -> bool:
        # s = list(s)
        def checkPalindrome(s, i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                else:
                    i += 1
                    j -= 1
            return True
        
        start, end = 0, len(s) - 1
        while start < end:
            if s[start] != s[end]:
                return checkPalindrome(s, start+1, end) or checkPalindrome(s, start, end-1)
            start += 1
            end -= 1
        return True
                
# two pointers with recursion can be generalized with k deletions
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def helper(i, j, k):
            while i < j:
                if s[i] != s[j]:
                    if k == 0:
                        return False
                    else:
                        return helper(i+1, j, k-1) or helper(i, j-1, k-1)
                i += 1
                j -= 1
            return True
        return helper(0, len(s)-1, 1)

# 88. Merge Sorted Array
# my while solution is a bit cumbersome, for one is probably a bit better
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        ptr1 = m - 1
        ptr2 = n - 1
        end = len(nums1) - 1
        while ptr1 >= 0 and ptr2 >= 0:
            if nums1[ptr1] > nums2[ptr2]:
                nums1[end] = nums1[ptr1]
                ptr1 -= 1
            else:
                nums1[end] = nums2[ptr2]
                ptr2 -= 1
            end -= 1
        
        while ptr2 >= 0:
            nums1[end] = nums2[ptr2]
            ptr2 -= 1
            end -= 1

# better solution but a bit harder to reason about
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        ptr1 = m - 1
        ptr2 = n - 1
        for end in range(m + n - 1, -1, -1):
            if ptr2 < 0:
                break
            elif ptr1 >= 0 and nums1[ptr1] > nums2[ptr2]:
                nums1[end] = nums1[ptr1]
                ptr1 -= 1
            else:
                nums1[end] = nums2[ptr2]
                ptr2 -= 1

            
# 215. Kth Largest Element in an Array
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quickSelect(nums, k):
            left, mid, right = [], [], []
            pivot = random.choice(nums)

            for num in nums:
                if num > pivot:
                    left.append(num)
                elif num < pivot:
                    right.append(num)
                else:
                    mid.append(num)

            if len(left) >= k:
                return quickSelect(left, k)
            elif len(left) + len(mid) < k:
                return quickSelect(right, k - len(left) - len(mid))
            
            return pivot
        return quickSelect(nums, k)


# 1650. Lowest Common Ancestor of a Binary Tree III
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        def findDepth(node):
            depth = 0
            while node:
                node = node.parent
                depth += 1
            return depth
        
        # alternative
        def findDepth(node):
            if node is None:
                return 0
            return findDepth(node.parent) + 1
        
        pDepth = findDepth(p)
        qDepth = findDepth(q)

        while pDepth > qDepth:
            p = p.parent
            pDepth -= 1
        
        while qDepth > pDepth:
            q = q.parent
            qDepth -= 1
        
        while p != q:
            p = p.parent
            q = q.parent
        return p

# my ugly solution...
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        def findDepth(node, depth):
            if not node.parent:
                return depth
            else:
                return findDepth(node.parent, depth + 1)
        
        pDepth = findDepth(p, 0)
        qDepth = findDepth(q, 0)

        i = 0
        if pDepth > qDepth:
            while pDepth - qDepth - i> 0:
                p = p.parent
                i += 1
        else:
            while qDepth - pDepth - i> 0:
                q = q.parent
                i += 1
        
        if p == q:
            return q if pDepth > qDepth else p
        else:
            while p.parent:
                p = p.parent
                q = q.parent
                if p == q:
                    return q if pDepth > qDepth else p

# 408. Valid Word Abbreviation
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

# 339. Nested List Weight Sum
# the important thing to remember is for all the examples the [] are actual lists representation (not NI), and one NI can either have a integer or list not both
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        queue = deque(nestedList)
        depth = 1
        res = 0
        while queue:
            for i in range(len(queue)):
                nl = queue.popleft()
                if nl.isInteger():
                    res += depth * nl.getInteger()
                else:
                    queue.extend(nl.getList())
            depth += 1
        return res

# 528. Random Pick with Weight
# dumb fuck question, remember to use prefix sum and random.random() which is [0,1)
class Solution:
    # [1,2,3]
    # [1,3,6]
    def __init__(self, w: List[int]):
        self.prefixSum = []
        self.sum = 0
        for num in w:
            self.sum += num
            self.prefixSum.append(self.sum)
        
    def pickIndex(self) -> int:
        target = self.sum * random.random()
        start, end = 0, len(self.prefixSum) - 1
        m = 0
        while start < end:
            m = (start + end) // 2
            if self.prefixSum[m] > target: # target = 2
                end = m
            else:
                start = m + 1
        return start


# 50. Pow(x, n)
class Solution:
    # 2^10 -> 2*2^5
    # 2^5 -> 2*(2*2)^2
    def myPow(self, x: float, n: int) -> float:
        
        def recurse(x, n):
            if n == 0:
                return 1
            if n % 2 == 0:
                return recurse(x * x, n // 2)
            else:
                return x * recurse(x * x, (n - 1) // 2)
        
        res = recurse(x, abs(n))
        return res if n > 0 else 1 / res

# 938. Range Sum of BST
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        res = 0
        def dfs(node, low, high) -> None:
            if node is None:
                return
            x = node.val
            # these two if took me ages
            if x > low:
                dfs(node.left, low, high)
            if x < high:
                dfs(node.right, low, high)
            nonlocal res
            if low <= node.val <= high:
                res += node.val
        dfs(root, low, high)
        return res

# 973. K Closest Points to Origin
class Solution:
    # -1, -2, -3,-4,-5
    # k = 2
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for i, point in enumerate(points):
            dis = sqrt(point[0] ** 2 + point[1] ** 2)
            heapq.heappush(heap, (-dis, i))
            if len(heap) > k:
                heapq.heappop(heap)

        return [points[i] for _, i in heap]


# 1091
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        deque = collections.deque([(0, 0)])
        n = len(grid) -1

        neighbors = [(-1,-1), (0, -1), (1, -1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
        def get_neighbours(row, col):
            for row_difference, col_difference in neighbors:
                new_row = row + row_difference
                new_col = col + col_difference
                if not(0 <= new_row <= n and 0 <= new_col <= n):
                    continue
                if grid[new_row][new_col] != 0:
                    continue
                yield (new_row, new_col)

        if grid[0][0] == 1 or grid[n][n] == 1:
            return -1
        grid[0][0] = 1
        while deque:
            for i in range(len(deque)):
                x, y = deque.popleft()
                distance = grid[x][y]
                # neighbors = 
                for ni, nj in get_neighbours(x, y):
                    grid[ni][nj] = distance + 1
                    deque.append((ni, nj))
        return -1 if grid[n][n] == 0 else grid[n][n]

# 1570. Dot Product of Two Sparse Vectors
class SparseVector:
    def __init__(self, nums: List[int]):
        self.dic = {}
        for i, num in enumerate(nums):
            if num != 0:
                self.dic[i] = num
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        res = 0
        for k, v in vec.dic.items():
            if k in self.dic.keys() :
                res += v * self.dic[k]
        return res

# 162
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        l = 0
        r = len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] > nums[mid + 1]:
                r = mid
            else:
                l = mid + 1

        print(l, r)
        return l

# 199. Binary Tree Right Side View
# bfs
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            for i in range(len(queue)):
                node = queue.popleft()
                if i == 0:
                    res.append(node.val)
                if node.right:
                    queue.append(node.right)
                if node.left:
                    queue.append(node.left)
        return res

# dfs
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        def dfs(node, cnt) -> None:
            if node is None:
                return
            if len(ans) == cnt:
                ans.append(node.val)
            dfs(node.right, cnt+1)
            dfs(node.left, cnt+1)
            
        dfs(root, 0)
        return ans

# 791
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        s_map = Counter(s)
        res = ""
        for char in order:
            if char in s_map:
                times = s_map[char]
                res += char * times
                s_map[char] -= times
        
        for k, v in s_map.items():
            for i in range(v):
                res += k
        return 
    
# 560
# note the if statement, that should go first as we check the last state
# instead of the new state, break case was nums=[1] k=0
class Solution:
    # [1,2,3]
    # [0,1,3,6]
    # if sum[i] - sum[j] == k
    def subarraySum(self, nums: List[int], k: int) -> int:
        dic = defaultdict(int) # {0:1,1:1,3:1,6:1}
        dic[0] = 1
        s = 0
        count = 0 
        for num in nums: # 1, 2, 3
            s += num # 1, 3, 6
            diff = s - k # -2, 0, 3
            if diff in dic:
                count += dic[diff] # 2
            dic[s] += 1 
        return count

# 71. Simplify Path
# three things to note, first split /hime/ gives ["","home",""]
# secondly, "if stack" needs to be separate, thirdly we need "/" at the start
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        path = path.split("/")

        for string in path:
            if string == "" or string == ".":
                continue
            elif string == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(string)
        return "/" + "/".join(stack)


# 426
# inorder dfs
# logic: last.right = cur
#         cur.left  = last



# 138
# key thing is to add head, new node mapping to dict before loop
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def __init__(self):
        self.visited = {}

    def copyNode(self, node):
        if not node:
            return None
        if node in self.visited:
            return self.visited[node]
        else:
            newNode = Node(node.val)
            self.visited[node] = newNode
            return newNode

    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        
        dummy_head = head
        new_node = Node(head.val)
        self.visited[head] = new_node

        while head:
            new_node.next = self.copyNode(head.next)
            new_node.random = self.copyNode(head.random)

            head = head.next
            new_node = new_node.next

        return self.visited[dummy_head]

# 215
# does nums have duplicate numbers
class Solution:
    # [3,2,1,5,6,4]
    # [5,6] [4,4] [3,2,1]
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(nums, k):
            left, mid, right = [], [], []
            pivot = random.choice(nums)
            for num in nums:
                if num > pivot:
                    left.append(num)
                elif num == pivot:
                    mid.append(num)
                else:
                    right.append(num)
            
            if len(left) >= k:
                return partition(left, k)
            elif len(left) + len(mid) < k:
                return partition(right, k - len(left) - len(mid))
            return pivot
        return partition(nums, k)

# 146 LRU Cache
# things to notes are
# 1. dict[key:node]
# 2. need helper add and remove func where add() adds to the back
# 3. dll needs both key and value (put needs to delete old key from dict)
# 4. its like a queue so FIFO
# 5. for put, check if in dict, then add then check cap
class ListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dic = {}
        self.head = ListNode(-1, -1)
        self.tail = ListNode(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1

        node = self.dic[key]
        self.remove(node)
        self.add(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            old_node = self.dic[key]
            self.remove(old_node)

        node = ListNode(key, value)
        self.dic[key] = node
        self.add(node)

        if len(self.dic) > self.capacity:
            node_to_delete = self.head.next
            self.remove(node_to_delete)
            del self.dic[node_to_delete.key]

    def add(self, node):
        previous_end = self.tail.prev
        previous_end.next = node
        node.prev = previous_end
        node.next = self.tail
        self.tail.prev = node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# 56 Merge Intervals
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x:x[0])

        res = []

        for interval in intervals:
            if len(res) == 0 or interval[0] > res[-1][1]:
                res.append(interval)
            else:
                res[-1][1] = max(interval[1], res[-1][1])
        return res
    
# 1004 Max consecutive ones III
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        for right in range(len(nums)):
            if nums[right] == 0:
                k -= 1
            
            if k < 0:
                if nums[left] == 0:
                    k += 1
                left += 1
        return right - left + 1
            
# 1762 Buildings With an Ocean View
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        max_num = -1
        n = len(heights)
        res = []
        for i in range(n - 1, -1, -1):
            height = heights[i]
            if height > max_num:
                res.append(i)
                max_num = height
        return res[::-1]
    

# 346. Moving Average from Data Stream
class MovingAverage:

    def __init__(self, size: int):
        self.queue = deque([])
        self.size = size
        self.sum = 0

    def next(self, val: int) -> float:
        self.queue.append(val)
        self.sum += val
        if len(self.queue) > self.size:
            self.sum -= self.queue.popleft()
        return self.sum / len(self.queue)
        


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)


# 347. Top K Frequent Elements
# heap solution
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)  # O(N)
        heap = []  # min-heap
        
        for num, freq in count.items():
            heapq.heappush(heap, (freq, num)) 
            if len(heap) > k:
                heapq.heappop(heap) 
        
        result = [num for freq, num in heap]
        return result

# for partition, remember to use left and right instead of 0, -1
# also remember to change pivot position after for loop
# for lenght, we want length of the keys
class Solution:
    #{1:3,2:2,3:1}
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        def partition(left, right, pivot_index):
            pivot = keys[pivot_index]
            pivot_freq = counter[pivot]
            index = left

            keys[pivot_index], keys[right] = keys[right], keys[pivot_index]

            for key in range(left, right):
                if counter[keys[key]] < pivot_freq:
                    keys[index], keys[key] = keys[key], keys[index]
                    index += 1

            keys[right], keys[index] = keys[index], keys[right]
            return index
        
        def quickselect(left, right, k_smallest):
            if left == right:
                return
            
            rand_idx = random.randint(left, right)
            pivot_index = partition(left, right, rand_idx)

            if pivot_index == k_smallest:
                return
            elif pivot_index > k_smallest:
                return quickselect(left, pivot_index-1, k_smallest)
            else:
                return quickselect(pivot_index+1, right, k_smallest)

        counter = Counter(nums)
        keys = list(counter.keys())
        n = len(keys)
        quickselect(0, n-1, n-k)
        return keys[n-k:]
        

# 986. Interval List Intersections
# one thing i make mistakes of is lo < hi, should be lo <= hi
# my init version
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        n = len(firstList)
        m = len(secondList)
        i = 0
        j = 0
        res = []
        while i < n and j < m:
            if firstList[i][1] < secondList[j][0]:
                i += 1
            elif firstList[i][0] > secondList[j][1]:
                j += 1
            else:
                interval = []
                if firstList[i][0] < secondList[j][0]:
                    interval.append(secondList[j][0])
                else:
                    interval.append(firstList[i][0])

                if firstList[i][1] < secondList[j][1]:
                    interval.append(firstList[i][1])
                    i += 1
                else:
                    interval.append(secondList[j][1])
                    j += 1
                res.append(interval)
        return res

# optimized version
class Solution:
    # ------ -----      ----
    #   ----   ------    ----- 
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        n = len(firstList)
        m = len(secondList)
        i, j = 0, 0
        res = []
        while i < n and j < m:
            low = max(firstList[i][0], secondList[j][0])
            high = min(firstList[i][1], secondList[j][1])

            if low <= high:
                res.append([low, high])
            
            if firstList[i][1] < secondList[j][1]:
                i += 1
            else:
                j += 1
        return res

# 125. Valid Palindrome
# isalnum() for checking if its a digit or char
# lower()
        
# 543. Diameter of Binary Tree
class Solution:
    # leaf to leaf
    # max(r.left + r.right, max_v)
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_d = 0
        def recurse(node) -> int:
            if node is None:
                return 0
            l_len = recurse(node.left)
            r_len = recurse(node.right)

            nonlocal max_d
            max_d = max(l_len + r_len, max_d)
            return max(l_len, r_len) + 1
        recurse(root)
        return max_d



# 23. Merge k Sorted Lists
# the important thing to remember is to do curr.next = ListNode(val)

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        for i, node in enumerate(lists):
            if lists[i]:
                heapq.heappush(heap, (node.val, i))
                lists[i] = lists[i].next # can't use node here
        # [1,2,4]
        dummyHead = head = ListNode()

        while heap:
            val, i = heapq.heappop(heap)
            head.next = ListNode(val) # 1
            head = head.next
            if lists[i] != None:
                heapq.heappush(heap, (lists[i].val, i))
                lists[i] = lists[i].next
        
        return dummyHead.next
        

# 863. All Nodes Distance K in Binary Tree
# i would use dfs for both creating parents map and traverse the tree
# as its the easiest, tho bfs for travering seems more intuitive
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # {node.val : pNode}
        pNodeMap = {}
        def getParent(node, parentNode):
            if node == None:
                return
            else:
                pNodeMap[node.val] = parentNode
                getParent(node.left, node)
                getParent(node.right, node)
        
        getParent(root, None)
        res = []
        visited = set()
        def dfs(node, distance):
            if node == None or node.val in visited:
                return
            if distance == 0:
                res.append(node.val)
                return
            visited.add(node.val)
            dfs(node.left, distance - 1)
            dfs(node.right, distance - 1)
            dfs(pNodeMap[node.val], distance - 1)
        
        dfs(target, k)
        return res
    
    # bfs answer just in case
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # Step 1: Create a hash map to track parent nodes for each node.
        parent_map = {}
        
        # Helper function to populate the parent_map.
        def build_parent_map(node, parent):
            if node:
                parent_map[node.val] = parent  # Map the current node's value to its parent node.
                build_parent_map(node.left, node)  # Recurse on the left child.
                build_parent_map(node.right, node)  # Recurse on the right child.
        
        # Populate the parent map starting from the root.
        build_parent_map(root, None)

        # Step 2: Perform BFS to find all nodes at distance K from the target node.
        result = []  # To store the result nodes.
        visited = set()  # To track visited nodes and avoid cycles.
        queue = collections.deque([(target, 0)])  # Start BFS from the target node with distance 0.

        while queue:
            current_node, distance = queue.popleft()
            if current_node.val in visited:
                continue
            visited.add(current_node.val)  # Mark the current node as visited.

            # If the current distance equals K, add the node to the result list.
            if distance == k:
                result.append(current_node.val)
                continue  # No need to explore further from nodes at distance K.

            # Explore the neighbors (left child, right child, and parent).
            if current_node.left and current_node.left.val not in visited:
                queue.append((current_node.left, distance + 1))
            if current_node.right and current_node.right.val not in visited:
                queue.append((current_node.right, distance + 1))
            parent = parent_map[current_node.val]
            if parent and parent.val not in visited:
                queue.append((parent, distance + 1))

        return result
    
# 766. Toeplitz Matrix
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        for j, row in enumerate(matrix):
            for i, val in enumerate(row):
                if j > 0 and i > 0 and matrix[j-1][i-1] != val:
                    return False
        return True
    
# 129. Sum Root to Leaf Numbers
# one answer that makes more sense to me
# tho I do like one recursion matches one pop idea from https://programmercarl.com/0129.%E6%B1%82%E6%A0%B9%E5%88%B0%E5%8F%B6%E5%AD%90%E8%8A%82%E7%82%B9%E6%95%B0%E5%AD%97%E4%B9%8B%E5%92%8C.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC
# but other parts are confusing af, for this solutions pop
# think of it as pop from the current node after traversing left and right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        stack = []
        res = []

        def dfs(node):
            if not node:  # Base case: stop if the node is None
                return

            # Add the current node's value to the stack
            stack.append(node.val)

            # If it's a leaf node, calculate the number and store it
            if not node.left and not node.right:
                val = 0
                for num in stack:
                    val = val * 10 + num
                res.append(val)

            # Recurse for left and right children
            dfs(node.left)
            dfs(node.right)

            # Backtrack: remove the current node's value from the stack
            stack.pop()

        dfs(root)
        return sum(res)


# 270. Closest Binary Search Tree Value
class Solution:
    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        closest = root.val
        while root:
            if abs(root.val - target) < abs(closest - target):
                closest = root.val
            elif abs(root.val - target) == abs(closest - target):
                closest = min(closest, root.val)
            
            if target <= root.val:
                root = root.left
            else:
                root = root.right
        return closest

# 14. Longest Common Prefix
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""
        for j in range(len(min(strs, key=lambda x : len(x)))):
            currChar = strs[0][j]
            for i in range(len(strs)):
                if currChar != strs[i][j]:
                    return res
            res += currChar
        return res

# optimized version
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        min_str = min(strs, key=lambda x : len(x))
        for j in range(len(min_str)):
            currChar = strs[0][j]
            for i in range(len(strs)):
                if currChar != strs[i][j]:
                    return strs[0][0:j]
        return min_str


# 31. Next Permutation
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # [2, 5, 6, 4, 3] -> [2, 6, 5, 4, 3] -> [2, 6, 3, 4, 5]
        temp_idx = -1 #[1]
        for i in range(len(nums) - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                temp_idx = i - 1 
                break

        if temp_idx != -1:
            for i in range(len(nums) - 1, 0, -1):
                if nums[i] > nums[temp_idx]:
                    nums[i], nums[temp_idx] = nums[temp_idx], nums[i]
                    break
        # [2, 6, 5, 4, 3]

        # temp_idx + 1:-1
        temp_idx += 1
        end = len(nums) - 1
        while temp_idx < end:
            nums[temp_idx], nums[end] = nums[end], nums[temp_idx]
            temp_idx += 1
            end -= 1
        return nums

# 958. Check Completeness of a Binary Tree
# the important thing is that we don't want to check if node.left
# because we are checking null as part of the process
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        isEmpty = False
        queue = deque([root])
        while queue:
            for i in range(len(queue)):
                node = queue.popleft()
                if not node:
                    isEmpty = True
                else:
                    if isEmpty:
                        return False
                    queue.append(node.left)
                    queue.append(node.right)
        return True
    
# 116. Populating Next Right Pointers in Each Node
# O(1) space solution
# kinda hard to remember, but both leftmost and head points to the upper level
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return root
        leftmost = root
        while leftmost.left:
            head = leftmost
            while head:
                head.left.next = head.right
                
                if head.next:
                    head.right.next = head.next.left
                head = head.next
            leftmost = leftmost.left
        return root
# O(n) space solution
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return root
        queue = deque([root])
        while queue:
            length = len(queue)
            for i in range(length):
                node = queue.popleft()
                if i + 1 < length:
                    node.next = queue[0]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root
    
# 708. Insert into a Sorted Circular Linked List
# thing to remember, set prev and curr before the loop
# while True
class Solution:
    # [2,3,4] 1 or 5
    # curr >= iv >= prev
    # iv >= prev or iv <= curr
    # iv == prev == curr
    # if not head
    def insert(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
        if not head:
            node = Node(insertVal)
            node.next = node
            return node
        prev = head
        curr = head.next
        while True:
            corrPos = False
            if curr.val >= insertVal >= prev.val:
                corrPos = True
            elif prev.val > curr.val and (insertVal >= prev.val or insertVal <= curr.val):
                corrPos = True
            
            if corrPos:
                node = Node(insertVal)
                prev.next = node
                node.next = curr
                return head
            
            if curr == head:
                break
            
            prev = curr
            curr = curr.next
        
        # prev.next = Node(insertVal, curr)
        node = Node(insertVal)
        prev.next = node
        node.next = curr

        return head
    

# 78. Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        path = []
        
        def bt(nums, startIndex):
            # if startIndex >= len(nums):
            #     return
            for i in range(startIndex, len(nums)):
                path.append(nums[i])
                res.append(path[:])
                bt(nums, i + 1)
                path.pop()

        bt(nums, 0)
        return res


# 139. Word Break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        queue = deque([0])
        wordSet = set(wordDict)
        visited = set()

        while queue:
            start = queue.popleft()
            if start == len(s):
                return True

            for end in range(start + 1, len(s) + 1):
                if end in visited:
                    continue

                word = s[start:end]
                if word in wordSet:
                    queue.append(end)
                    visited.add(end)

        return False
    
# 169. Majority Element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        res = None
        for num in nums:
            if count == 0:
                res = num
            
            if num == res:
                count += 1
            else:
                count -= 1
        return res

# 1768. Merge Strings Alternately
# could use one pointer, and without the final if else
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        ptr1, ptr2 = 0, 0
        res = ""
        while ptr1 < len(word1) and ptr2 < len(word2):
            res += word1[ptr1]
            res += word2[ptr2]
            ptr1 += 1
            ptr2 += 1
        
        if ptr1 < len(word1):
            return res + word1[ptr1:]
        else:
            return res + word2[ptr2:]
        

# 721. Accounts Merge
from collections import defaultdict

class Solution:
    def accountsMerge(self, account_list):
        # Build the adjacency list
        adjacent = defaultdict(list)
        for account in account_list:
            first_email = account[1]
            for email in account[2:]:
                adjacent[first_email].append(email)
                adjacent[email].append(first_email)

        # Traverse the accounts to find components
        visited = set()
        merged_accounts = []

        def dfs(email, merged_account):
            visited.add(email)
            merged_account.append(email)
            for neighbor in adjacent[email]:
                if neighbor not in visited:
                    dfs(neighbor, merged_account)

        for account in account_list:
            account_name = account[0]
            first_email = account[1]

            if first_email not in visited:
                merged_account = []
                dfs(first_email, merged_account)
                merged_accounts.append([account_name] + sorted(merged_account[:]))

        return merged_accounts
    
# 133. Clone Graph
# i think the takeaway is that if the node is in the queue, then we are traversing
# otherwise we have visited hence no point of putting it to the queue again
# same as neighbor, no need to create it again
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
            
        visited = {}
        queue = deque([node])
        visited[node] = Node(1)

        while queue:
            for i in range(len(queue)):
                curr = queue.popleft()
                for neighbor in curr.neighbors:
                    if neighbor not in visited:
                        visited[neighbor] = Node(neighbor.val)
                        queue.append(neighbor)
                    visited[curr].neighbors.append(visited[neighbor])
        return visited[node]

# 415. Add Strings
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        carry = 0
        res = []
        ptr1, ptr2 = len(num1) - 1, len(num2) - 1
        while ptr1 >= 0 or ptr2 >= 0:
            val1 = ord(num1[ptr1]) - ord("0") if ptr1 >= 0 else 0
            val2 = ord(num2[ptr2]) - ord("0") if ptr2 >= 0 else 0
            sum = val1 + val2 + carry
            carry = 1 if sum >= 10 else 0
            sum = sum % 10
            res.append(sum)
            ptr1 -= 1
            ptr2 -= 1
        
        if carry == 1:
            res.append(1)

        return "".join(str(num) for num in res[::-1])


# 121. Best Time to Buy and Sell Stock
# its like kadane's algo
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minPrice = float("inf")
        profit = 0
        for price in prices:
            minPrice = min(price, minPrice)
            profit = max(profit, price-minPrice)
        return profit
        
