import heapq

n = int(input())
arr = list(map(int, input().split()))

if n < 3:
    print(0)
    exit()

left = [i - 1 for i in range(n)]
right = [i + 1 for i in range(n)]
heap = []

def is_peak(i):
    if left[i] >= 0 and right[i] < n:
        return arr[i] > arr[left[i]] and arr[i] > arr[right[i]]
    return False

for i in range(1, n - 1):
    if is_peak(i):
        heapq.heappush(heap, (-arr[i], i))

total_wood = 0
removed = [False] * n

while heap:
    value, i = heapq.heappop(heap)
    
    if removed[i] or not is_peak(i):
        continue
    
    total_wood += arr[i]
    removed[i] = True
    
    l = left[i]
    r = right[i]
    
    if l >= 0:
        right[l] = r
    if r < n:
        left[r] = l
    
    if l > 0 and not removed[l] and is_peak(l):
        heapq.heappush(heap, (-arr[l], l))
    if r < n - 1 and not removed[r] and is_peak(r):
        heapq.heappush(heap, (-arr[r], r))

print(total_wood)