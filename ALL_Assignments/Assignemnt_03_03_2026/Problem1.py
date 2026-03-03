n = int(input())
arr = list(map(int, input().split()))

found = False

for num in arr:
    if num % 3 == 0:
        print(num, end=" ")
        found = True

if not found:
    print(-1)