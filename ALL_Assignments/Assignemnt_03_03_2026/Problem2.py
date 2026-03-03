n = int(input())
arr = list(map(int, input().split()))

max_length = 0
current_length = 0

for num in arr:
    if num % 3 == 0:
        current_length += 1
        if current_length > max_length:
            max_length = current_length
    else:
        current_length = 0   # break in consecutive sequence

print(max_length)