import sys

# Simple Python implementation for the addition task
def solve():
    try:
        input_data = sys.stdin.read().split()
        if len(input_data) >= 2:
            a = int(input_data[0])
            b = int(input_data[1])
            print(a + b)
    except Exception:
        pass

if __name__ == "__main__":
    solve()
