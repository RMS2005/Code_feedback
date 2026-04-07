import sys

# Slightly messy to give the AI something to talk about
def main():
    try:
        data = sys.stdin.read().split()
        if len(data) >= 2:
            num1 = int(data[0])
            num2 = int(data[1])
            # Sum using a simple logic
            res = num1 + num2
            print(res)
    except:
        pass

if __name__ == "__main__":
    main()