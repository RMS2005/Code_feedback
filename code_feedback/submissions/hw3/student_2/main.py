import math
def is_prime(num):
    if num <= 1: return False
    if num == 2: return True
    if num % 2 == 0: return False
    for i in range(3, int(math.sqrt(num)) + 1, 2):
        if num % i == 0:
            return False
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(is_prime(int(sys.argv[1])))
