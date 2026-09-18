with open("data/LaBr3 spectrum 1kb/LaBr3_Ba133_5mC.txt") as f:
    for i, line in enumerate(f):
        print(repr(line))
        if i >= 3:
            break