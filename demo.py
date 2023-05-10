lines=[{"id":1},{"id":2}]
with open('your_file.txt', 'w') as f:
    for line in lines:
        f.write(f"{line}\n")