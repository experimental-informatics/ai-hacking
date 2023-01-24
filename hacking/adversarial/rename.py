
import os

dir = './4'

for i in range(101):
    # Absolute path of a file
    old_name = f"{dir}/{i}.jpg"
    new_name = f"{dir}/{str(i).zfill(4)}.jpg"

    os.rename(old_name, new_name)
    print(new_name)


# Renaming the file