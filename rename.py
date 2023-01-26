
import os

dir = 'images'

for i in range(2000):
    # Absolute path of a file
    old_name = f"{dir}/img{i}.png"
    new_name = f"{dir}/{str(i).zfill(4)}.png"

    os.rename(old_name, new_name)
    print(new_name)


# Renaming the file
