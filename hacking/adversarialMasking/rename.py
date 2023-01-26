
import os

for k in range(0,1):
    dir = f'./{k}'

    for i in range(1001):
        # Absolute path of a file
        old_name = f"{dir}/{i}.jpg"
        new_name = f"{dir}/{str(i).zfill(4)}.jpg"

        os.rename(old_name, new_name)
        print(new_name)


    # Renaming the file