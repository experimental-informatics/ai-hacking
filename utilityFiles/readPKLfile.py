import pickle


with open('cache/ViT-L-14_openai_artists.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))

for key in data:
    print(f"{key}->{data[key]}")
    break



with open('openai_artists.txt', 'w') as f:
    
    for key in data:
        f.write(f"{key}->\n")
        for i in data[key]:
            f.write(f"{i}\n")
        f.write('\n')
        


        