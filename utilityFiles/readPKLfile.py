import pickle


with open('../example/cache/ViT-L-14_openai_trendings.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))

for key in data:
    print(f"{key}->{data[key]}")
    break



with open('trendings.txt', 'w') as f:
    
    for key in data:
        f.write(f"{key}->\n")
        for i in data[key]:
            f.write(f"{i}\n")
        f.write('\n')
        break


        