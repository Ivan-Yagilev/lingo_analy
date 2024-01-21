import pandas as pd


file_location = "./Datasets/doc_comment_summary.xlsx"
df = pd.read_excel(file_location, index_col=0)

border = len(df) // 2
pos_index = 0
neg_index = 0
unsup_index = 0

for i in range(len(df)):
    try:
        emoj = int(str(df.iloc[i])[6:8])
        if emoj < 0:
            with open(f'./Datasets/2016comments/train/neg/{neg_index}_{abs(emoj)}.txt', 'w') as f:
                f.write(str(df.iloc[i])[14:-13])
                neg_index += 1
        elif emoj == 0:
            with open(f'./Datasets/2016comments/train/unsup/{unsup_index}_{emoj}.txt', 'w') as f:
                f.write(str(df.iloc[i])[14:-13])
                unsup_index += 1
        else:
            with open(f'./Datasets/2016comments/train/pos/{pos_index}_{emoj}.txt', 'w') as f:
                f.write(str(df.iloc[i])[14:-13])
                pos_index += 1
    except ValueError:
        pass

pos_index = 0
neg_index = 0
unsup_index = 0
for i in range(border, len(df)):
    try:
        emoj = int(str(df.iloc[i])[6:8])
        if emoj < 0:
            with open(f'./Datasets/2016comments/test/neg/{neg_index}_{abs(emoj)}.txt', 'w') as f:
                f.write(str(df.iloc[i])[14:-13])
                neg_index += 1
        elif emoj == 0:
            with open(f'./Datasets/2016comments/test/unsup/{unsup_index}_{emoj}.txt', 'w') as f:
                f.write(str(df.iloc[i])[14:-13])
                unsup_index += 1
        else:
            with open(f'./Datasets/2016comments/test/pos/{pos_index}_{emoj}.txt', 'w') as f:
                f.write(str(df.iloc[i])[14:-13])
                pos_index += 1
    except ValueError:
        pass
