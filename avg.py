import os
import pickle

avg_scores = {}
score_files = os.listdir('.')
count = 0
for score_file in score_files:
    #if '1.pkl' in score_file or '2.pkl' in score_file:
    #    continue
    if not score_file.endswith('.pkl'):
        continue
    count += 1
    with open(score_file, 'rb') as f:
        x = pickle.load(f)
    for key, item in x.items():
        if key not in avg_scores:
            avg_scores[key] = item
        else:
            avg_scores[key] += item

for key in avg_scores.keys():
    avg_scores[key] /= count
    print(f"{key}: {avg_scores[key]:.4f}")
print(count)
