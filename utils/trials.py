m_start = "ILSVRC_RN152_B256_E="
acc_dict = {
    'model1': [0.8, 0.9, 0.7, 0.85],
    'model2': [0.75, 0.95, 0.6, 0.9],
    'model3': [0.85, 0.85, 0.8, 0.95]
}
with open(f'acc_{m_start.split("_E")[0]}.txt', 'w') as f:
    for model, acc in acc_dict.items():
        line = f"{model} {' '.join(str(x) for x in acc)}\n"
        f.write(line)