import torch
import matplotlib.pyplot as plt


dataset = torch.load('./datasets/cloud_dataset.pt', map_location='cpu')

fig, axis = plt.subplots(4, 20, figsize=(20, 4), dpi=300)

for i in range(0, 80):
    a = axis[i//20, i%20]
    # a.imshow((dataset['radiances'][i].numpy() - dataset_ao['radiances'][i].numpy()) ** (1.0/2.2))
    a.imshow((dataset['radiances'][i].numpy()) ** (1.0/2.2))
    a.invert_yaxis()
    a.axis('off')
fig.tight_layout(pad=0.0)
plt.show()

