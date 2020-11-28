import torch
import cv2
from tqdm import tqdm
import numpy as np
import torch.optim
from tensorboardX import SummaryWriter
import torchvision
from PIL import Image

writer = SummaryWriter(comment='visual')

feature_result = None
def feature_hoook(layer, data_input, data_output):
    global feature_result
    feature_result = data_output


vgg16 = torchvision.models.vgg16(pretrained=True).cuda().eval()  # Test NetWork : Vgg16
vgg16.features[24].register_forward_hook(feature_hoook)


def visiual(sz=56, nthfeaturemap=1):

    img = np.uint8(np.random.uniform(0, 250, (sz, sz, 3))) / 255

    for i in range(15):

        img = torch.tensor(img.transpose((2, 0, 1))).unsqueeze(0).to(torch.float32)
        img = img.cuda()
        img.requires_grad = True
        optim = torch.optim.Adam([img], lr=0.1, weight_decay=1e-6)  # 这里参数需要是Tensors，使用列表代替

        for n in range(12):
            optim.zero_grad()
            vgg16(img)
            loss = -1 * feature_result[0, nthfeaturemap].mean()
            loss.backward()
            optim.step()

        img = img.data.cpu().numpy()[0].transpose(1, 2, 0)
        sz = int(1.2 * sz)
        img = cv2.resize(img, (sz, sz))
        img = cv2.blur(img, (5, 5))
    return img

for i in tqdm(range(512)):
    img = visiual(56, i)
    cv2.imwrite(f'.//FeatureMap//24-{i}.png', img*255)

