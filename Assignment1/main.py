import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def query_crop(que, tx):
    img = cv2.cvtColor(cv2.imread(que), cv2.COLOR_BGR2RGB)
    txt = np.loadtxt(tx)
    if np.size(txt) == 4:
        return img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :]
    else:
        if txt[0][2]*txt[0][3] >= txt[1][2]*txt[1][3]:
            return img[int(txt[0][1]):int(txt[0][1] + txt[0][3]), int(txt[0][0]):int(txt[0][0] + txt[0][2]), :]
        else:
            return img[int(txt[1][1]):int(txt[1][1] + txt[1][3]), int(txt[1][0]):int(txt[1][0] + txt[1][2]), :]


def vgg_19_extraction(img, feat):
    img_transform = torch.unsqueeze(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])(transforms.ToTensor()(img)), 0)
    vgg19 = models.vgg19(pretrained=True)
    vgg19_feat_extractor = vgg19.features
    vgg19_feat_extractor.eval()
    np.save(feat, vgg19(img_transform).cpu().detach().numpy())

def main():
    for jpg_name in tqdm(os.listdir('./datasets/gallery/')):
        path = os.path.join('./datasets/gallery/', jpg_name)
        img = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (224, 224),
                                interpolation=cv2.INTER_CUBIC)
        npy_name = jpg_name.split('.')[0] + '.npy'
        feat = os.path.join('./datasets/gallery_feature/', npy_name)
        vgg_19_extraction(img, feat)


    dic = {}
    f = open(r'./rankList.txt', 'w')
    for i in range(50):
        Q_num = 'Q'+str(i+1)+":"
        f.write(Q_num)

        vgg_19_extraction(cv2.resize(query_crop(('./datasets/query/'+str(i)+".jpg"), './datasets/query_txt/' + str(i) + '.txt'),
                                     (224, 224), interpolation=cv2.INTER_CUBIC), './datasets/query_feat/'+str(i)+'.npy')
        dic[i] = dict()
        for j in os.listdir('./datasets/gallery_feature/'):
            dic[i][j.split('.')[0]] = np.squeeze(cosine_similarity(np.load('./datasets/query_feat/'+str(i)+'.npy'),
                                                                np.load('./datasets/gallery_feature/' + j.split('.')[0] + '.npy')))
        keys = lambda item: item[1]
        sorted_dict = sorted(dic[i].items(), key=keys, reverse=True)
        for j in range(len(sorted_dict)):
            f.write(str(np.int32(sorted_dict[j][0]))+' ')
        f.write('\n')

if __name__ == "__main__":
    main()