import torch
from torchvision.models import inception_v3
from torchvision.transforms import v2
from torchvision.datasets import UCF101
from utils import Identity


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception=inception_v3(pretrained=True)
embed_num=inception.fc.in_features
inception.fc=Identity()

for param in inception.parameters():
    param.requires_grad = False
inception.eval()
preprocess = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(299, antialias=True),
        v2.CenterCrop(299),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Lambda(inception)
    ])
def save_tensor(data, x_file, y_file):
    x_lst=[]
    y_lst=[]
    length=len(data)
    for i in range(length):
        print(f"{i}/{length}")
        x,_,y= data[i]
        x_lst.append(x.unsqueeze(0))
        y_lst.append(y)
    data_x=torch.vstack(x_lst)
    data_y=torch.tensor(y_lst)
    print(data_x.shape, data_y.shape)
    torch.save(data_x, x_file+".pt")
    torch.save(data_y, y_file + ".pt")


if __name__=="__main__":
    seq_length=20
    train_dataset=UCF101(root="C:\\Users\\Renat\\Downloads\\UCF101\\UCF101\\", annotation_path="C:\\Users\\Renat\\Downloads\\UCF101TrainTestSplits-RecognitionTask\\ucfTrainTestlist\\",
                   frames_per_clip=seq_length, step_between_clips=1, output_format="TCHW", num_workers=4, transform=preprocess)

    save_tensor(train_dataset, "train_x", "train_y")