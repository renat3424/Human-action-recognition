import torch
import torch.nn as nn
from torchvision.datasets import UCF101
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import train, accuracy, custom_collate
from model import TransformerModel
from preprocess_data import preprocess, embed_num



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")




if __name__=="__main__":
    print(device)
    batch_size=32
    seq_length=20
    train_dataset=UCF101(root="C:\\Users\\Renat\\Downloads\\ucf101subsample\\UCF101\\UCF101\\", annotation_path="C:\\Users\\Renat\\Downloads\\ucf101subsample\\UCF101TrainTestSplits-RecognitionTask\\ucfTrainTestlist\\",
                   frames_per_clip=seq_length, step_between_clips=5*seq_length, output_format="TCHW", num_workers=4, transform=preprocess)


    print(train_dataset)
    heads_num=1
    num_layers=3
    dropout=0.3
    learning_rate=0.0001
    num_classes=len(train_dataset.classes)

    print(num_classes, len(train_dataset))
    model=TransformerModel(embed_num, heads_num, num_layers, seq_length, dropout, num_classes, device=device).to(device=device)
    num_epochs=20
    Loss=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)
    train_iter = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate)
    lr_scheduler=optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)
    train(model, optimizer, Loss, train_iter, num_epochs, device, lr_scheduler=lr_scheduler)
    test_dataset=UCF101(root="C:\\Users\\Renat\Downloads\\ucf101subsample\\UCF101\\UCF101\\", annotation_path="C:\\Users\\Renat\\Downloads\\ucf101subsample\\UCF101TrainTestSplits-RecognitionTask\\ucfTrainTestlist\\",
                   frames_per_clip=seq_length, step_between_clips=5*seq_length, output_format="TCHW", num_workers=4, transform=preprocess, train=False)


    test_iter = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True, num_workers=4, collate_fn=custom_collate)

    with torch.no_grad():
        model.eval()
        x, y=next(iter(test_iter))
        x=x.to(device=device)

        y_pred=model(x).to(device=torch.device("cpu"))
        print(accuracy(y_pred, y).item())

        video=test_dataset.video_clips.get_clip(0)[0]
        target=test_dataset.video_clips.get_clip(0)[-1]
        x=preprocess(video).unsqueeze(dim=0).to(device=torch.device("cuda"))
        y=model(x).to(device=torch.device("cpu"))
        targets=train_dataset.classes
        print(targets[torch.argmax(y, dim=1)[0].item()], targets[target])
        plt.imshow(video[10].permute(1, 2, 0))
        plt.show()

