import torch
import torch.nn as nn



def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def train(model, optimizer, Loss, data_iter, num_epochs, device, lr_scheduler=None):
    additional=[]
    for epoch in range(num_epochs):
        if epoch == 1:
            data_iter=additional
        checkpoint={"state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename="hum_action.pth.tar")
        print(f"epoch={epoch}/{num_epochs}")
        model.train()
        for i, (x, y) in enumerate(data_iter):
            if epoch==0:
                additional.append((x,y))
            x=x.to(device=device)
            y=y.to(device=device)
            optimizer.zero_grad()
            pred=model(x)

            loss=Loss(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            print(f"i={i}, loss={loss.item()}, optim_lr={optimizer.param_groups[0]['lr']}, accuracy={accuracy(pred, y)}")

        if not lr_scheduler==None:
            lr_scheduler.step()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



def accuracy(y_pred:torch.Tensor, y):
    y_pred=torch.argmax(y_pred, dim=1)
    return (y==y_pred).sum()/y.shape[0]
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])