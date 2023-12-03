import torch
from src.models import backbone
if __name__ == "__main__":

    #model_path = "C:/Users/Amar/Documents/resnet50_best.pt"
#
    #model = torch.load(model_path)
#
    #model_scripted = torch.jit.script(model)
    #model_scripted.save('C:/Users/Amar/Documents/resnet50_scripted.pt')

    model_path = "C:/Users/Amar/Documents/densenet121_epoch_15.pth"
    densenet = cnn_models.sequential_model("densenet121", n_classes=13)
    dict = torch.load(model_path)
    densenet.load_state_dict(dict['model'])

    model_scripted = torch.jit.script(densenet)
    model_scripted.save("C:/Users/Amar/Documents/densenet121_scripted.pt")
    torch.save(densenet, "C:/Users/Amar/Documents/dsn.pt")
