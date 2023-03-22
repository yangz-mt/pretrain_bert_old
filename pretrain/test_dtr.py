import torch
# import pdb;pdb.set_trace()

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

dummy_inp = torch.ones(1, 3, 224, 224)

if torch.cuda.is_available():
    input_batch = dummy_inp.to('cuda')
    model.to('cuda')

model._apply(lambda t: t.detach().checkpoint())

with torch.no_grad():
    output = model(input_batch)

print("\n\n\nbefore print\n\n\n")
print(output)
