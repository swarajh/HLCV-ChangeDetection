import timm

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True
)

print(type(model))
print(model)