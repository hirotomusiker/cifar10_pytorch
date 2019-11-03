import torchvision.transforms as T


def build_transforms(cfg, is_train=True):
    list_transforms = list()

    # Resizing. Some models must have ImageNet-size images as input
    if is_train and cfg.DATA.IMG_SIZE != 32:
        list_transforms.append(T.Resize((cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)))
    elif is_train == False and cfg.TEST.IMG_SIZE != 32:
        list_transforms.append(T.Resize((cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)))

    if is_train:
        if cfg.DATA.RANDOMCROP:
            list_transforms.append(T.RandomCrop(cfg.DATA.IMG_SIZE, padding=4))
        if cfg.DATA.LRFLIP:
            list_transforms.append(T.RandomHorizontalFlip())

    list_transforms.extend([
        T.ToTensor(),
        T.Normalize(cfg.DATA.NORMALIZE_MEAN, cfg.DATA.NORMALIZE_STD),
    ])

    transforms = T.Compose(list_transforms)

    return transforms
