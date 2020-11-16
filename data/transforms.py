import pydoc

import albumentations as albu
from albumentations.pytorch import ToTensor, ToTensorV2


def get_transforms(cfg):
    pre_transforms = parse_transforms(cfg.dataset.pre_transforms)
    augmentations = parse_transforms(cfg.dataset.augmentations)
    post_transforms = parse_transforms(cfg.dataset.post_transforms)

    transforms = dict(
        train=pre_transforms + augmentations + post_transforms,
        valid=pre_transforms + post_transforms,
        test=pre_transforms + post_transforms,
    )
    for k, v in transforms.items():
        transforms[k] = albu.Compose(v)

    return transforms


def parse_transforms(cfg) -> list:
    transforms = []

    for transform in cfg:
        if transform.name in ("Compose", "OneOf", "OneOrOther"):
            # get inner list of transforms
            inner_transforms = parse_transforms(transform.list)
            params = {k: v for k, v in transform.items() if k not in ("name", "list")}
            transform_obj = get_albu_object(transform.name, inner_transforms, **params)

        else:
            params = {k: v for k, v in transform.items() if k not in ("name", "list")}
            transform_obj = get_albu_object(transform.name, **params)

        transforms += [transform_obj]

    return transforms


def get_albu_object(name, *args, **kwargs):
    if name in globals():
        cls = globals()[name]
    else:
        cls_path = "albumentations." + name
        cls = pydoc.locate(cls_path)

    return cls(*args, **kwargs)


if __name__ == "__main__":
    # For pycharm to not remove imports during imports optimization
    ToTensor()
    ToTensorV2()
