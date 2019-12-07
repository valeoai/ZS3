from torch.utils.data import DataLoader

from zs3.dataloaders.datasets import combine_dbs, pascal, sbd, context


def make_data_loader(
    args,
    transform=True,
    load_embedding=None,
    w2c_size=300,
    weak_label=False,
    unseen_classes_idx_weak=[],
    **kwargs,
):
    if args.dataset == "pascal":
        train_set = pascal.VOCSegmentation(
            args,
            transform=transform,
            split="train",
            load_embedding=load_embedding,
            w2c_size=w2c_size,
            weak_label=weak_label,
            unseen_classes_idx_weak=unseen_classes_idx_weak,
        )
        val_set = pascal.VOCSegmentation(
            args, split="val", load_embedding=load_embedding, w2c_size=w2c_size
        )
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(
                args,
                transform=transform,
                split=["train_noval"],
                load_embedding=load_embedding,
                w2c_size=w2c_size,
                weak_label=weak_label,
                unseen_classes_idx_weak=unseen_classes_idx_weak,
            )
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "context":
        train_set = context.ContextSegmentation(
            args,
            transform=transform,
            split="train",
            load_embedding=load_embedding,
            w2c_size=w2c_size,
            weak_label=weak_label,
            unseen_classes_idx_weak=unseen_classes_idx_weak,
        )
        val_set = context.ContextSegmentation(
            args, split="val", load_embedding=load_embedding, w2c_size=w2c_size
        )
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError
