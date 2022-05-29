if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', help='Path to the training dataset in hdf5 format')
    parser.add_argument('test_dataset', help='Path to the test dataset in hdf5 format')

    parser.add_argument('-w', '--workers',
                        help='Number of workers for dataset loading',
                        type=int,
                        default=8)
    parser.add_argument('-p', '--prefetch',
                        help='Number of sample to prefetch for each training sample',
                        type=int,
                        default=2)

    args = parser.parse_args()

    import logging

    from torch.utils.data import DataLoader

    from sod import SegmentationDatasetHDF5Groupped, config
    from sod.unet import Experiment
    # from sod.unet_resnet import Experiment

    train_dataset = SegmentationDatasetHDF5Groupped(
        args.train_dataset,
        config.INPUT_IMAGE_WIDTH,
        config.INPUT_IMAGE_HEIGHT,
        augment=True
    )
    test_dataset = SegmentationDatasetHDF5Groupped(
        args.test_dataset,
        config.INPUT_IMAGE_WIDTH,
        config.INPUT_IMAGE_HEIGHT,
    )

    logging.info(f"Found {len(train_dataset)} examples in the training set...")
    logging.info(f"Found {len(test_dataset)} examples in the test set...")

    # create the training and test data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=args.prefetch,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
    	batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # create a new experiment
    experiment = Experiment.create(config=config)

    # perform full training of datasets
    train_losses, test_losses = experiment.train(
        train_dataset=train_loader,
        test_dataset=test_loader,
        plot_loss_history=True
    )

    # serialize the model to disk
    experiment.save()
