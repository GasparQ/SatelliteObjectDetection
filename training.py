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

    from sod import SegmentationDatasetHDF5, config
    from sod.unet import Experiment

    # train_images_path = glob(os.path.join(args.train_images, '*.png'))
    # train_masks_path = glob(os.path.join(args.train_masks, '*.png'))
    # if len(train_images_path) != len(train_masks_path):
    #     raise RuntimeError("Number of images must be equal to number of masks for Training dataset")

    # test_images_path = glob(os.path.join(args.test_images, '*.png'))
    # test_masks_path = glob(os.path.join(args.test_masks, '*.png'))
    # if len(test_images_path) != len(test_masks_path):
    #     raise RuntimeError("Number of images must be equal to number of masks for Test dataset")

    # create the train and test datasets
    # train_dataset = SegmentationDataset(
    #     image_paths=train_images_path,
    #     mask_paths=train_masks_path,
    #     crop_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
    #     device=config.DEVICE,
    #     sample_factor=args.prefetch,
    #     nworkers=int(args.workers * 0.8)
    # )

    # test_dataset = SegmentationDataset(
    #     image_paths=test_images_path,
    #     mask_paths=test_masks_path,
    #     crop_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
    #     device=config.DEVICE,
    #     nworkers=int(args.workers * 0.2),
    #     auto_resample=False
    # )

    train_dataset = SegmentationDatasetHDF5(args.train_dataset)
    test_dataset = SegmentationDatasetHDF5(args.test_dataset)

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
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
    	batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # create a new experiment
    unet_experiment = Experiment.create(config=config)

    # perform full training of datasets
    train_losses, test_losses = unet_experiment.train(
        train_dataset=train_loader,
        test_dataset=test_loader,
        plot_loss_history=True
    )

    # serialize the model to disk
    unet_experiment.save()
