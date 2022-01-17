from pytorch_lightning.callbacks import ModelCheckpoint

# experimenr variables
dataset_names = ('20_news_group',
                 '5234_event_tweets',
                 'tag_my_news')
topic_nums = (20, 30, 50, 75, 100)


def configure_model_checkpoints(monitor: str, save_dir: str, version: str):
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=monitor,
        mode="max",
        dirpath=save_dir,
        filename=f"best_{monitor}_{{epoch:02d}}-{{{monitor}:.4f}}_{version}",
        save_weights_only=True
    )
    return [checkpoint_callback]
