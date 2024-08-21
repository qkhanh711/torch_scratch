import model.pl_modules.simple_nn as plm
import model.nn_modules.simple_nn as nnm
import torch
from torch.utils.tensorboard import SummaryWriter

import dataloader
import config

train_loader, test_loader = dataloader.get_dataloader(
    config.DATASET, config.BATCH_SIZE, config.NUM_WORKERS
)

plmNet = plm.Net
nnmNet = nnm.Net

class Train:
    def __init__(self, framework, config, train_loader, test_loader) -> None:
        self.framework = framework
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        if self.framework == "pl":
            self.model = plmNet(
                in_dims=self.config.N_INPUT,
                n_classes=self.config.N_CLASSES,
                n_layer_1=self.config.N_LAYER_1,
                n_layer_2=self.config.N_LAYER_2,
                lr=self.config.LEARNING_RATE,
            )
        else:
            self.model = nnmNet(
                n_input=self.config.N_INPUT,
                n_output=self.config.N_CLASSES,
            )

        self.writer = SummaryWriter(log_dir=self.config.LOG_DIR)

    def train(self, num_epochs):
        if self.framework == "pl":
            self.train_pl(num_epochs)
        else:
            self.train_pytorch(num_epochs)

    def train_pl(self, num_epochs):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

        early_stop_callback = EarlyStopping(
            monitor="valid_acc",
            min_delta=0.00,
            patience=self.config.PATIENCE,
            verbose=False,
            mode="max",
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_acc",
            dirpath=self.config.MODEL_DIR,
            filename="model-{epoch:02d}-{valid_acc:.2f}",
            save_top_k=1,
            mode="max",
        )

        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=num_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            log_every_n_steps=self.config.LOG_EVERY_N_STEPS,
        )

        trainer.fit(self.model, self.train_loader, self.test_loader)

    def train_pytorch(self, num_epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0

            for inputs, targets in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == targets).sum().item()

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader.dataset)

            self.model.eval()
            test_loss = 0.0
            test_acc = 0.0

            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_acc += (predicted == targets).sum().item()

            test_loss /= len(self.test_loader)
            test_acc /= len(self.test_loader.dataset)

            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
            self.writer.add_scalar("Test/Loss", test_loss, epoch)
            self.writer.add_scalar("Test/Accuracy", test_acc, epoch)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

        self.writer.flush()
        self.writer.close()


if __name__ == "__main__":
    train = Train("pl", config, train_loader, test_loader)
    train.train(config.EPOCHS)

    train = Train("nn", config, train_loader, test_loader)
    train.train(config.EPOCHS)