"""
author: Jørn Eirik Betten
Acknowledges that the base trainer is fully based on Lucía Moran's
class Trainer (MLPTrainer is a copy, and not Jørn Eirik Betten's work), and BaseTrainer is a generalization that will be
used to train Gaussian Processes and Deep Kernel Gaussian Processes aswell.
"""
from abc import ABC, abstractmethod
import numpy as np
import torch
import gpytorch
import torch.nn.functional as F
import pandas as pd
from .data_preprocessing import get_target_list
import tqdm

class BaseTrainer(ABC):
    """
    Base trainer class for machine learning models.
    """
    def __init__(self, model, optimizer):
        self._optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print("Using " + str(self.device) + ".")
        self._model = model.to(self.device)

    @abstractmethod
    def _train(self):
        """
        To be overwritten by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """
        To be overwritten by subclass.
        """
        raise NotImplementedError

    def _mae(self, predictions, targets):
        """
        Mean absolute error (expected L1 Loss).
        """
        #predictions = predictions.numpy(); targets = targets.numpy()
        l1_distances = np.abs(targets - predictions)
        return np.mean(l1_distances)

    def _rmse(self, predictions, targets):
        """
        Root mean squared error.
        """
        #predictions = predictions.numpy(); targets = targets.numpy()
        l1_distances = np.abs(targets - predictions)
        return np.sqrt(np.mean(np.power(l1_distances, 2)))

    def _mse(self, predictions, targets):
        """
        Mean squared error.
        """
        #predictions = predictions.numpy(); targets = targets.numpy()
        diff = targets - predictions
        return np.mean(np.power(diff, 2))

    def _r_squared(self, predictions, targets):
        #predictions = predictions.numpy(); targets = targets.numpy()
        target_mean = np.mean(targets)

        return 1 - (np.sum(np.power(targets - predictions, 2)) / np.sum(np.power(targets - target_mean, 2)))

    def predict_batch(self, batch, target_means=0, target_stds=1):

        """Makes predictions on a given batch.

        Returns:
            list: The predictions.
        """

        self._model.eval()
        batch_x = batch[0].to(self.device)

        # get predictions for batch
        predictions = (self.model(batch_x).cpu().detach().numpy() * target_stds + target_means).tolist()

        return predictions

    def predict_loader(self, loader, target_means=0, target_stds=1):

        """Makes predictions on a given dataloader.

        Returns:
            list: The predictions.
        """

        predictions = []

        for batch in loader:
            predictions.extend(self.predict_batch(batch, target_means=target_means, target_stds=target_stds))

        return predictions

    @property
    def model(self):
        """
        Getter for model
        """
        return self._model

class MLPTrainer(BaseTrainer):
    """
    Class containing functions for training the MLP.
    """
    def __init__(self, model, optimizer, scheduler=None, gradient_accumulation_splits=1):
        super().__init__(model, optimizer)
        self._scheduler = scheduler
        self._gradient_accumulation_splits = gradient_accumulation_splits
        self._trained = False

    def _train(self, train_loader):

        """Performs a full training step. Depending on the setting for gradient accumulation, performs backward pass only
           every n batch.

        Returns:
            float: The obtained training loss.
        """

        self._model.train()
        loss_all = 0

        for batch_idx, batch in enumerate(train_loader):

            # transfer to device
            batch_x = batch[0].to(self.device)
            batch_y = batch[1].view(-1).to(self.device)

            loss = F.mse_loss(self._model(batch_x), batch_y)
            loss.backward()
            loss_all += loss.item() * len(batch[0])

            # gradient accumulation
            if ((batch_idx + 1) % self._gradient_accumulation_splits == 0) or (batch_idx + 1 == len(train_loader)):
                self._optimizer.step()
                self._optimizer.zero_grad()

        return loss_all / len(train_loader.dataset)


    def training_info(self):
        """
        Getter for training information.
        """
        if not self._trained:
            print("You need to train the model before you get the training information.")
            return 0
        else:
            return self._training_information

    def run(self, train_loader, train_loader_unshuffled, val_loader, test_loader, n_epochs=300, target_means=0, target_stds=1):

        """Runs a full training loop with metric ltraining_proceduresogging.

        Args:
            train_loader (Dataloader): The dataloader for the training points.
            train_loader_unshuffled (Dataloader): The dataloader for the training points but unshuffled. This is used to calculate metrics on the training set.
            val_loader (Dataloader): The dataloader for the validation points.
            test_loader (Dataloader): The dataloader for the test points.
            n_epochs (int): The number of epochs to perform.
            target_means(np.array): An array of the target means from standard scaling.
            target_stds(np.array): An array of the target stds from standard scaling.

        Returns:
            model: The trained model.
        """
        log_epoch = []
        log_lr = []
        log_loss = []

        log_train_error = []
        log_val_error = []
        log_test_error = []

        log_train_mae = []
        log_val_mae = []
        log_test_mae = []

        log_train_rmse = []
        log_val_rmse = []
        log_test_rmse = []

        log_train_r_squared = []
        log_val_r_squared = []
        log_test_r_squared = []

        # get targets off all sets
        train_targets = get_target_list(train_loader_unshuffled, target_means=target_means, target_stds=target_stds)
        val_targets = get_target_list(val_loader, target_means=target_means, target_stds=target_stds)
        test_targets = get_target_list(test_loader, target_means=target_means, target_stds=target_stds)

        best_val_error = None
        for epoch in range(1, n_epochs + 1):

            # get learning rate from scheduler
            if self._scheduler is not None:
                lr = self._scheduler.optimizer.param_groups[0]['lr']

            # training step
            loss = self._train(train_loader)

            # get predictions for all sets
            train_predictions = np.array(self.predict_loader(train_loader_unshuffled, target_means=target_means, target_stds=target_stds))
            val_predictions = np.array(self.predict_loader(val_loader, target_means=target_means, target_stds=target_stds))
            test_predictions = np.array(self.predict_loader(test_loader, target_means=target_means, target_stds=target_stds))

            train_error = self._mae(train_targets, train_predictions)
            val_error = self._mae(val_targets, val_predictions)

            # learning rate scheduler step
            if self._scheduler is not None:
                self._scheduler.step(val_error)

            # retain early stop test error
            if best_val_error is None or val_error <= best_val_error:
                test_error = self._mae(test_targets, test_predictions)
                best_val_error = val_error

            output_line = f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, 'f'Train MAE: {train_error:.7f}, 'f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}'
            #print(output_line)


            # Logging training data
            log_epoch.append(epoch)
            log_lr.append(lr)
            log_loss.append(loss)

            log_train_error.append(train_error)
            log_val_error.append(val_error)
            log_test_error.append(test_error)

            log_train_mae.append(self._mae(train_predictions, train_targets))
            log_val_mae.append(self._mae(val_predictions, val_targets))
            log_test_mae.append(self._mae(test_predictions, test_targets))

            log_train_rmse.append(self._rmse(train_predictions, train_targets))
            log_val_rmse.append(self._rmse(val_predictions, val_targets))
            log_test_rmse.append(self._rmse(test_predictions, test_targets))

            log_train_r_squared.append(self._r_squared(train_predictions, train_targets))
            log_val_r_squared.append(self._r_squared(val_predictions, val_targets))
            log_test_r_squared.append(self._r_squared(test_predictions, test_targets))

        training_info = {"epoch": log_epoch,
                         "learning_rate": log_lr,
                         "loss": log_loss,
                         "train_error": log_train_error,
                         "val_error": log_val_error,
                         "test_error": log_test_error,
                         "train_mae": log_train_mae,
                         "val_mae": log_val_mae,
                         "test_mae": log_test_mae,
                         "train_rmse": log_train_rmse,
                         "val_rmse": log_val_rmse,
                         "test_rmse": log_test_rmse,
                         "train_r_squared": log_train_r_squared,
                         "val_r_squared": log_val_r_squared,
                         "test_r_squared": log_test_r_squared
                         }
        self._training_information = pd.DataFrame(data=training_info)
        self._trained = True
        return self.model


class GPTrainer(BaseTrainer):
    def __init__(self, model, optimizer, likelihood, scheduler=None):
        super().__init__(model, optimizer)
        self._likelihood = likelihood
        self._scheduler = scheduler
        self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.device = torch.device("cpu")
        #print("Using " + str(self.device) + ".")
        self._model = model.to(self.device)

    def _train(self, x_train, train_targets):
        self._model.train(True)
        self._likelihood.train(True)
        self._optimizer.zero_grad()
        output = self._likelihood(self._model(x_train))
        loss = torch.sum(-self._mll(output, train_targets))
        loss.backward()
        self._optimizer.step()


    def training_info(self):
        """
        Getter for training information.
        """
        if not self._trained:
            print("You need to train the model before you get the training information.")
            return 0
        else:
            return self._training_information


    def run(self, x_train, train_targets, x_val, val_targets, x_test, test_targets, n_epochs=100):

         log_epoch = []
         log_lr = []
         log_loss = []

         log_train_mae = []
         log_val_mae = []
         log_test_mae = []

         log_train_rmse = []
         log_val_rmse = []
         log_test_rmse = []

         log_train_r_squared = []
         log_val_r_squared = []
         log_test_r_squared = []

         best_val_error = None
         iterator = tqdm.tqdm(range(n_epochs))
         for epoch in iterator:
             self._train(x_train, train_targets)
             # get learning rate from scheduler
             if self._scheduler is not None:
                 lr = self._scheduler.optimizer.param_groups[0]['lr']

             self._model.train(False)
             self._likelihood.eval()
             # gpytorch.settings.usetoeplitz(False)
             with torch.no_grad(),gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
                 val_predictions = self._likelihood(self._model(x_val))
                 train_predictions = self._likelihood(self._model(x_train))
                 # Mean absolute error
                 val_error = torch.mean(torch.abs(val_predictions.mean - val_targets))
                 if best_val_error is None or val_error <= best_val_error:
                    test_predictions = self._likelihood(self._model(x_test))
                    best_model = self._model
                    best_likelihood = self._likelihood
                    best_val_error = val_error

             iterator.set_postfix(loss = val_error)
             if self._scheduler is not None:
                 self._scheduler.step(val_error)

             # Logging training data
             log_epoch.append(epoch+1)
             log_lr.append(lr)
             log_train_mae.append(self._mae(train_predictions.mean.numpy(), train_targets.numpy()))
             log_val_mae.append(self._mae(val_predictions.mean.numpy(), val_targets.numpy()))
             log_test_mae.append(self._mae(test_predictions.mean.numpy(), test_targets.numpy()))

             log_train_rmse.append(self._rmse(train_predictions.mean.numpy(), train_targets.numpy()))
             log_val_rmse.append(self._rmse(val_predictions.mean.numpy(), val_targets.numpy()))
             log_test_rmse.append(self._rmse(test_predictions.mean.numpy(), test_targets.numpy()))

             log_train_r_squared.append(self._r_squared(train_predictions.mean.numpy(), train_targets.numpy()))
             log_val_r_squared.append(self._r_squared(val_predictions.mean.numpy(), val_targets.numpy()))
             log_test_r_squared.append(self._r_squared(test_predictions.mean.numpy(), test_targets.numpy()))

         training_info = {"epoch": log_epoch,
                          "learning_rate": log_lr,
                          "train_mae": log_train_mae,
                          "val_mae": log_val_mae,
                          "test_mae": log_test_mae,
                          "train_rmse": log_train_rmse,
                          "val_rmse": log_val_rmse,
                          "test_rmse": log_test_rmse,
                          "train_r_squared": log_train_r_squared,
                          "val_r_squared": log_val_r_squared,
                          "test_r_squared": log_test_r_squared
                          }
         self._training_information = pd.DataFrame(data=training_info)
         self._trained = True
         return best_model


class DKLGPTrainer(BaseTrainer):
    def __init__(self, model, optimizer, likelihood, scheduler=None, gradient_accumulation_splits=1):
        super().__init__(model, optimizer)
        self._likelihood = likelihood
        self._scheduler = scheduler
        self._gradient_accumulation_splits = gradient_accumulation_splits
        self._trained = False

    def _train(self, train_loader):

        """Performs a full training step. Depending on the setting for gradient accumulation, performs backward pass only
           every n batch.

        Returns:
            float: The obtained training loss.
        """

        self._model.train()
        loss_all = 0

        for batch_idx, batch in enumerate(train_loader):

            # transfer to device
            batch_x = batch[0].to(self.device)
            batch_y = batch[1].view(-1).to(self.device)

            loss = F.mse_loss(self._model(batch_x), batch_y)
            loss.backward()
            loss_all += loss.item() * len(batch[0])

            # gradient accumulation
            if ((batch_idx + 1) % self._gradient_accumulation_splits == 0) or (batch_idx + 1 == len(train_loader)):
                self._optimizer.step()
                self._optimizer.zero_grad()

        return loss_all / len(train_loader.dataset)

    def training_info(self):
        """
        Getter for training information.
        """
        if not self._trained:
            print("You need to train the model before you get the training information.")
            return 0
        else:
            return self._training_information

    def run(self, train_loader, train_loader_unshuffled, val_loader, test_loader, n_epochs=300, target_means=0, target_stds=1):

        """Runs a full training loop with metric ltraining_proceduresogging.

        Args:
            train_loader (Dataloader): The dataloader for the training points.
            train_loader_unshuffled (Dataloader): The dataloader for the training points but unshuffled. This is used to calculate metrics on the training set.
            val_loader (Dataloader): The dataloader for the validation points.
            test_loader (Dataloader): The dataloader for the test points.
            n_epochs (int): The number of epochs to perform.
            target_means(np.array): An array of the target means from standard scaling.
            target_stds(np.array): An array of the target stds from standard scaling.

        Returns:
            model: The trained model.
        """
        log_epoch = []
        log_lr = []
        log_loss = []

        log_train_error = []
        log_val_error = []
        log_test_error = []

        log_train_mae = []
        log_val_mae = []
        log_test_mae = []

        log_train_rmse = []
        log_val_rmse = []
        log_test_rmse = []

        log_train_r_squared = []
        log_val_r_squared = []
        log_test_r_squared = []

        # get targets off all sets
        train_targets = get_target_list(train_loader_unshuffled, target_means=target_means, target_stds=target_stds)
        val_targets = get_target_list(val_loader, target_means=target_means, target_stds=target_stds)
        test_targets = get_target_list(test_loader, target_means=target_means, target_stds=target_stds)

        best_val_error = None
        for epoch in range(1, n_epochs + 1):

            # get learning rate from scheduler
            if self._scheduler is not None:
                lr = self._scheduler.optimizer.param_groups[0]['lr']

            # training step
            loss = self._train(train_loader)

            # get predictions for all sets
            train_predictions = np.array(self.predict_loader_DKLGP(train_loader_unshuffled, target_means=target_means, target_stds=target_stds))
            val_predictions = np.array(self.predict_loader_DKLGP(val_loader, target_means=target_means, target_stds=target_stds))
            test_predictions = np.array(self.predict_loader_DKLGP(test_loader, target_means=target_means, target_stds=target_stds))

            train_error = self._mae(train_targets, train_predictions)
            val_error = self._mae(val_targets, val_predictions)

            # learning rate scheduler step
            if self._scheduler is not None:
                self._scheduler.step(val_error)

            # retain early stop test error
            if best_val_error is None or val_error <= best_val_error:
                test_error = self._mae(test_targets, test_predictions)
                best_val_error = val_error

            output_line = f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, 'f'Train MAE: {train_error:.7f}, 'f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}'
            #print(output_line)


            # Logging training data
            log_epoch.append(epoch)
            log_lr.append(lr)
            log_loss.append(loss)

            log_train_error.append(train_error)
            log_val_error.append(val_error)
            log_test_error.append(test_error)

            log_train_mae.append(self._mae(train_predictions, train_targets))
            log_val_mae.append(self._mae(val_predictions, val_targets))
            log_test_mae.append(self._mae(test_predictions, test_targets))

            log_train_rmse.append(self._rmse(train_predictions, train_targets))
            log_val_rmse.append(self._rmse(val_predictions, val_targets))
            log_test_rmse.append(self._rmse(test_predictions, test_targets))

            log_train_r_squared.append(self._r_squared(train_predictions, train_targets))
            log_val_r_squared.append(self._r_squared(val_predictions, val_targets))
            log_test_r_squared.append(self._r_squared(test_predictions, test_targets))

        training_info = {"epoch": log_epoch,
                         "learning_rate": log_lr,
                         "loss": log_loss,
                         "train_error": log_train_error,
                         "val_error": log_val_error,
                         "test_error": log_test_error,
                         "train_mae": log_train_mae,
                         "val_mae": log_val_mae,
                         "test_mae": log_test_mae,
                         "train_rmse": log_train_rmse,
                         "val_rmse": log_val_rmse,
                         "test_rmse": log_test_rmse,
                         "train_r_squared": log_train_r_squared,
                         "val_r_squared": log_val_r_squared,
                         "test_r_squared": log_test_r_squared
                         }
        self._training_information = pd.DataFrame(data=training_info)
        self._trained = True
        return self.model

    def predict_batch_DKLGP(self, batch, target_means=0, target_stds=1):

        """Makes predictions on a given batch.

        Returns:
            list: The predictions.
        """

        self._model.eval()
        batch_x = batch[0].to(self.device)

        # get predictions for batch
        predictions = (self.model(batch_x).means.cpu().detach().numpy() * target_stds + target_means).tolist()

        return predictions

    def predict_loader_DKLGP(self, loader, target_means=0, target_stds=1):

        """Makes predictions on a given dataloader.

        Returns:
            list: The predictions.
        """

        predictions = []

        for batch in loader:
            predictions.extend(self.predict_batch(batch, target_means=target_means, target_stds=target_stds))

        return predictions
