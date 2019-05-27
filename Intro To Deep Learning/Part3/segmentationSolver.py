from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(ignore_index=-1)):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def calculate_loss(self,X,Y,oneStep=500):
        size = X.shape[0]

        totalSteps = int(size/oneStep)
        lossArray = np.zeros(totalSteps)

        for i in range(0,totalSteps):
            X_batch = torch.from_numpy(X[i*oneStep:(i+1)*oneStep])
            y_batch = torch.from_numpy(Y[i*oneStep:(i+1)*oneStep])

            output = self.model(Variable(X_batch))
            lossArray[i] = self.loss_func(output,Variable(y_batch)).data.numpy()

        return np.mean(lossArray)

    def _step(self,X,Y):
        self.optimizer.zero_grad()
        output = self.model(Variable(X))
        loss = self.loss_func(output,Variable(Y))
        returnData = loss.data.numpy()
        loss.backward()
        self.optimizer.step()
        return returnData

    def train(self, model, train_loader, val_loader, num_epochs=2, log_nth=10):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        self.optimizer = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        
        self.model = model

        num_iterations = num_epochs * iter_per_epoch

        for t in range(num_iterations):
            X = train_loader[t%iter_per_epoch][0]
            X.unsqueeze_(0)
            Y = train_loader[t%iter_per_epoch][1]
            Y.unsqueeze_(0)

            lossData = self._step(X,Y)
            
            self.train_loss_history.append(lossData)
            if (t % log_nth == 0):
                print('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, lossData))

        print('Final Training Loss')
        #print(self.calculate_loss(self.X_train,self.y_train))
        print('Final Validation Loss')
        #print(self.calculate_loss(self.X_val,self.y_val))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
