import torch

class CallBacks:
    """
    CallBack is a class of functions related to callbacks used in deep learning.

    Callbacks are functions or methods that are executed at specific points during training
    to perform various tasks, such as saving model weights or implementing early stopping.
    """

    @staticmethod
    def saveModel(model):
        """
        Save the model's weights to a file.

        parameters:
        -------------
            model (torch.nn.Module): The PyTorch model whose weights need to be saved.

        Returns:
        ----------

        """
        savePath = "/content/drive/MyDrive/ZamanPersonalUsage/vesselSegmentation/models/BestModel.pth"
        torch.save(model.state_dict(), savePath)

    @staticmethod
    def earlyStopping(bestIOU, IOU, patience, model):
        """
        Implement early stopping based on a validation metric.

        Early stopping monitors a validation metric and stops training when the metric stops improving.
        This prevents overfitting and saves training time.

        parameters:
        ------------
            bestIOU (float): The best IOU (Jaccard Index) obtained so far during training.
            IOU (float): The current IOU obtained on the validation set.
            patience (int): The number of epochs with no improvement in the IOU before stopping.
            model (torch.nn.Module): The PyTorch model being trained.

        Returns
        ----------------
            Tuple: [float, int]
            A tuple containing the updated bestIOU and patience values.

        ```
        """
        if IOU > bestIOU:
            bestIOU = IOU
            patience = 0
            CallBacks.saveModel(model)
        else:
            patience += 1
        return bestIOU, patience
