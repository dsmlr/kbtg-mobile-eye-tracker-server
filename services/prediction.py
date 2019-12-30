import os

import numpy as np
import torch
import torch.backends.cudnn
from torch.autograd.variable import Variable

from metadata.ITrackerModel import ITrackerModel

CHECKPOINTS_PATH = 'metadata'
DEVICE = torch.device("cuda:0")
PARAMS = {'batch_size': 20, 'shuffle': False, 'num_workers': 2}


class Predictor:
    @staticmethod
    def predict(test_generator):
        model = ITrackerModel()
        Predictor.__initialize_model(model, 'checkpoint.pth.tar')
        criterion = torch.nn.MSELoss().cuda()

        predictions, y = Predictor.__process_predict(test_generator, model, criterion)

        feature_error_list = Predictor.__calculate_xy_error(predictions[:, 0], y[:, 1], predictions[:, 1])

        return 'Fine-tuned new base for meta-learning: ' + str(np.mean(feature_error_list))

    @staticmethod
    def __initialize_model(model, filename):
        model = torch.nn.DataParallel(model)
        model.cuda()
        torch.backends.cudnn.benchmark = True
        checkpoint = Predictor.__load_checkpoint(filename)

        if checkpoint:
            state = checkpoint['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
        else:
            print('Warning: Could not read checkpoint!')

    @staticmethod
    def __load_checkpoint(filename):
        checkpoint = os.path.join(CHECKPOINTS_PATH, filename)

        if not os.path.isfile(checkpoint):
            return None
        else:
            return torch.load(checkpoint, map_location='cpu')

    @staticmethod
    def __process_predict(val_loader, model, criterion):
        losses = AverageMeter()
        losses_lin = AverageMeter()

        model.eval()

        prediction_list = list()
        act_list = list()

        for local_batch, local_labels in val_loader:
            image_face = (local_batch[0]).to(DEVICE).permute(0, 3, 1, 2).float().cuda()
            image_left_eye = (local_batch[1]).to(DEVICE).permute(0, 3, 1, 2).float().cuda()
            image_right_eye = (local_batch[2]).to(DEVICE).permute(0, 3, 1, 2).float().cuda()
            face_grid = (local_batch[3]).to(DEVICE).float().cuda()
            gaze = torch.t(torch.stack(local_labels).to(DEVICE).float()).cuda()

            image_face = Variable(image_face, requires_grad=False)
            image_left_eye = Variable(image_left_eye, requires_grad=False)
            image_right_eye = Variable(image_right_eye, requires_grad=False)
            face_grid = Variable(face_grid, requires_grad=False)
            gaze = Variable(gaze, requires_grad=False)

            with torch.no_grad():
                output = model(image_face, image_left_eye, image_right_eye, face_grid)

            loss = criterion(output, gaze)

            act_list.append(gaze.cpu().numpy())
            prediction_list.append(output.cpu().numpy())

            loss_lin = output - gaze
            loss_lin = torch.mul(loss_lin, loss_lin)
            loss_lin = torch.sum(loss_lin, 1)
            loss_lin = torch.mean(torch.sqrt(loss_lin))

            losses.update(loss.data.item(), image_face.size(0))
            losses_lin.update(loss_lin.item(), image_face.size(0))

        predictions = np.concatenate(prediction_list)
        y = np.concatenate(act_list)

        return predictions, y

    @staticmethod
    def __calculate_xy_error(act_x, predict_x, act_y, predict_y):
        error_x = (act_x - predict_x) ** 2
        error_y = (act_y - predict_y) ** 2
        error = (error_x + error_y) ** 0.5

        return error


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
