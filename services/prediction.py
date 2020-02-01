import copy
import os

import numpy as np
import torch
import torch.backends.cudnn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from torch.autograd.variable import Variable

from metadata.ITrackerModel import ITrackerModel


def load_checkpoint(filename):
    filename = os.path.join(CHECKPOINTS_PATH, filename)

    if not os.path.isfile(filename):
        return None
    else:
        return torch.load(filename, map_location='cpu')


def initialize_model(model, filename):
    model = torch.nn.DataParallel(model)
    model.cuda()
    torch.backends.cudnn.benchmark = True
    saved = load_checkpoint(filename)

    if saved:
        state = saved['state_dict']
        try:
            model.module.load_state_dict(state)
        except:
            model.load_state_dict(state)
    else:
        print('Warning: Could not read checkpoint!')


def extract_feature(val_loader, model):
    feature_extraction_model = copy.deepcopy(model)
    new_fc = torch.nn.Sequential(*list(model.fc.children())[:-1])
    feature_extraction_model.fc = new_fc

    # switch to evaluate mode
    model.eval()

    feature_list = list()
    act_list = list()
    for local_batch, local_labels in val_loader:
        imFace = (local_batch[0]).to(DEVICE).permute(0, 3, 1, 2).float().cuda()
        imEyeL = (local_batch[1]).to(DEVICE).permute(0, 3, 1, 2).float().cuda()
        imEyeR = (local_batch[2]).to(DEVICE).permute(0, 3, 1, 2).float().cuda()
        faceGrid = (local_batch[3]).to(DEVICE).float().cuda()
        gaze = torch.t(torch.stack(local_labels).to(DEVICE).float()).cuda()
        # imFace = (local_batch[0]).to(DEVICE).permute(0, 3, 1, 2).float()
        # imEyeL = (local_batch[1]).to(DEVICE).permute(0, 3, 1, 2).float()
        # imEyeR = (local_batch[2]).to(DEVICE).permute(0, 3, 1, 2).float()
        # faceGrid = (local_batch[3]).to(DEVICE).float()
        # gaze = torch.t(torch.stack(local_labels).to(DEVICE).float())

        imFace = Variable(imFace, requires_grad=False)
        imEyeL = Variable(imEyeL, requires_grad=False)
        imEyeR = Variable(imEyeR, requires_grad=False)
        faceGrid = Variable(faceGrid, requires_grad=False)
        gaze = Variable(gaze, requires_grad=False)

        # compute output
        with torch.no_grad():
            output = feature_extraction_model(imFace, imEyeL, imEyeR, faceGrid)

        feature_list.append(output.cpu().numpy())
        act_list.append(gaze.cpu().numpy())
    features = np.concatenate(feature_list)
    y = np.concatenate(act_list)

    return features, y


def calculate_xy_error(act_x, predict_x, act_y, predict_y):
    error_x = (act_x - predict_x) ** 2
    error_y = (act_y - predict_y) ** 2
    error = (error_x + error_y) ** 0.5

    return error


CHECKPOINTS_PATH = 'metadata'
DEVICE = torch.device('cuda:0')
# DEVICE = torch.device('cpu')
PARAMS = {'batch_size': 20, 'shuffle': False, 'num_workers': 2}
MODEL = ITrackerModel()
initialize_model(MODEL, 'new_check_point.pth.tar')
REGRO = None
REGR1 = None


class Predictor:
    @staticmethod
    def svr_predict(test_generator):
        global REGRO, REGR1

        if REGRO is None and REGR1 is None:
            print('Please calibrate before using the SVR prediction')
            return None

        features, y = extract_feature(test_generator, MODEL)
        predictions = np.zeros((len(y[:, 0]), 2))
        predictions[:, 0] = REGRO.predict(features)

        features, y = extract_feature(test_generator, MODEL)
        predictions[:, 1] = REGR1.predict(features)

        return predictions

    @staticmethod
    def predict(test_generator):
        criterion = torch.nn.MSELoss().cuda()
        # criterion = torch.nn.MSELoss()

        predictions, y = Predictor.__process_predict(test_generator, MODEL, criterion)

        return predictions

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
            # image_face = (local_batch[0]).to(DEVICE).permute(0, 3, 1, 2).float()
            # image_left_eye = (local_batch[1]).to(DEVICE).permute(0, 3, 1, 2).float()
            # image_right_eye = (local_batch[2]).to(DEVICE).permute(0, 3, 1, 2).float()
            # face_grid = (local_batch[3]).to(DEVICE).float()
            # gaze = torch.t(torch.stack(local_labels).to(DEVICE).float())

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


class Calibrator:
    @staticmethod
    def calibrate(training_generator, validation_generator):
        global REGRO, REGR1

        train_length = 5 * 26
        valid_length = 4 * 26

        search_range = [10.0 ** i for i in np.arange(-4, 5)]
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': search_range, 'C': search_range},
                            {'kernel': ['linear'], 'C': search_range}]
        regr = SVR()

        features1, y1 = extract_feature(training_generator, MODEL)
        features2, y2 = extract_feature(validation_generator, MODEL)

        features = np.concatenate([features1, features2], axis=0)
        y = np.concatenate([y1, y2], axis=0)

        REGRO = GridSearchCV(regr, tuned_parameters,
                             cv=[(np.arange(train_length), np.arange(train_length, train_length + valid_length))])
        REGRO.fit(features, y[:, 0])

        REGR1 = GridSearchCV(regr, tuned_parameters,
                             cv=[(np.arange(train_length), np.arange(train_length, train_length + valid_length))])
        REGR1.fit(features, y[:, 1])

        features, y = extract_feature(training_generator, MODEL)
        predictions = np.zeros((len(y[:, 0]), 2))
        predictions[:, 0] = REGRO.predict(features)
        predictions[:, 1] = REGR1.predict(features)

        ft_err_list = calculate_xy_error(y[:, 0], predictions[:, 0], y[:, 1], predictions[:, 1])
        print('Training: ' + str(np.mean(ft_err_list)))

        features, y = extract_feature(validation_generator, MODEL)
        predictions = np.zeros((len(y[:, 0]), 2))
        predictions[:, 0] = REGRO.predict(features)
        predictions[:, 1] = REGR1.predict(features)

        ft_err_list = calculate_xy_error(y[:, 0], predictions[:, 0], y[:, 1], predictions[:, 1])
        print('Validation: ' + str(np.mean(ft_err_list)))


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
