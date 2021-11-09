from collections import defaultdict
import numpy as np
import pickle
import torch


def loadTrainValidTest(trainValidTestFN, userId):
    with open(trainValidTestFN, "rb") as f:
        trainSet, validSet, testSet, contextMap = pickle.load(f)
    return trainSet[userId], validSet[userId], testSet[userId], contextMap


def workout2index(processed_data):
    wid2idx = {}
    for i in range(len(processed_data)):
        wid = processed_data[i]['id']
        wid2idx[wid] = i
    return wid2idx


def dataIteratorSupervised(processed_data, wid2idx, userId, allTrain, allValid, allTest, contextMap, isTrainValidTest, inputAtts, targetAtts, includeTemporal, num_steps):
    # load train, valid, test wids from pickles
    input_dim = len(inputAtts)
    output_dim = 1
    targetAtts = 'tar_' + targetAtts[0]

    if isTrainValidTest == 'train':
        indices = allTrain[userId]
    elif isTrainValidTest == 'valid':
        indices = allValid[userId]
    elif isTrainValidTest == 'test':
        indices = allTest[userId]
    else:
        raise (Exception("invalid dataset type: must be 'train', 'valid', or 'test'"))
    # loop each data point
    for wid in indices:
        current_input = processed_data[wid2idx[wid]]
        num_steps = len(current_input['distance'])  # num_steps = 500
        
        inputs = np.zeros([input_dim, num_steps]) 
        outputs = np.zeros([output_dim, num_steps]) 
        for att_idx, att in enumerate(inputAtts):
            if att == 'time_elapsed':
                inputs[att_idx, :] = np.ones([1, num_steps]) * current_input[att][num_steps-1] # given the total workout length
                # inputs[att_idx, :] = current_input[att][:num_steps] # given the total workout length
            else:
                inputs[att_idx, :] = current_input[att][:num_steps]

        outputs[0, :] = current_input[targetAtts][:num_steps]
        inputs = np.transpose(inputs)
        outputs = np.transpose(outputs)

        trimmed_workout_len = num_steps
        # build context input
        if includeTemporal:
            context_wid = contextMap[wid][2][-1]   # previous workout id
            context_input = processed_data[wid2idx[context_wid]]

            context_since_last = np.ones([1, trimmed_workout_len]) * contextMap[wid][0] # [1,500]
            context_inputs = np.zeros([input_dim, trimmed_workout_len])
            context_outputs = np.zeros([output_dim, trimmed_workout_len])
            for att_idx, att in enumerate(inputAtts):
                if att == 'time_elapsed':
                    context_inputs[att_idx, :] = np.ones([1, trimmed_workout_len]) * context_input[att][trimmed_workout_len-1]
                else:
                    context_inputs[att_idx, :] = context_input[att][:trimmed_workout_len]

            context_outputs[0, :] = context_input[targetAtts][:trimmed_workout_len]
            context_input_1 = np.transpose(np.concatenate([context_inputs, context_since_last], axis=0))
            context_input_2 = np.transpose(context_outputs)

        inputs_dict = {'input':inputs}
        if includeTemporal:
            inputs_dict['context_input_1'] = context_input_1
            inputs_dict['context_input_2'] = context_input_2

        yield (inputs_dict, outputs, wid)


def generator_for_autotrain(processed_data, wid2idx, userId, allTrain, allValid, allTest, contextMap, isTrainValidTest, inputAtts, targetAtts, includeTemporal, batch_size, num_steps):
    input_dim = len(inputAtts)
    output_dim = 1
    batchGen = dataIteratorSupervised(processed_data, wid2idx, userId, allTrain, allValid, allTest, contextMap, isTrainValidTest, inputAtts, targetAtts, includeTemporal, num_steps)

    if isTrainValidTest=="train":
        data_len = len(allTrain[userId])
    elif isTrainValidTest=="valid":
        data_len = len(allValid[userId])
    elif isTrainValidTest=="test":
        data_len = len(allTest[userId])
    else:
        raise(ValueError("isTrainValidTest is not a valid value"))

    if data_len%batch_size==0:
        num_batches = int(data_len / batch_size)
    else:
        num_batches = int(data_len / batch_size) + 1

    for i in range(num_batches):
        if (data_len%batch_size!=0) & (i==num_batches-1):
            batch_size=data_len%batch_size
        inputs = np.zeros([batch_size, num_steps, input_dim])
        outputs = np.zeros([batch_size, num_steps, output_dim])
        workoutids = np.zeros([batch_size])

        if includeTemporal:
            context_input_1 = np.zeros([batch_size, num_steps, input_dim + 1])  # + since last
            context_input_2 = np.zeros([batch_size, num_steps, output_dim])

        inputs_dict = {'input': inputs, 'workoutid': workoutids}
        for j in range(batch_size):
            current = next(batchGen)
            inputs[j,:,:] = current[0]['input']
            outputs[j,:,:] = current[1]
            workoutids[j] = current[2]
            
            if includeTemporal:
                context_input_1[j,:,:] = current[0]['context_input_1']
                context_input_2[j,:,:] = current[0]['context_input_2']
                inputs_dict['context_input_1'] = context_input_1
                inputs_dict['context_input_2'] = context_input_2
        yield (inputs_dict, outputs)


def RMSE(output, target):
    loss = torch.mean((output-target)**2)
    loss = torch.sqrt(loss)
    return loss


def clip_grad_norm_(parameters, max_norm, norm_type):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm