import random
from collections import defaultdict, OrderedDict
import pickle
import numpy as np
import torch
from tqdm import trange

from experiments.pfedhn.HRFL_target import LSTMTarget
from experiments.pfedhn.HRFL_hyper import LSTMHyper
from HRFL_utils import generator_for_autotrain, workout2index
from HRFL_utils import clip_grad_norm_, RMSE

use_cuda = torch.cuda.is_available()
print("Is CUDA available? ", use_cuda)
cuda0 = torch.device('cuda:0')
# cuda1 = torch.device('cuda:1')


@torch.no_grad()
def evaluate(hnet, net, all_userId, train_batches_dict, valid_batches_dict, test_batches_dict, split='valid'):
    num_nodes = len(all_userId)
    hnet.eval()
    running_loss = 0.

    if split == 'valid':
        data = valid_batches_dict
    elif split == 'test':
        data = test_batches_dict
    else:
        data = train_batches_dict

    for node_idx in range(num_nodes):
        node_loss = 0.
        weights = hnet(torch.LongTensor([node_idx]).to(cuda0))
        net.load_state_dict(weights)
        userId = all_userId[node_idx]
        batches = data[userId]
        for batch in batches:
            inputs, target = net.embed_inputs(batch)
            inputs = inputs.float().to(cuda0)
            out = net.forward(inputs)
            node_loss += RMSE(out, target.to(cuda0))
        running_loss += node_loss
    avg_loss = running_loss / num_nodes
    return avg_loss



def train(country, optim, steps, inner_steps, lr, inner_lr, embed_lr, wd, inner_wd, hyper_hid, inner_hid, embed_dim, n_hidden, batch_size, eval_every, random_state=12, use_cuda=True) -> None:
    loss_vector = []

    # import all userId in the country
    pickle_name = '../../../country_data/pkl/FL_data/' + country + '_FL.pkl'
    with open(pickle_name, "rb") as pkl:
        allTrain, allValid, allTest, contextMap = pickle.load(pkl)

    all_userId = list(allTrain.keys())
    userId_to_idx = defaultdict()
    for idx, userId in enumerate(all_userId):
        userId_to_idx[userId] = idx


    # import country dataset
    processedFN = '../../../country_data/processed_fitrec_' + country + '.npy'
    processed_data = np.load(processedFN, allow_pickle=True)[0]

    wid2idx = workout2index(processed_data)

    # number of nodes
    num_nodes = len(all_userId)
    print('Total number of users = {}'.format(num_nodes))

    # create hypernet
    print('Initializing HyperNetwork')
    hnet = LSTMHyper(input_dim=3, output_dim=1, hidden_dim=hyper_hid, embed_dim=embed_dim, context_final_dim=32, num_nodes=num_nodes, n_hidden=n_hidden, lr=lr, wd=wd,spec_norm=False, use_cuda=use_cuda)

    print('Initializing target network')
    net = LSTMTarget(country=country, userId=userId, inputAtts=['distance','altitude','time_elapsed'], targetAtts=['heart_rate'], includeTemporal=True, hidden_dim=inner_hid, context_final_dim=32, lr=inner_lr, max_local_epochs=inner_steps, weight_decay=inner_wd, use_cuda=use_cuda)

    hnet.to(cuda0)
    net.to(cuda0)


    # load data for all users
    train_batches_dict = defaultdict(list)
    valid_batches_dict = defaultdict(list)
    test_batches_dict = defaultdict(list)
    for user_idx in range(num_nodes):
        userId = all_userId[user_idx]
        print(f'Loading batches for user {userId} ({user_idx+1}/{num_nodes})')
        userTrainBatch = generator_for_autotrain(processed_data=processed_data, wid2idx=wid2idx, userId=userId, allTrain=allTrain, allValid=allValid, allTest=allTest, contextMap=contextMap, inputAtts=['distance','altitude','time_elapsed'], targetAtts=['heart_rate'], includeTemporal=True, batch_size=batch_size, num_steps=500, isTrainValidTest='train')

        userValidBatch = generator_for_autotrain(processed_data=processed_data, wid2idx=wid2idx, userId=userId, allTrain=allTrain, allValid=allValid, allTest=allTest, contextMap=contextMap, inputAtts=['distance','altitude','time_elapsed'], targetAtts=['heart_rate'], includeTemporal=True, batch_size=batch_size, num_steps=500, isTrainValidTest='valid')

        userTestBatch = generator_for_autotrain(processed_data=processed_data, wid2idx=wid2idx, userId=userId, allTrain=allTrain, allValid=allValid, allTest=allTest, contextMap=contextMap, inputAtts=['distance','altitude','time_elapsed'], targetAtts=['heart_rate'], includeTemporal=True, batch_size=batch_size, num_steps=500, isTrainValidTest='test')

        train_batches_dict[userId] = list(userTrainBatch)
        valid_batches_dict[userId] = list(userValidBatch)
        test_batches_dict[userId] = list(userTestBatch)

        print(f'    number of train/valid/test batches = {len(train_batches_dict[userId])}/{len(valid_batches_dict[userId])}/{len(test_batches_dict[userId])}')


    # define optimizer for the hypernet (hnet)
    embed_lr = embed_lr if embed_lr is not None else lr
    optimizers = optimizers = {
        'RMSprop': torch.optim.RMSprop(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr, weight_decay=wd
        ),
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr, momentum=0.9, weight_decay=wd
        ),
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=lr)
    }

    hnet_optimizer = optimizers[optim]
    step_iter = trange(steps)

    # in each step, randomly choose on user to train the hnet
    for s in step_iter:
        hnet.train()
        user_idx = random.choice(range(num_nodes))
        userId = all_userId[user_idx]

        # load train/valid/test batches
        userTrainBatch = train_batches_dict[userId]
        userValidBatch = valid_batches_dict[userId]
        userTestBatch = test_batches_dict[userId]

        weights = hnet(torch.LongTensor([user_idx]).to(cuda0))
        net.load_state_dict(weights)
        
        inner_optimizer = torch.optim.RMSprop(
            [
                {'params': [p for n, p in net.named_parameters()]}
            ], lr=inner_lr, weight_decay=inner_wd
        )

        # theta_i, for updating later
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        # evaluate the current local model
        # with torch.no_grad():
        #     net.eval()
        #     prvs_loss = 0.
        #     for batch in userValidBatch:
        #         inputs, target = net.embed_inputs(batch)
        #         inputs = inputs.float().to(cuda0)
        #         out = net.forward(inputs)
        #         prvs_loss += RMSE(out, target.to(cuda0))


        # train target network -> obtaining theta_tilde
        for epoch in range(inner_steps):
            net.train()
            for batch in userTrainBatch:
                inputs, target = net.embed_inputs(batch)
                inputs = inputs.float().to(cuda0)
                out = net.forward(inputs)
                inner_optimizer.zero_grad()
                hnet_optimizer.zero_grad()
                loss = RMSE(out, target.to(cuda0))
                loss.backward()
                clip_grad_norm_(parameters=net.parameters(), max_norm=.5, norm_type=2.0)
                inner_optimizer.step()

            if (epoch+1) % eval_every == 0 or epoch==0:
                step_iter.set_description(
                    f"Step: {s+1}, NodeId: {user_idx}, UserId: {userId}, Epoch: {epoch+1}, Training loss: {loss.item():.4f}"
                )

        hnet_optimizer.zero_grad()

        final_state = net.state_dict()

        # calculate delta theta
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        # calculate phi gradient
        hnet_grads = torch.autograd.grad(
            list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
        )

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g
        
        clip_grad_norm_(parameters=hnet.parameters(), max_norm=.5, norm_type=2.0)
        hnet_optimizer.step()

        # print evaluate every N steps
        # if s==0 or (s+1) % eval_every == 0:
        valid_loss = evaluate(hnet, net, all_userId, train_batches_dict, valid_batches_dict, test_batches_dict, split='valid')
        loss_vector.append(valid_loss.cpu().numpy().item())
        print('---Step: %d, Validation loss: %1.5f' % (s+1, valid_loss))
    

    # test here, after all test
    test_loss = 0
    for userId in all_userId:
        test_loss += evaluate(hnet, net, all_userId, train_batches_dict, valid_batches_dict, test_batches_dict, split='test')
    avg_test_loss = test_loss/num_nodes
    print('*'*25 + 'Final test loss' + '*'*25)
    print('After: %d steps, final test loss: %1.5f' % (s+1, avg_test_loss))

    hnet_fn = './results/hnet_' + country + '.pt'
    valid_loss_fn = './results/valid_loss_' + country + '.pt'
    # torch.save(hnet.state_dict(), hnet_fn)
    # torch.save(loss_vector, valid_loss_fn)
    print('All validation loss:')
    print(loss_vector)
    print('Results saved. Complete!')


    fixed_weights = hnet(torch.LongTensor([0]).to(cuda0))
    benign_weights = OrderedDict({k: tensor.data for k, tensor in fixed_weights.items()})
    
    norm = 0
    for k in benign_weights.keys():
        norm += (benign_weights[k].cpu().numpy()**2).sum()
        norm = np.sqrt(norm)
    print(f'L2 norm = {norm}')

    
if __name__ == '__main__':

    train(
        country='France',
        optim='adam',
        steps=500,   #5000
        inner_steps=50,    # max local epochs
        lr=0.025,   #1e-2
        inner_lr=0.0075,
        embed_lr=None,
        wd=0.001,
        inner_wd=1e-4,
        hyper_hid=64,
        inner_hid=64,
        embed_dim=5,
        n_hidden=3,
        batch_size=128,
        eval_every=10,
        random_state=12,
        use_cuda=use_cuda
    )


'''
--------------------
+    TO-DO LIST    +
--------------------
1. 
'''