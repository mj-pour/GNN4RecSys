import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")


# ====== Tracking lists ======
loss_list = []
recall_list = []
precision_list = []
ndcg_list = []
# ============================

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        # ===== TEST =====
        if epoch % 10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

            # store metrics
            recall_list.append(results['recall'][0])
            precision_list.append(results['precision'][0])
            ndcg_list.append(results['ndcg'][0])

        # ===== TRAIN =====
        output_information = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
        )

        # Extract numeric loss from string
        loss_value = float(output_information.split('-')[0].replace('loss',''))
        loss_list.append(loss_value)

        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.tensorboard:
        w.close()


import matplotlib.pyplot as plt

# ===== Plot Loss Curve =====
plt.figure()
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("BPR Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()

# ===== Plot Evaluation Curves =====
if len(recall_list) > 0:
    plt.figure()
    plt.plot(range(0, world.TRAIN_epochs, 10), recall_list, label="Recall@10")
    plt.plot(range(0, world.TRAIN_epochs, 10), ndcg_list, label="NDCG@10")
    plt.plot(range(0, world.TRAIN_epochs, 10), precision_list, label="Precision@10")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.title("Evaluation Metrics Curve")
    plt.grid()
    plt.show()
