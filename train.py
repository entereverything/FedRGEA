import os
import sys
import copy
import math
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
try:
    # Prefer PyTorch built-in writer (more compatible with newer protobuf stacks)
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    # Fallback for older environments
    from tensorboardX import SummaryWriter  # type: ignore

from utils.FedAvg import FedAvg
from utils.fedHybrid_three_line import fedHybridThreeLine
from dataset.get_dataset import get_datasets
from val import compute_bacc, compute_global_prototypes_from_loader
from networks.networks import efficientb0, FedAvgResNet18, FedProxResNet18, FedFSAResNet18
from utils.local_training import LocalUpdate
from utils.utils import set_seed, classify_label
from utils.sample_dirichlet import clients_indices
from utils.immune import ImmuneDetector


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='brainTumor', help='dataset name', choices=['isic2019', 'ich', 'brainTumor', 'ham10000', 'bloodmnist'])
    parser.add_argument('--exp', type=str,
                        default='test', help='experiment name')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='batch_size per gpu (larger for stable gradients on imbalanced data)')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')
    parser.add_argument('--alpha', type=float,
                        default=1, help='parameter for non-iid')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--data_seed', type=int, default=0, help='random seed for data partition only')
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--local_ep', type=int,
                        default=1, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=100, help='rounds')
    parser.add_argument('--n_clients', type=int,  default=10, help='number of federated clients')
    parser.add_argument(
        '--local_train_mode',
        type=str,
        default='base',
        choices=[
            'base','train_dcrl'
        ],
    )
    parser.add_argument(
        '--client_attack_types',
        type=str,
        default='0,0,0,2,2,2,2,0,0,0',
        help='comma-separated attack types per client, 0=normal, 1=constant, 2=sign-flip, 3=random-gradients, 4=update-scaling, 5=IPM, 6=LIE',
    )

    # ------------------------------ immune detection (negative selection) ------------------------------
    parser.add_argument(
        '--use-immune-detection',
        action='store_true',
        default=True,
        help='whether to use self-calibrated immune detection before aggregation',
    )
    parser.add_argument(
        '--detection-threshold',
        type=float,
        default=0.8,
        help='immune detection strictness in [0,1]; higher is stricter',
    )
    parser.add_argument(
        '--csea-pop-size',
        type=int,
        default=8,
        help='population size for conflict-aware three-line aggregation',
    )
    parser.add_argument(
        '--csea-generations',
        type=int,
        default=10,
        help='number of search generations for conflict-aware three-line aggregation',
    )
    parser.add_argument(
        '--csea-prior-mode',
        type=str,
        default='reliability',
        choices=['uniform', 'reliability', 'sample_size'],
        help='aggregation prior for CSEA; reliability avoids relying on sample counts',
    )
    parser.add_argument(
        '--csea-prior-strength',
        type=float,
        default=0.15,
        help='strength of the CSEA prior regularizer',
    )
    parser.add_argument(
        '--csea-diversity-strength',
        type=float,
        default=0.02,
        help='strength of the CSEA concentration penalty',
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        default='fedHybridThreeLine5',
        choices=['fedavg', 'fedHybridThreeLine', 'fedHybridThreeLine1', 'fedHybridThreeLine2',
                 'fedHybridThreeLine3', 'fedHybridThreeLine4', 'fedHybridThreeLine5'],
        help='aggregation method: fedavg / fedHybridThreeLine / 1=Random / 2=GA / 3=PSO / 4=DE / 5=BACO',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args)

    # ------------------------------ output files ------------------------------
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    # Auto-increment experiment directory: <exp>1, <exp>2, ...
    base_exp_dir = os.path.join(outputs_dir, args.exp)
    run_id = 1
    exp_dir = f"{base_exp_dir}{run_id}"
    while os.path.exists(exp_dir):
        run_id += 1
        exp_dir = f"{base_exp_dir}{run_id}"
    os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)

    # ------------------------------ immune detector ------------------------------
    immune_detector = None
    if getattr(args, "use_immune_detection", False):
        immune_detector = ImmuneDetector(
            threshold=getattr(args, "detection_threshold", 0.8),
        )
        logging.info(
            f"[IMMUNE] enabled: threshold={immune_detector.threshold}, "
            f"min_clients={immune_detector.min_clients_for_filtering}, "
            f"max_reject_ratio={immune_detector.max_reject_ratio}, "
            f"reliability_momentum={immune_detector.reliability_momentum}"
        )

    # ------------------------------ dataset and dataloader ------------------------------
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    if args.local_train_mode == "base" and isinstance(train_dataset.transform, list):
        # Base training consumes only the first view, so avoid generating and
        # discarding the second augmented view for every sample.
        train_dataset.transform = train_dataset.transform[0]
        logging.info("[DATA] base mode: using one augmented training view")
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # ------------------------------ global and local settings ------------------------------
    n_classes = train_dataset.n_classes
    # if args.local_train_mode == "fedavg":
    #     net_glob = FedAvgResNet18(n_classes=n_classes, args=args).cuda()
    #     logging.info("[MODEL] FedAvg mode: ResNet-18 (ImageNet pretrained head replaced)")
    # else:
    #     net_glob = efficientb0(n_classes=n_classes, args=args).cuda()
    net_glob = efficientb0(n_classes=n_classes, args=args).cuda()
    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    user_id = list(range(args.n_clients))

    # ------------------------------ client attack types ------------------------------
    if args.client_attack_types:
        try:
            parsed_attack_types = [
                int(x) for x in args.client_attack_types.split(',') if x.strip() != ''
            ]
        except ValueError:
            parsed_attack_types = []
    else:
        parsed_attack_types = []

    # Pad / trim to n_clients, default 0 (benign)
    attack_types = (parsed_attack_types + [0] * args.n_clients)[:args.n_clients]
    logging.info(f'client attack types: {attack_types}')

    # Here, we follow CreFF (https://arxiv.org/abs/2204.13399).
    list_label2indices = classify_label(train_dataset.targets, n_classes)
    dict_users = clients_indices(list_label2indices, n_classes, args.n_clients, args.alpha, args.data_seed)
    dict_len = [len(dict_users[id]) for id in user_id]

    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(train_dataset), dict_users[id]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())

    # ------------------------------ begin training ------------------------------
    set_seed(args)
    best_performance = 0.
    best_test_acc = -1.0
    test_acc_path = os.path.join(logs_dir, 'test_acc_each_round.txt')
    best_test_acc_path = os.path.join(logs_dir, 'best_test_acc.txt')
    with open(test_acc_path, 'w', encoding='utf-8') as f:
        f.write('round\tacc\n')
    with open(best_test_acc_path, 'w', encoding='utf-8') as f:
        f.write('round\tbest_acc\n')
    lr = args.base_lr
    acc = []
    # Cosine LR decay to avoid late-stage collapse on imbalanced + non-IID data
    def lr_cosine(round_idx, total_rounds, base_lr):
        return base_lr * 0.5 * (1 + math.cos(math.pi * round_idx / max(1, total_rounds)))
    for com_round in range(args.rounds):
        lr = lr_cosine(com_round, args.rounds, args.base_lr)
        logging.info(f'\n======================> round: {com_round} <======================')
        loss_locals = []
        writer.add_scalar('train/lr', lr, com_round)

        w_glob_start = None

        # local training
        for id in user_id:
            trainer_locals[id].lr = lr
            local = trainer_locals[id]
            net_local = net_locals[id]
            if args.local_train_mode == 'train_dcrl':
                w, loss = local.train_dcrl(copy.deepcopy(net_local), writer)
            else:
                w, loss = local.train(copy.deepcopy(net_local), writer)
            # apply potential model poisoning attacks on local updates
            attack_type = attack_types[id]
            if attack_type in [1, 2, 3, 4]:
                w_attacked = copy.deepcopy(w)
                for k in w_attacked.keys():
                    if k not in w_glob:
                        continue
                    if not torch.is_floating_point(w_glob[k]):
                        continue
                    delta = w_attacked[k] - w_glob[k]
                    if attack_type == 1:
                        delta = torch.zeros_like(delta)
                    elif attack_type == 2:
                        delta = -delta
                    elif attack_type == 3:
                        # Random Gradients: send standard normal noise
                        delta = torch.randn_like(delta)
                    elif attack_type == 4:
                        delta = delta * 100.0
                    w_attacked[k] = w_glob[k] + delta
                w_locals[id] = w_attacked
                attack_name = {
                    1: "constant",
                    2: "sign-flip",
                    3: "random-gradients",
                    4: "update-scaling",
                }.get(attack_type, "unknown")
                logging.info(
                    f'Client {id}: applied {attack_name} attack'
                )
            elif attack_type in [5, 6]:
                # Store raw update first; coordinated attacks handled after loop
                w_locals[id] = copy.deepcopy(w)
            else:
                w_locals[id] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

        # Coordinated attacks (IPM, ALIE) — need honest-update statistics
        coordinated_ids = [id for id in user_id if attack_types[id] in [5, 6]]
        if coordinated_ids:
            honest_ids = [id for id in user_id if attack_types[id] == 0]
            if honest_ids:
                for cid in coordinated_ids:
                    attack_type = attack_types[cid]
                    w_attacked = copy.deepcopy(w_locals[cid])
                    if attack_type == 5:  # IPM — Inner Product Manipulation
                        # malicous delta = -λ * mean(honest deltas), opposes the true gradient
                        lambda_ipm = 1.0
                        for k in w_attacked.keys():
                            if k not in w_glob or not torch.is_floating_point(w_glob[k]):
                                continue
                            honest_deltas = torch.stack([
                                w_locals[hid][k].detach().to(torch.float32) - w_glob[k].detach().to(torch.float32)
                                for hid in honest_ids
                            ], dim=0)
                            mean_delta = honest_deltas.mean(dim=0)
                            w_attacked[k] = w_glob[k] - lambda_ipm * mean_delta
                        attack_name = "IPM"
                    elif attack_type == 6:  # LIE — A Little is Enough (Baruch et al., 2019)
                        # Coordinate-wise: send mean - z * std, i.e. z standard
                        # deviations below the honest mean per coordinate.
                        z_max = 1.5
                        for k in w_attacked.keys():
                            if k not in w_glob or not torch.is_floating_point(w_glob[k]):
                                continue
                            honest_deltas = torch.stack([
                                w_locals[hid][k].detach().to(torch.float32) - w_glob[k].detach().to(torch.float32)
                                for hid in honest_ids
                            ], dim=0)
                            mean_delta = honest_deltas.mean(dim=0)
                            std_delta = honest_deltas.std(dim=0, unbiased=False).clamp(min=1e-12)
                            malicious_delta = mean_delta - z_max * std_delta
                            w_attacked[k] = w_glob[k] + malicious_delta
                        attack_name = "LIE"
                    w_locals[cid] = w_attacked
                    logging.info(f'Client {cid}: applied {attack_name} attack (coordinated)')

        normal_clients = user_id
        # upload and download (aggregation)
        with torch.no_grad():
            all_clients = user_id
            if immune_detector is not None:
                client_params = {cid: w_locals[cid] for cid in all_clients}
                normal_clients = immune_detector.detect(
                    client_params=client_params,
                    all_clients=all_clients,
                    previous_model=w_glob,
                )
                abnormal_clients = [cid for cid in all_clients if cid not in normal_clients]
                logging.info(f"[IMMUNE] normal_clients={normal_clients} abnormal_clients={abnormal_clients}")
                logging.info(
                    f"[IMMUNE] reference_clients={immune_detector.last_reference_clients} "
                    f"scores={{{', '.join(f'{cid}: {immune_detector.last_client_scores.get(cid, 0.0):.4f}' for cid in all_clients)}}}"
                )

            if args.aggregation == 'fedHybridThreeLine':
                aggregator = fedHybridThreeLine
            else:
                aggregator = None

            if aggregator is not None:
                w_glob = aggregator(
                    w_locals=w_locals,
                    dict_len=dict_len,
                    w_global=w_glob,
                    normal_clients=normal_clients,
                    pop_size=args.csea_pop_size,
                    generations=args.csea_generations,
                    client_reliability=(
                        immune_detector.reliability_scores
                        if immune_detector is not None else None
                    ),
                    prior_mode=args.csea_prior_mode,
                    prior_strength=args.csea_prior_strength,
                    diversity_strength=args.csea_diversity_strength,
                )
            else:
                w_glob = FedAvg(w_locals, dict_len, normal_clients)
        net_glob.load_state_dict(w_glob)
        for id in user_id:
            net_locals[id].load_state_dict(w_glob)

        # global validation
        net_glob = net_glob.cuda()
        bacc_g, acc_g, f1_macro_g, auc_g, conf_matrix = compute_bacc(
            net_glob, val_loader, get_confusion_matrix=True, args=args)
        writer.add_scalar(f'glob/bacc_val', bacc_g, com_round)
        writer.add_scalar(f'glob/acc_val', acc_g, com_round)
        writer.add_scalar(f'glob/macro_f1_val', f1_macro_g, com_round)
        writer.add_scalar(f'glob/auc_val', auc_g, com_round)
        logging.info('global conf_matrix')
        logging.info(conf_matrix)

        test_bacc_g, test_acc_g, test_f1_macro_g, test_auc_g, test_conf_matrix = compute_bacc(
            net_glob, test_loader, get_confusion_matrix=True, args=args)
        best_test_acc = max(best_test_acc, test_acc_g)
        with open(test_acc_path, 'a', encoding='utf-8') as f:
            f.write(f'{com_round + 1}\t{test_bacc_g:.6f}\n')
        with open(best_test_acc_path, 'a', encoding='utf-8') as f:
            f.write(f'{com_round + 1}\t{best_test_acc:.6f}\n')
        writer.add_scalar(f'glob/bacc_test', test_bacc_g, com_round)
        writer.add_scalar(f'glob/acc_test', test_acc_g, com_round)
        writer.add_scalar(f'glob/macro_f1_test', test_f1_macro_g, com_round)
        writer.add_scalar(f'glob/auc_test', test_auc_g, com_round)
        logging.info('global test conf_matrix')
        logging.info(test_conf_matrix)

        # save model
        if acc_g > best_performance:
            best_performance = acc_g
            torch.save(net_glob.state_dict(),  models_dir +
                       f'/best_model_{com_round}_{best_performance}.pth')
            torch.save(net_glob.state_dict(),  models_dir+'/best_model.pth')
        logging.info(
            f'best val acc: {best_performance}, now val acc: {acc_g}, '
            f'BACC: {bacc_g:.7f}, F1(macro): {f1_macro_g:.7f}, AUC: {auc_g:.7f}'
        )
        logging.info(
            f'best test acc: {best_test_acc}, now test acc: {test_acc_g}, '
            f'BACC: {test_bacc_g:.7f}, F1(macro): {test_f1_macro_g:.7f}, AUC: {test_auc_g:.7f}'
        )
        acc.append(acc_g)

    writer.close()
    logging.info(acc)
