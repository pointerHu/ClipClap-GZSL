from copy import deepcopy
import pathlib
import yaml
import logging
import pickle
import socket
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import subprocess
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.logger import PD_Stats, create_logger


def read_features(path):

    with open(path, 'rb') as f:
        x = pickle.load(f)

    data = x['features']
    fps=x['fps']
    url = [str(u) for u in list(x['video_names'])]

    return data, url, fps


def fix_seeds(seed=42):

    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_model_size(model, logger):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    logger.info(
        "Created network [%s] with total number of parameters: %.1f million."
        % (type(model).__name__, num_params / 1000000)
    )


def get_git_revision_hash():
    try:
        hash_string = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        hash_string = ""
    return hash_string

def dump_config_yaml(args, exp_dir):
    args_dict = deepcopy(vars(args))
    for k,v in args_dict.items():
        if isinstance(v, pathlib.PosixPath):
            args_dict[k] = v.as_posix()

    with open((exp_dir/"args.yaml"), "w") as f:
        yaml.safe_dump(args_dict, f)

def log_hparams(writer, args, metrics):
    args_dict = vars(args)
    for k,v in args_dict.items():
        if isinstance(v, pathlib.PosixPath):
            args_dict[k] = v.as_posix()
    del metrics["recall"]
    metrics = {"Eval/"+k: v for k,v in metrics.items()}
    # del args_dict['audio_hip_blocks']
    # del args_dict['video_hip_blocks']

    writer.add_hparams(args_dict, metrics)


def setup_experiment(args, *stats):
    if args.exp_name == "":
        exp_name = f"{datetime.now().strftime('%b%d_%H-%M-%S_%f')}_{socket.gethostname()}"
    else:
        exp_name = str(args.exp_name) + f"_{datetime.now().strftime('%b%d_%H-%M-%S_%f')}_{socket.gethostname()}"

    exp_dir = (args.log_dir / exp_name)
    exp_dir.mkdir(parents=True)

    (exp_dir / "checkpoints").mkdir()
    pickle.dump(args, (exp_dir / "args.pkl").open("wb"))

    dump_config_yaml(args, exp_dir)

    train_stats = PD_Stats(exp_dir / "train_stats.pkl", stats)
    val_stats = PD_Stats(exp_dir / "val_stats.pkl", stats)

    logger = create_logger(exp_dir / "train.log")

    logger.info(f"Start experiment {exp_name}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"The experiment will be stored in {exp_dir.resolve()}\n")
    logger.info("")

    writer = SummaryWriter(log_dir=exp_dir)
    return logger, exp_dir, writer, train_stats, val_stats


def setup_evaluation(args, *stats):

    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()

    # test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    test_stats = PD_Stats(eval_dir / "test_stats.pkl", ['seen', 'unseen', 'hm', 'zsl'])
    logger = create_logger(eval_dir / "eval.log")

    logger.info(f"Start evaluation {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    # for Tensorboard hparam logging
    writer = SummaryWriter(log_dir=eval_dir)

    return logger, eval_dir, test_stats, writer

def setup_visualizations(args, *stats):

    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()

    # test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    test_stats = PD_Stats(eval_dir / "test_stats.pkl", ['seen', 'unseen', 'hm', 'zsl'])
    logger = create_logger(eval_dir / "visuals.log")

    logger.info(f"Start visualizing {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    # for Tensorboard hparam logging
    writer = SummaryWriter(log_dir=eval_dir)

    return logger, eval_dir, test_stats, writer


def save_best_model(epoch, best_metric, model, optimizer, log_dir, args, metric="", checkpoint=False):
    logger = logging.getLogger()
    logger.info(f"Saving model to {log_dir} with {metric} = {best_metric:.4f}")
    if optimizer is None:
        optimizer = model.optimizer_gen
    save_dict = {
        "epoch": epoch + 1,
        "model": model.state_dict() if args.data_parallel == False else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric": metric
    }
    if checkpoint:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}_ckpt_{epoch}.pt"
        )
    else:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}.pt"
        )




def check_best_loss(epoch, best_loss, best_epoch, val_loss, model, optimizer, log_dir, args):
    if not best_loss:
        save_best_model(epoch, val_loss, model, optimizer, log_dir, args, metric="loss")
        return val_loss, epoch

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        save_best_model(epoch, best_loss, model, optimizer, log_dir, args, metric="loss")
    return best_loss, best_epoch



def check_best_score(epoch, best_score, best_epoch, hm_score, model, optimizer, log_dir, args):
    if not best_score:
        save_best_model(epoch, hm_score, model, optimizer, log_dir, args, metric="score")
        return hm_score, epoch
    if hm_score > best_score:
        best_score = hm_score
        best_epoch = epoch
        save_best_model(epoch, best_score, model, optimizer, log_dir, args, metric="score")
    return best_score, best_epoch


def load_model_parameters(model, model_weights):
    logger = logging.getLogger()
    loaded_state = model_weights
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys():
            self_state[name].copy_(param)
        else:
            logger.info("didnt load ", name)


def load_args(path):
    return pickle.load((path / "args.pkl").open("rb"))


def cos_dist(a, b):
    # https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res


def evaluate_dataset_baseline(dataset_tuple, model, device, distance_fn, best_beta=None,
                              new_model_sequence=False,
                              args=None, save_performances=False):

    dataset=dataset_tuple[0]
    data_loader=dataset_tuple[1]
    data_t = torch.tensor(dataset.all_data['text']).to(device)
    accumulated_audio_emb=[]
    accumulated_video_emb=[]
    accumulated_data_num=[]

    for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
        data_a = data['positive']["audio"].to(device)
        data_v = data['positive']["video"].to(device)
        data_num = target["positive"].to(device)
        masks = {}
        masks['positive'] = {'audio': data['positive']['audio_mask'], 'video': data['positive']['video_mask']}
        timesteps = {}
        timesteps['positive'] = {'audio': data['positive']['timestep']['audio'], 'video': data['positive']['timestep']['video']}


        all_data = (
            data_a, data_v, data_num, data_t, masks['positive'], timesteps['positive']
        )
        try:
            if args.z_score_inputs:
                all_data = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in all_data])
        except AttributeError:
            print("Namespace has no fitting attribute. Continuing")

        model.eval()
        with torch.no_grad():
            audio_emb, video_emb, emb_cls = model.get_embeddings(all_data[0], all_data[1], all_data[3], all_data[4], all_data[5])
            accumulated_audio_emb.append(audio_emb)
            accumulated_video_emb.append(video_emb)
            outputs_all = (audio_emb, video_emb, emb_cls)

        accumulated_data_num.append(data_num)

    stacked_audio_emb=torch.cat(accumulated_audio_emb)
    stacked_video_emb=torch.cat(accumulated_video_emb)
    data_num=torch.cat(accumulated_data_num)
    emb_cls=outputs_all[2]
    outputs_all=(stacked_audio_emb, stacked_video_emb, emb_cls)



    a_p, v_p, t_p = outputs_all
    # a_p = None


    video_evaluation = get_best_evaluation(dataset, data_num, a_p, v_p, t_p, mode="video", device=device,
                                           distance_fn=distance_fn, best_beta=best_beta, save_performances=save_performances,args=args)


    return {
        "audio": video_evaluation,
        "video": video_evaluation,
        "both": video_evaluation
    }

def get_best_evaluation(dataset, targets, a_p, v_p, t_p, mode, device, distance_fn, best_beta=None, save_performances=False, args=None, attention_weights=None):
    seen_scores = []
    zsl_scores = []
    unseen_scores = []
    hm_scores = []
    per_class_recalls = []
    start = 0 # 0
    end = 5 # 3
    steps = (end - start) * 15 + 1  # steps = (end - start) * 5 + 1
    betas = torch.tensor([best_beta], dtype=torch.float, device=device) if best_beta else torch.linspace(start, end, steps,
                                                                                                         device=device)
    seen_label_array = torch.tensor(dataset.seen_class_ids, dtype=torch.long, device=device)
    unseen_label_array = torch.tensor(dataset.unseen_class_ids, dtype=torch.long, device=device)
    seen_unseen_array = torch.tensor(np.sort(np.concatenate((dataset.seen_class_ids, dataset.unseen_class_ids))),
                                     dtype=torch.long, device=device)

    classes_embeddings = t_p
    with torch.no_grad():
        for beta in betas:
            if a_p == None:
                distance_mat = torch.zeros((v_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                           device=device) + 99999999999999
                distance_mat_zsl = torch.zeros((v_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                               device=device) + 99999999999999
            else:
                distance_mat = torch.zeros((a_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                           device=device) + 99999999999999
                distance_mat_zsl = torch.zeros((a_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                               device=device) + 99999999999999
            if mode == "audio":
                distance_mat[:, seen_unseen_array] = torch.cdist(a_p, classes_embeddings)  # .pow(2)
                mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device)
                mask[seen_label_array] = 99999999999999
                distance_mat_zsl = distance_mat + mask
                if distance_fn == "SquaredL2Loss":
                    distance_mat[:, seen_unseen_array] = distance_mat[:, seen_unseen_array].pow(2)
                    distance_mat_zsl[:, unseen_label_array] = distance_mat_zsl[:, unseen_label_array].pow(2)
            elif mode == "video":
                # L2

                v_p = v_p.type(torch.float32)
                distance_mat[:, seen_unseen_array] = torch.cdist(v_p, classes_embeddings)  # .pow(2)
                mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device)
                mask[seen_label_array] = 99999999999999
                distance_mat_zsl = distance_mat + mask
                if distance_fn == "SquaredL2Loss":
                    distance_mat[:, seen_unseen_array] = distance_mat[:, seen_unseen_array].pow(2)
                    distance_mat_zsl[:, unseen_label_array] = distance_mat_zsl[:, unseen_label_array].pow(2)
            elif mode == "both":
                # L2
                audio_distance = torch.cdist(a_p, classes_embeddings, p=2)  # .pow(2)
                video_distance = torch.cdist(v_p, classes_embeddings, p=2)  # .pow(2)

                if distance_fn == "SquaredL2Loss":
                    audio_distance = audio_distance.pow(2)
                    video_distance = video_distance.pow(2)

                # Sum
                distance_mat[:, seen_unseen_array] = (audio_distance + video_distance)

                mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device)
                mask[seen_label_array] = 99999999999999
                distance_mat_zsl = distance_mat + mask

            mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device) + beta
            mask[unseen_label_array] = 0
            neighbor_batch = torch.argmin(distance_mat + mask, dim=1)
            match_idx = neighbor_batch.eq(targets.int()).nonzero().flatten()
            match_counts = torch.bincount(neighbor_batch[match_idx], minlength=len(dataset.all_class_ids))[
                seen_unseen_array]
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
            per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            seen_recall_dict = per_class_recall[seen_label_array]
            unseen_recall_dict = per_class_recall[unseen_label_array]
            s = seen_recall_dict.mean()
            u = unseen_recall_dict.mean()

            if save_performances:
                seen_dict = {k: v for k, v in zip(np.array(dataset.all_class_names)[seen_label_array.cpu().numpy()], seen_recall_dict.cpu().numpy())}
                unseen_dict = {k: v for k, v in zip(np.array(dataset.all_class_names)[unseen_label_array.cpu().numpy()], unseen_recall_dict.cpu().numpy())}
                save_class_performances(seen_dict, unseen_dict, dataset.dataset_name)

            hm = (2 * u * s) / ((u + s) + np.finfo(float).eps)

            neighbor_batch_zsl = torch.argmin(distance_mat_zsl, dim=1)
            match_idx = neighbor_batch_zsl.eq(targets.int()).nonzero().flatten()
            match_counts = torch.bincount(neighbor_batch_zsl[match_idx], minlength=len(dataset.all_class_ids))[
                seen_unseen_array]
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
            per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            zsl = per_class_recall[unseen_label_array].mean()

            zsl_scores.append(zsl.item())
            seen_scores.append(s.item())
            unseen_scores.append(u.item())
            hm_scores.append(hm.item())
            per_class_recalls.append(per_class_recall.tolist())
        argmax_hm = np.argmax(hm_scores)
        max_seen = seen_scores[argmax_hm]
        max_zsl = zsl_scores[argmax_hm]
        max_unseen = unseen_scores[argmax_hm]
        max_hm = hm_scores[argmax_hm]
        max_recall = per_class_recalls[argmax_hm]
        best_beta = betas[argmax_hm].item()
    return {
        "seen": max_seen,
        "unseen": max_unseen,
        "hm": max_hm,
        "recall": max_recall,
        "zsl": max_zsl,
        "beta": best_beta
    }

def get_class_names(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        classes = sorted([line.strip() for line in f])
    return classes


def load_model_weights(weights_path, model):
    logging.info(f"Loading model weights from {weights_path}")
    load_dict = torch.load(weights_path)
    model_weights = load_dict["model"]
    epoch = load_dict["epoch"]
    logging.info(f"Load from epoch: {epoch}")
    load_model_parameters(model, model_weights)
    return epoch

def plot_hist_from_dict(dict):
    plt.bar(range(len(dict)), list(dict.values()), align="center")
    plt.xticks(range(len(dict)), list(dict.keys()), rotation='vertical')
    plt.tight_layout()
    plt.show()

def save_class_performances(seen_dict, unseen_dict, dataset_name, args=None):
    roor_path = args.load_path_stage_B
    seen_dir = os.path.join(roor_path, f'class_performance_{dataset_name}_seen.pkl')
    unseen_dir = os.path.join(roor_path, f'class_performance_{dataset_name}_unseen.pkl')

    seen_path = Path(seen_dir)
    unseen_path = Path(unseen_dir)
    with seen_path.open("wb") as f:
        pickle.dump(seen_dict, f)
        logging.info(f"Saving seen class performances to {seen_path}")
    with unseen_path.open("wb") as f:
        pickle.dump(unseen_dict, f)
        logging.info(f"Saving unseen class performances to {unseen_path}")
