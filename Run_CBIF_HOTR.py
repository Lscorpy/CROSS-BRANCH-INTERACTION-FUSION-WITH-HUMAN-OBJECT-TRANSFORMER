import os
import sys
import time
from datetime import datetime
from datetime import timedelta
import types
import argparse
import torch
from torch.utils.data import DataLoader


from wasabi import msg

import numpy as np, random


# DATA_PATH = r"output_dataset"
# WEIGHTS   = r"vcoco_q16.pth"
# OUTPUT    = r"\checkpoints\DUAL_cross_branch_model" 

DATA_PATH = r"D:\coding\CROSS-BRANCH INTERACTION FUSION WITH HUMAN-OBJECT TRANSFORMER\dataset"
WEIGHTS   = r"D:\coding\FYP_Project\Merge\vcoco_q16.pth"
OUTPUT    = r"D:\coding\checkpoints\DUAL_cross_branch_model"

EPOCHS = 60
BATCH  = 2
LR     = 1e-4
NUMWORKERS=2


Calculate_mAP = True  # Set to True to calculate mAP after each epoch (slower)
DEVICE = "cuda"


# RESUME    = r"best.pth"
RESUME=r""


EVAL_ONLY = False

VIEW_pretrined_model_loading=False


from CBIF_HOTR.models import build_model
from CBIF_HOTR.engine.trainer import train_one_epoch
from CBIF_HOTR.engine import Dual_evaluator
from CBIF_HOTR.data.datasets import build_dataset
import CBIF_HOTR.util.misc as utils
from CBIF_HOTR.util.logger import print_args

def build_args():
    args = types.SimpleNamespace(

        # ===== Backbone =====
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        masks             = False,

        data_path=DATA_PATH,

        device=DEVICE,

        # ── Training ─────────────────────────────────────────────────
        batch_size    = BATCH,
        lr            = LR,
        lr_backbone   = 1e-6,
        weight_decay  = 1e-5,
        epochs        = EPOCHS,
        lr_drop       = 30,
        num_workers   = NUMWORKERS,
        clip_max_norm = 0.1,


        # Evaluation
        validate=Calculate_mAP,   # run val loop each epoch
        print_validate=True,
        eval=EVAL_ONLY,
 
        wandb = False,

        # Needed by save_ckpt / hoi_evaluator
        output_dir=OUTPUT,
        distributed=False,
        seed=42,
        start_epoch=0,

        # transformer
        hidden_dim=256,
        enc_layers=6,
        dec_layers=6,
        nheads=8,
        dim_feedforward=2048,
        dropout=0.1,
        pre_norm=False,

        # HHI transformer
        HHI_enc_layers=6,
        HHI_dec_layers=6,
        HHI_nheads=8,
        HHI_dim_feedforward=2048,

        # HOI transformer
        HOI_enc_layers=6,
        HOI_dec_layers=6,
        HOI_nheads=8,
        HOI_dim_feedforward=2048,

        # DETR
        num_classes=93,
        num_queries=100,
        # ── DETR loss coefficients
        aux_loss=True,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        eos_coef=0.1,

        # HOI
        HOIDet=True,
        num_vcoco_actions=29, #29 VCOCO and 1 background
        num_hoi_queries=16,
        hoi_aux_loss=True,
        hoi_idx_loss_coef=1,
        hoi_act_loss_coef=5.0,
        hoi_violence_loss_coef = 10.0, # first use:0.8


        # ── Violence head ────────────────────────────────────────────
        num_violence_actions = 6,

        vcoco_action_names=None,
        human_actions_vcoco=None,
        object_actions_vcoco=None,
        num_human_act_vcoco=None,

        violence_action_names=None,
        human_actions_violence=None,
        object_actions_violence=None,
        num_human_act_violence=None,

        # VCOCO
        valid_ids_vcoco=list(range(29)),
        invalid_ids_vcoco=[],
        valid_ids_violence=list(range(6)),

        # HHI
        HHIDet=True,
        num_HHI_queries=8,
        num_HHI_action=4,
        HHI_idx_loss_coef=1,
        HHI_action_idx_loss_coef=10,
        HHI_aux_loss=True,

        HHI_action_names=None,
        human_actions_HHI=None,
        num_human_act_HHI=None,



        #HHI, HOTR
        share_enc=True,
        pretrained_dec=True,
        temperature=0.05,
        frozen_weights=None,
        freeze_detr=False,


        # matcher
        set_cost_class=1,
        set_cost_bbox=5,
        set_cost_giou=2,
        set_cost_idx=10,
        set_cost_act   = 5,

        # CBAF  (NEW)
        cbaf_nhead              = 8,
        cbaf_dropout            = 0.1,
        cbaf_use_gate           = True,
        cbaf_ffn                = True,
        cbaf_use_conf_gate  = True,
        anchor_loss_weight  =  0.5,
    )

    return args

def get_trainable_parameters_for_4gb(model,args, epoch=0, freeze_until=20):
    """Optimized for 4GB VRAM - unfreeze specific layers"""
    trainable_params = []
    # class_embed_params  = []
    violence_params     = []  # all violence-related params in one list
    backbone_params = []
    detr_params = []
    interaction_params = []
    pointer_Dclass_params = []
    cross_branch_params=[]
    new_HHI_params = [] 

    print(epoch)
    change_train_layer=False
    if epoch>=freeze_until and epoch<=39:
        change_train_layer=True
        print("unfreeze backbone and detr")

    # change_train_layer = (epoch >= freeze_until) and (epoch<=40)
    
    # Block names you want to freeze
    freeze_keywords = ['detr.backbone', 'detr.transformer.decoder', 'detr.transformer.encoder',
                       'detr.bbox_embed', 'detr.input_proj', 'detr.query_embed',
                        'action_embed'] # HOTR action embed is also frozen to preserve V-COCO knowledge

    for name, param in model.named_parameters():
        param.requires_grad = False  # freeze first

        # Check if the parameter matches any of our freeze keywords
        if any(keyword in name for keyword in freeze_keywords):
            # param.requires_grad = False
            param.requires_grad = change_train_layer
            if "detr.backbone" in name:
                # print(f"  [trainable] {name}")
                if change_train_layer:
                    backbone_params.append(param)
            elif "detr." in name:
                # print(f"  [trainable] {name}")
                if change_train_layer:
                    detr_params.append(param)
            else:
                # print(f"  [trainable] {name}")
                if change_train_layer:
                    trainable_params.append(param)

        else:
            # This keeps Interaction Transformer, class_embed, Pointers, and Violence heads active
            param.requires_grad = True
            if "violence_adapter" in name or \
            "violence_action_embed" in name:
                violence_params.append(param)
                # print(f"  [trainable] {name}")

            elif "interaction_transformer" in name or \
            "hoi_query_embed" in name:
                interaction_params.append(param)
                # print(f"  [trainable] {name}")

            elif "H_Pointer_embed" in name or "O_Pointer_embed" in name\
                or "detr.class_embed" in name:
                pointer_Dclass_params.append(param)
                # print(f"  [trainable] {name}")
            elif "cbaf" in name:
                cross_branch_params.append(param)
                # print(f"  [trainable] {name}")
            else:
                new_HHI_params.append(param)
                # print(f"  [trainable] {name}")


        # ── summary ──────────────────────────────────────────────────────
    print(f"\n🔧 Freeze status (epoch={epoch}, unfreeze={change_train_layer})")
    print(f"\n  violence params     : {sum(p.numel() for p in violence_params):,}")
    print(f"  interaction params  : {sum(p.numel() for p in interaction_params):,}")
    print(f"  pointer/class params      : {sum(p.numel() for p in pointer_Dclass_params):,}")
    print(f"  cross_branch params    : {sum(p.numel() for p in cross_branch_params):,}")
    print(f"  new_HHI params    : {sum(p.numel() for p in new_HHI_params):,}")
    print(f"  backbone_params params    : {sum(p.numel() for p in backbone_params):,}")
    print(f"  detr_params params    : {sum(p.numel() for p in detr_params):,}")
    print(f"  trainable_params params    : {sum(p.numel() for p in trainable_params):,}")
    # print(f"  Total trainable     : {sum(p.numel() for p in [param for group in param_groups for param in group['params']]):,}")

    param_groups = [
        {'params': pointer_Dclass_params, 'lr': 2e-4},
        {'params': violence_params,    'lr': 5e-4},
        {'params': interaction_params,    'lr': 1e-4},
        {'params': new_HHI_params,    'lr': 2e-4},
        {'params': cross_branch_params,    'lr': 2e-4},
    ]
    if change_train_layer:
        param_groups.append({'params': backbone_params,    'lr': args.lr_backbone}) # 1e-6
        param_groups.append({'params': detr_params,         'lr': args.lr * 0.1}) # 1e-4*0.1
        param_groups.append({'params': trainable_params,    'lr': 1e-5})

    print(f"  Total trainable     : {sum(p.numel() for p in [param for group in param_groups for param in group['params']]):,}")

    

    return param_groups


# ==========================
# CHECKPOINT HELPERS
# ==========================
def save_model_report(model, optimizer=None,output_dir=None, filename="model_report.txt"):
    """
    Save model architecture, modules, parameter details,
    trainable/frozen status, and optimizer information.
    """

    if output_dir is None:
        save_path = filename
    else:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)

    total_params = 0
    trainable_params = 0

    with open(save_path, "w", encoding="utf-8") as f:

        f.write("=" * 120 + "\n")
        f.write("MODEL STRUCTURE\n")
        f.write("=" * 120 + "\n\n")
        f.write(str(model))

        # ======================================================
        # Module Summary
        # ======================================================
        f.write("\n\n")
        f.write("=" * 120 + "\n")
        f.write("TOP-LEVEL MODULE SUMMARY\n")
        f.write("=" * 120 + "\n\n")

        for name, module in model.named_children():

            params = sum(p.numel() for p in module.parameters())
            trainable = sum(
                p.numel()
                for p in module.parameters()
                if p.requires_grad
            )

            f.write(
                f"{name:40s} "
                f"Total={params:12,d} "
                f"Trainable={trainable:12,d}\n"
            )

        # ======================================================
        # Parameter Details
        # ======================================================
        f.write("\n\n")
        f.write("=" * 120 + "\n")
        f.write("PARAMETER DETAILS\n")
        f.write("=" * 120 + "\n\n")

        for name, param in model.named_parameters():

            num_params = param.numel()
            total_params += num_params

            if param.requires_grad:
                trainable_params += num_params

            f.write(
                f"{name}\n"
                f"  Shape      : {tuple(param.shape)}\n"
                f"  Parameters : {num_params:,}\n"
                f"  Trainable  : {param.requires_grad}\n\n"
            )

        # ======================================================
        # Summary
        # ======================================================
        f.write("=" * 120 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 120 + "\n")

        f.write(f"Total Parameters     : {total_params:,}\n")
        f.write(f"Trainable Parameters : {trainable_params:,}\n")
        f.write(f"Frozen Parameters    : {total_params - trainable_params:,}\n")

        # ======================================================
        # Optimizer
        # ======================================================
        if optimizer is not None:

            f.write("\n")
            f.write("=" * 120 + "\n")
            f.write("OPTIMIZER\n")
            f.write("=" * 120 + "\n")

            f.write(f"Type : {optimizer.__class__.__name__}\n")

            for i, group in enumerate(optimizer.param_groups):

                f.write(f"\nParam Group {i}\n")

                for key, value in group.items():
                    if key != "params":
                        f.write(f"  {key}: {value}\n")

    print(f"✅ Model report saved: {save_path}")


def save_checkpoint(output_dir, epoch, model, optimizer, lr_scheduler, args, filename=None):
    """
    Save full training state so training can resume exactly.
    Saves both a per-epoch file and overwrites 'best.pth' when best=True.
    """
    state = {
            "model"       : model.state_dict(),
            "optimizer"   : optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch"       : epoch,
            "args"        : args,          # ← added to match main.py format
        }
    
    # Rolling checkpoint (same name main.py uses)
    rolling = os.path.join(output_dir, "checkpoint.pth")
    torch.save(state, rolling)
 
    # Per-epoch snapshot
    epoch_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}.pth")
    torch.save(state, epoch_path)
    print(f"  [ckpt] saved → {epoch_path}")
 
 
    if filename == "best":
        best_path = os.path.join(output_dir, "best.pth")
        torch.save(state, best_path)
        print(f"  [ckpt] new best → {best_path}")
 
 
def load_checkpoint(args,path, model, optimizer, lr_scheduler, device):
    """
    Resume from a checkpoint saved by save_checkpoint().
    Returns the epoch to start from.
    """
    print(f"  [resume] loading {path}")
    ckpt = torch.load(path, map_location=device,weights_only=False)
    start_epoch = ckpt["epoch"] + 1
    trainable_params = get_trainable_parameters_for_4gb(model,args,start_epoch)
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    
    print(f"  [resume] continuing from epoch {start_epoch + 1}")
    return start_epoch



# ==========================
# LOG SAVING
# ==========================
log_dir = os.path.join(OUTPUT, "logs")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


def log_print(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

import csv

csv_file = os.path.join(log_dir, "CBIF-HOTR_metrics.csv")

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "total_loss", "object_label_loss",
            "loss_human_bbox","loss_human_giou",
            "violence_action_loss","violence_object_loss","violence_human_loss",
            "vcoco_action_loss","vcoco_object_loss", "vcoco_human_loss",
            "loss_HHI_action","loss_aggressor","loss_victim","loss_visibility",
            "Scenario1_HOIvcoco","Scenario2_HOIvcoco","Scenario1_HOIviolence","Scenario2_HOIviolence","Scenario1_HHI","Scenario2_HHI",
            "Detection pair recall (HOI) ","Detection pair recall (HHI)",
            "iou_hoi_human_mean","iou_hoi_object_mean",
            "iou_HHI_aggressor_mean","iou_HHI_victim_mean",
            "lr"
        ])




def main():
    args = build_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT, exist_ok=True)

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {gpu_mem:.1f} GB")


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Dataset ──────────────────────────────────────────────────────
    print("📚 Building datasets...")
    dataset_train = build_dataset("train", args)
    dataset_val   = build_dataset("val",   args)
 
    assert dataset_train.HHI_num_action() == dataset_val.HHI_num_action(), \
        "Train / val HHI action counts differ!"
    assert dataset_train.num_vio_action() == dataset_val.num_vio_action(), \
        "Train / val Violence HOTR action counts differ!"
    

    args.num_classes   = dataset_train.num_COCO_category()

    args.num_actions = 29 # force V-COCO num_actions be 29 match vcoco_q16.pth
    args.num_violence_actions = dataset_train.num_vio_action() # maps to your 6 danger actions
    args.num_HHI_action=dataset_train.HHI_num_action()


    args.HHI_action_names=dataset_train.get_HHI_actions()
    args.human_actions_HHI = dataset_train.get_HHI_human_action()
    args.num_human_act_HHI  = dataset_train.num_HHI_human_act()   


    args.vcoco_action_names = dataset_train.get_vcoco_actions()
    args.violence_action_names  = dataset_train.get_vio_actions()

    # ENSURE the shared encoder (interaction and detr) using the same weights
    if args.share_enc:     args.HHI_enc_layers=args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.HHI_dec_layers=args.hoi_dec_layers = args.dec_layers
    
    
    args.valid_ids_violence = np.array(dataset_train.get_valid_object_label_idx()).nonzero()[0] 
    args.valid_ids_vcoco = np.array(dataset_train.get_object_label_idx_vcoco()).nonzero()[0] 
    """
    only save valid vcoco action index
    """
    args.invalid_ids_vcoco = np.argwhere(np.array(dataset_train.get_object_label_idx_vcoco()) == 0).squeeze(1)

    args.human_actions_vcoco  = dataset_train.get_human_action_vcoco()       # 25 V-COCO actions with human involvement
    args.object_actions_vcoco = dataset_train.get_object_action_vcoco()  
    args.num_human_act_vcoco  = dataset_train.num_human_act_vcoco()         
# 29 V-COCO actions with object involvement (includes the 25 + 4 non-object actions) 

    args.human_actions_violence  = dataset_train.get_vio_human_action()       # 6 violence-specific human actions
    args.object_actions_violence = dataset_train.get_vio_object_action()      # 6 violence-specific object actions (mapped from the original 29 V-COCO actions)     
    args.num_human_act_violence  = dataset_train.num_vio_human_act()   

    print_args(args)


 # ── DataLoaders ──────────────────────────────────────────────────
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )
    loader_val = DataLoader( 
        dataset_val,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
    )

    print("🚀 Building model...")
    model, criterion, postprocessors  = build_model(args)
    model.to(device)
    # n_parameters = print_params(model)

    print("📦 Loading pretrained weights...")
    # ckpt = torch.load(WEIGHTS, map_location="cpu")
    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(WEIGHTS, map_location="cpu",weights_only=False)
# `weights_only=False` bypasses security restrictions (not recommended for production)
    ckpt_state = ckpt["model"]


    model_state = model.state_dict()
    filtered, skipped = {}, []
    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            model_shape = tuple(model_state[k].shape) if k in model_state else "ABSENT"
            skipped.append(f"{k}  ckpt={tuple(v.shape)}  model={model_shape}")
    print()
    missing, unexpected = model.load_state_dict(filtered, strict=False)


# marked as missing and are left initialized with their default random weights (Xavier/He initialization).
    if VIEW_pretrined_model_loading:
        print(f"  [load] missing keys: {missing}")
        print(f"  [load] unexpected keys: {unexpected}")

        print(f"  [load] transferred  : {len(filtered)} tensors")
        if skipped:
            print(f"  [load] skipped (shape mismatch / not in ckpt):")
            for s in skipped:
                print(f"         {s}")
        # Keys in model but absent in ckpt → new heads, Xavier-init kept
        print(f"  [load] new heads (Xavier-init) : {missing}")

    if 'detr.class_embed.weight' in missing:
        with torch.no_grad():
            # Copy original 92 classes
            # model.detr.class_embed.weight[:92] = ckpt_state['detr.class_embed.weight']
            # model.detr.class_embed.bias[:92] = ckpt_state['detr.class_embed.bias']
            old_weight = ckpt_state["detr.class_embed.weight"]
            old_bias   = ckpt_state["detr.class_embed.bias"]
            # print("Before:", model.detr.class_embed.weight.data[0][:5])

            num_old = old_weight.shape[0]
            # print(num_old)
            model.detr.class_embed.weight[:num_old] = old_weight
            model.detr.class_embed.bias[:num_old] = old_bias


    print("\n🔧 4GB GPU detected - using conservative fine-tuning strategy")
    
    trainable_params = get_trainable_parameters_for_4gb(model,args)
    # args.lr = 5e-5

    criterion.to(device)


    assert len(trainable_params) > 0, "No trainable parameters found!"
    optimizer    = torch.optim.AdamW(trainable_params, lr=args.lr,
                                     weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    save_model_report(model,optimizer,OUTPUT)

  # ── 6. Resume (optional) ─────────────────────────────────────────
    start_epoch = 0
    resume=False

    if RESUME and os.path.isfile(RESUME):
        start_epoch = load_checkpoint(args,RESUME, model, optimizer, lr_scheduler, device)
        args.start_epoch = start_epoch
        
        resume=True

    elif RESUME:
        print(f"  [resume] file not found: {RESUME} — starting fresh")

    # ── Eval-only mode ───────────────────────────────────────────────
    if args.eval:
        print("\n🔍 Evaluation-only mode")
        sc1, sc2, sc1_v, sc2_v, sc1_HHI,sc2_HHI = Dual_evaluator(model, criterion, postprocessors,loader_val, device,args)

        print(f"  Scenario #1   (VCOCO)  mAP : {sc1:.2f}")
        print(f"  Scenario #2   (VCOCO)  mAP : {sc2:.2f}")
        print(f"  Scenario #1 (Violence) mAP : {sc1_v:.2f}")
        print(f"  Scenario #2 (Violence) mAP : {sc2_v:.2f}")
        print(f"  Scenario #1 (HHI) mAP : {sc1_HHI:.2f}")
        print(f"  Scenario #2 (HHI) mAP : {sc2_HHI:.2f}")
        return


    print(f"\nStarting training from epoch {start_epoch + 1}/{args.epochs}\n")
    best_loss    = float("inf")
    scenario1    = 0.0   # best mAP scenario 1 (with target)
    scenario2    = 0.0   # best mAP scenario 2 (without target)
    scenario1_v  = 0.0   # best mAP scenario 1 (violence)
    scenario2_v  = 0.0   # best mAP scenario 2 (violence without target)
    scenario1_HHI    = 0.0   # best mAP scenario 1 (with target)
    scenario2_HHI    = 0.0   # best mAP scenario 2 (without target)

    start_time   = time.time()

    for epoch in range(start_epoch, args.epochs):

        # Check if we need to adjust or unfreeze base parameters
        if epoch == 20 or resume or epoch==40:
            print("🔓 Unfreezing/freeze new layers + updating optimizer")
            resume=False
            args.freeze_detr=not args.freeze_detr
            param_groups = get_trainable_parameters_for_4gb(model, args, epoch)
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr,weight_decay=args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)



        train_stats = train_one_epoch(
            model,
            criterion,
            loader_train,
            optimizer,
            device,
            epoch,
            args.epochs,
            args.clip_max_norm,
            log=args.wandb,
        )
        lr_scheduler.step()
# branch 1
        violence_loss_total     = train_stats.get("loss_violence_total", 0)
        hoi_loss_total          = train_stats.get("loss_hoi_total", 0)

        violence_object_loss    =train_stats.get("loss_oidx_v",0)
        violence_human_loss     =train_stats.get("loss_hidx_v",0)
        vcoco_object_loss       =train_stats.get("loss_oidx",0)
        vcoco_human_loss        =train_stats.get("loss_hidx",0)
        vcoco_action_loss       = train_stats.get("loss_act", 0)
        violence_loss           = train_stats.get("loss_violence", 0)
        object_label_loss       = train_stats.get("loss_ce", 0)
# branch 2
        loss_human_bbox     = train_stats.get("loss_human_bbox", float("inf"))
        loss_human_giou     = train_stats.get("loss_human_giou", float("inf"))
        loss_action     = train_stats.get("loss_action", float("inf"))
        loss_aggressor  = train_stats.get("loss_aggressor", float("inf"))
        loss_victim     = train_stats.get("loss_victim", float("inf"))
        loss_visibility     = train_stats.get("loss_visibility", float("inf"))

        epoch_loss    = train_stats.get("loss_total", float("inf"))

        is_best_loss  = epoch_loss < best_loss

        if is_best_loss:
            best_loss = epoch_loss

        log_print(
                f"\nEpoch {epoch + 1:3d}/{args.epochs}  "
                f"\ntotal={epoch_loss:.4f}  "
                f"\nlr={optimizer.param_groups[0]['lr']:.2e}"
                + ("  ← best" if is_best_loss else "")
            )

         # ── Validation & mAP  (mirrors main.py exactly) ──────────────
        if args.validate:
            log_print('-' * 100)
            log_print("🔍 Running validation...")
 
            if utils.get_rank() == 0:
                sc1, sc2, sc1_v, sc2_v, sc1_HHI,sc2_HHI ,\
                recall_hoi_pair, recall_HHI_pair_visible,\
                iou_hoi_human_mean, iou_hoi_object_mean,\
                iou_HHI_aggressor_mean, iou_HHI_victim_mean= Dual_evaluator(model, criterion, postprocessors,loader_val, device,args)
                if sc1 > scenario1 or sc1_v > scenario1_v or sc1_HHI > scenario1_HHI:
                    scenario1 = sc1
                    scenario2 = sc2
                    scenario1_v = sc1_v
                    scenario2_v = sc2_v 
                    scenario1_HHI = sc1_HHI
                    scenario2_HHI = sc2_HHI
                    if (epoch>20):
                        save_checkpoint(OUTPUT, epoch, model, optimizer,
                                        lr_scheduler, args, filename="best")
                log_print(f'| Scenario #1 mAP (VCOCO): {sc1:.8f} ({scenario1:.8f})')
                log_print(f'| Scenario #2 mAP (VCOCO): {sc2:.8f} ({scenario2:.8f})')
                log_print(f'| Scenario #1 mAP (Violence): {sc1_v:.8f} ({scenario1_v:.8f})')
                log_print(f'| Scenario #2 mAP (Violence): {sc2_v:.8f} ({scenario2_v:.8f})')
                log_print(f'| Scenario #1 mAP (HHI): {sc1_HHI:.8f} ({scenario1_HHI:.8f})')
                log_print(f'| Scenario #2 mAP (HHI): {sc2_HHI:.8f} ({scenario2_HHI:.8f})')
            log_print('-' * 100)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                epoch_loss,
                object_label_loss,
                loss_human_bbox,loss_human_giou,
                violence_loss,violence_object_loss,violence_human_loss,
                vcoco_action_loss,vcoco_object_loss,vcoco_human_loss,
                loss_action,loss_aggressor,loss_victim,loss_visibility,
                sc1, sc2,
                sc1_v, sc2_v,
                sc1_HHI, sc2_HHI,
                recall_hoi_pair,recall_HHI_pair_visible, 
                iou_hoi_human_mean,iou_hoi_object_mean,
                iou_HHI_aggressor_mean,iou_HHI_victim_mean,
                optimizer.param_groups[0]['lr']
            ])

        # Save every 3 epochs, or when we have the best training loss,    
        # or when we have a new best mAP (already saved inside val block).
        if (
            (epoch <= 25 and epoch % 10 == 0)
            or
            (epoch > 25 and (is_best_loss or (epoch + 1) % 5 == 0))
        ):
            save_checkpoint(OUTPUT,epoch,model,
                optimizer,lr_scheduler,args,
                filename="checkpoint" 
            )
            
    total_time = str(timedelta(seconds=int(time.time() - start_time)))
    log_print(f"\n✅ Training finished in {total_time}")
    log_print(f"   Best training loss : {best_loss:.4f}")
    log_print(f"   Best Scenario #1 (VCOCO)  : {scenario1:.2f}")
    log_print(f"   Best Scenario #2 (VCOCO)  : {scenario2:.2f}")
    log_print(f"   Best Scenario #1 (Violence)  : {scenario1_v:.2f}")
    log_print(f"   Best Scenario #2 (Violence)  : {scenario2_v:.2f}")
    log_print(f"   Best Scenario #1 (HHI)  : {scenario1_HHI:.2f}")
    log_print(f"   Best Scenario #2 (HHI)  : {scenario2_HHI:.2f}")
    log_print(f"   Checkpoints saved  : {OUTPUT}")

    # Final model weights only (lightweight, easy to distribute)
    final_path = os.path.join(OUTPUT, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"   Final weights      : {final_path}")



if __name__ == "__main__":

    main()





