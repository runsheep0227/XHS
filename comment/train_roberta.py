from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
    TrainingArguments,
    Trainer,
)

try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    from transformers.trainer_pt_utils import get_parameter_names

    try:
        from transformers.trainer_utils import is_sagemaker_mp_enabled
    except ImportError:
        try:
            from transformers.utils import is_sagemaker_mp_enabled
        except ImportError:

            def is_sagemaker_mp_enabled() -> bool:
                return False

    _OPT_GROUP_HELPERS = True
except ImportError:
    _OPT_GROUP_HELPERS = False
    ALL_LAYERNORM_LAYERS = ()  # type: ignore[assignment]
    get_parameter_names = None  # type: ignore[assignment]

    def is_sagemaker_mp_enabled() -> bool:
        return False

# ==================== 标签（原始 -1/0/1 -> 模型 0/1/2）====================
ORIGINAL_VALID = {-1, 0, 1}
ORIGINAL_TO_MODEL = {-1: 0, 0: 1, 1: 2}
LABEL2ID_STR = {"-1": 0, "0": 1, "1": 2}
ID2LABEL_STR = {0: "-1", 1: "0", 2: "1"}

# ==================== 默认：MacBERT-large（脚本会自动降为 train 4×4 等）====================
DEFAULT_MODEL_NAME = "hfl/chinese-macbert-large"
# 验证集上长评论误判多时可略增；OOM 时用 --max_length 256
DEFAULT_MAX_LENGTH = 512
# 有效 batch 12×2≈24；8GB 通常有余量
DEFAULT_TRAIN_BATCH = 12
DEFAULT_GRAD_ACCUM = 2
DEFAULT_EVAL_BATCH = 32

# 含 large / 340m 等时，若未显式改 batch，可在 main 里套用（见 apply_large_model_defaults）
LARGE_TRAIN_BATCH = 4
LARGE_GRAD_ACCUM = 4
LARGE_EVAL_BATCH = 16

# ---------- 默认侧重「少漏真负向」（验证集 recall/F1 导向）----------
# 默认：sqrt 类权重 + sqrt 抽样 + cap 限制极端权重 + 按 eval_f1_neg 存最优模型。
# 若负向 precision 崩、中/正大量被判负：显式加
#   --class_weight_scheme none --train_sampler none --best_metric f1_macro
# 若仍嫌漏负：可试 --class_weight_scheme balanced --class_weight_cap 2 或 --focal_gamma 1.5
# ------------------------------------------------------------------

# 终端打印用（与上文 docstring 一致）
MODEL_CATALOG = """
---------------------------------------------------------------------------
推荐中文分类基座（--model_name 直接填 HuggingFace id）
---------------------------------------------------------------------------
  hfl/chinese-macbert-large            【当前默认】Large；建议 lr≈1e-5；batch 由脚本自动 4×4
  hfl/chinese-macbert-base             Base，数据少/噪声大时更稳；显存更省
  hfl/chinese-roberta-wwm-ext-large    Large，可与 macbert-large 对比
  hfl/chinese-roberta-wwm-ext          经典 base
  hfl/chinese-bert-wwm-ext             BERT-wwm 对照
---------------------------------------------------------------------------"""

HPARAM_GUIDE = """
===========================================================================
train_roberta.py 主要可调参数与影响（验证集表现为主，需与数据量/噪声匹配）
===========================================================================

【数据与输入】
  --max_length      截断长度。更大能保留长评论语义，但显存↑、单步变慢；过短会丢信息。
  --seed            固定随机性（划分若在外部脚本完成则主要影响 dropout 等）。

【模型与初始化】
  --model_name      预训练骨干；默认 macbert-large。数据少/易过拟合可改 macbert-base。

【优化与学习率】
  --lr              学习率。过大易震荡/不收敛；过小训练慢、可能欠拟合。Large 默认已略降。
  --epochs          上限（默认 20）；实际多由早停提前结束。
  --warmup_ratio    预热比例（默认略增，利于 Large 稳定）。
  --weight_decay    AdamW L2 正则。↑ 抑制过拟合，过大可能欠拟合。
  --grad_accum      梯度累积。等效放大 batch，显存不够时用它换「大 batch」效果。
  --train_batch     每卡 batch。↑ 梯度更稳、显存↑；与 grad_accum 相乘为有效 batch。
  --classifier_lr_multiplier  分类头相对 encoder 的学习率倍数（1=关闭）。微调时常用 3~10。

【损失与类别】
  --label_smoothing 标签平滑（仅 CE 路径）。减轻过信、略抗噪；过大预测会变「糊」。
  --focal_gamma     >0 时用 Focal，更关注难分类样本；不平衡或难分边界时可试。
  --class_weight_scheme  balanced / sqrt / none；默认 sqrt，配合少漏负向。
  --class_weight_cap     限制类权重相对均值倍数(默认 2.0)；抑制误报失控；极大值≈关闭。

【训练流程与选型】
  --best_metric     默认 f1_neg（验证集负向 F1 最优 checkpoint）；可改 f1_macro 等。
  --early_stopping_patience  验证指标连续多少个 epoch 无提升则停训；↑ 给更多机会但更耗时。
  --early_stopping_threshold  指标相对历史最优需超过该幅度才算「有提升」，过滤验证抖动。
  --lr_scheduler    学习率调度：cosine / linear / cosine_with_restarts（默认 cosine）。
  --train_sampler   none / sqrt / inv；默认 sqrt，batch 内略多抽少数类。
  --max_grad_norm   梯度裁剪上限。
  --save_total_limit  磁盘上保留最近多少个 epoch 的 checkpoint；↑ 可多保留阶段供人工比。

【运行目录与续训】
  --run_name        将 checkpoint 写入 comment/training_runs/<run_name>/，避免覆盖旧实验。
  --resume_from_checkpoint  latest 或具体 checkpoint 路径，从断点恢复优化器与学习率调度。
  --export_dir      最终「验证最优」模型导出目录（相对 comment/ 或绝对路径）。

【显存与速度】
  --no_gradient_checkpointing  关梯度检查点：更快、显存↑。
  --no_8bit_adam    不用 8bit 优化器：略多显存、部分环境更稳。
  --torch_compile   PyTorch 2 编译加速（环境不支持时会跳过）。

说明：默认组合面向「少漏负向」；若更在乎整体均衡或负向误报，请改 best_metric 与 class_weight/sampler。
===========================================================================
"""


def has_training_checkpoints(output_dir: str) -> bool:
    if not os.path.isdir(output_dir):
        return False
    return any(
        name.startswith("checkpoint-")
        and os.path.isdir(os.path.join(output_dir, name))
        for name in os.listdir(output_dir)
    )


def resolve_resume_kw(output_dir: str, resume: str | None) -> dict:
    """返回传给 trainer.train 的 resume_from_checkpoint 关键字参数。"""
    if not resume or not str(resume).strip():
        return {}
    r = str(resume).strip()
    low = r.lower()
    if low in ("true", "latest", "yes", "1"):
        if not has_training_checkpoints(output_dir):
            print(
                "[警告] --resume_from_checkpoint=latest 但 output_dir 下无 checkpoint-*，"
                "将从头训练。请确认 --run_name 与上次一致。"
            )
            return {}
        return {"resume_from_checkpoint": True}
    path = os.path.expanduser(r)
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(os.getcwd(), path))
    if not os.path.isdir(path):
        print(f"[错误] 续训路径不存在或不是目录: {path}", file=sys.stderr)
        sys.exit(1)
    return {"resume_from_checkpoint": path}


def write_checkpoint_eval_summary(output_dir: str) -> str | None:
    """扫描各 checkpoint 目录中的 trainer_state.json，汇总验证指标到 JSON。"""
    if not os.path.isdir(output_dir):
        return None
    rows: list[dict] = []
    for name in sorted(os.listdir(output_dir)):
        if not name.startswith("checkpoint-"):
            continue
        ckpt = os.path.join(output_dir, name)
        state_path = os.path.join(ckpt, "trainer_state.json")
        if not os.path.isfile(state_path):
            continue
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        log_hist = state.get("log_history") or []
        eval_row: dict = {}
        for entry in reversed(log_hist):
            if any(k.startswith("eval_") for k in entry):
                eval_row = {k: v for k, v in entry.items() if k.startswith("eval_")}
                break
        rows.append(
            {
                "checkpoint": name,
                "global_step": state.get("global_step"),
                "epoch": state.get("epoch"),
                **eval_row,
            }
        )
    if not rows:
        return None
    report_path = os.path.join(output_dir, "checkpoint_eval_summary.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[警告] 无法写入摘要文件: {e}")
        return None
    return report_path


def output_dir_for_run(current_dir: str, run_name: str | None) -> str:
    if run_name and str(run_name).strip():
        return os.path.join(current_dir, "training_runs", str(run_name).strip())
    return os.path.join(current_dir, "checkpoint_temp")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    f1_per_class = f1_score(
        labels, predictions, average=None, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_neg": f1_per_class[0],
        "f1_neu": f1_per_class[1],
        "f1_pos": f1_per_class[2],
    }


def build_class_weights(
    y: np.ndarray,
    classes: np.ndarray,
    scheme: str,
    cap_ratio: float | None = None,
) -> torch.Tensor:
    """由训练集标签构造类权重（均值归一为 1）。可选 cap 限制相对均值的倍数，防止少数类权重过猛。"""
    s = (scheme or "none").strip().lower()
    if s not in ("balanced", "sqrt", "none"):
        raise ValueError(f"未知 class_weight_scheme: {scheme!r}，可选 balanced|sqrt|none")
    if s == "none":
        w = np.ones(len(classes), dtype=np.float64)
    else:
        w = compute_class_weight("balanced", classes=classes, y=y)
        if s == "sqrt":
            w = np.sqrt(w)
    t = torch.tensor(w, dtype=torch.float32)
    t = t / t.mean()
    if (
        s != "none"
        and cap_ratio is not None
        and cap_ratio > 1.0
    ):
        m = t.mean()
        t = torch.clamp(t, m / cap_ratio, m * cap_ratio)
        t = t / t.mean()
    return t


class WeightedTrainer(Trainer):
    """类权重 + CE（可选 label smoothing）或 Focal Loss（gamma>0 时忽略 smoothing）。"""

    def __init__(
        self,
        class_weights=None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        classifier_lr_multiplier: float = 1.0,
        train_sampler_mode: str = "sqrt",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.classifier_lr_multiplier = float(classifier_lr_multiplier)
        self._train_sampler_mode = (train_sampler_mode or "none").strip().lower()

    @staticmethod
    def _param_is_classifier_head(param_name: str) -> bool:
        n = param_name.lower()
        if "classifier" in n:
            return True
        return n.endswith("score.weight") or n.endswith("score.bias")

    def create_optimizer(self):
        mult = getattr(self, "classifier_lr_multiplier", 1.0)
        if self.optimizer is not None:
            return self.optimizer
        if mult == 1.0:
            return super().create_optimizer()
        if not _OPT_GROUP_HELPERS or get_parameter_names is None:
            print(
                "[警告] 当前 transformers 版本缺少分层优化依赖，"
                "--classifier_lr_multiplier 已忽略。"
            )
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        head_decay: list[torch.nn.Parameter] = []
        head_nodecay: list[torch.nn.Parameter] = []
        back_decay: list[torch.nn.Parameter] = []
        back_nodecay: list[torch.nn.Parameter] = []
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = self._param_is_classifier_head(name)
            is_decay = name in decay_parameters
            if is_head:
                (head_decay if is_decay else head_nodecay).append(param)
            else:
                (back_decay if is_decay else back_nodecay).append(param)

        if not head_decay and not head_nodecay:
            return super().create_optimizer()

        lr = self.args.learning_rate
        wd = self.args.weight_decay
        optimizer_grouped_parameters = []
        if back_decay:
            optimizer_grouped_parameters.append(
                {"params": back_decay, "lr": lr, "weight_decay": wd}
            )
        if back_nodecay:
            optimizer_grouped_parameters.append(
                {"params": back_nodecay, "lr": lr, "weight_decay": 0.0}
            )
        if head_decay:
            optimizer_grouped_parameters.append(
                {"params": head_decay, "lr": lr * mult, "weight_decay": wd}
            )
        if head_nodecay:
            optimizer_grouped_parameters.append(
                {"params": head_nodecay, "lr": lr * mult, "weight_decay": 0.0}
            )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        strat = getattr(self.args, "train_sampling_strategy", "random")
        if strat != "random":
            return super()._get_train_sampler(train_dataset)
        mode = getattr(self, "_train_sampler_mode", "none")
        if mode == "none":
            return super()._get_train_sampler(train_dataset)
        ws = getattr(self.args, "world_size", 1) or 1
        if ws > 1:
            print("[提示] 多卡训练下不使用 train_sampler 重加权，已退回默认抽样。")
            return super()._get_train_sampler(train_dataset)
        if train_dataset is None or not len(train_dataset):
            return super()._get_train_sampler(train_dataset)
        labels = np.array(train_dataset["labels"], dtype=np.int64)
        counts = np.bincount(labels, minlength=3).astype(np.float64)
        if mode == "sqrt":
            class_w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        elif mode == "inv":
            class_w = 1.0 / np.maximum(counts, 1.0)
        else:
            return super()._get_train_sampler(train_dataset)
        sample_w = torch.as_tensor(class_w[labels], dtype=torch.double)
        g = torch.Generator()
        g.manual_seed(int(self.args.seed))
        return WeightedRandomSampler(
            sample_w, num_samples=len(labels), replacement=True, generator=g
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        if self.focal_gamma and self.focal_gamma > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            mod = (1.0 - pt).clamp(min=1e-6).pow(self.focal_gamma)
            loss = -(mod * log_pt)
            if w is not None:
                loss = loss * w[labels]
            loss = loss.mean()
        else:
            loss_fn = nn.CrossEntropyLoss(
                weight=w,
                label_smoothing=self.label_smoothing,
            )
            loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def is_large_encoder(model_name: str) -> bool:
    n = model_name.lower()
    return any(k in n for k in ("large", "340m", "330m", "xlarge"))


def apply_large_model_batch_defaults(args: argparse.Namespace) -> None:
    """Large 模型默认显存占用高：若仍为 base 默认 batch，则自动降为 4×4。"""
    if not is_large_encoder(args.model_name):
        return
    if args.train_batch == DEFAULT_TRAIN_BATCH and args.grad_accum == DEFAULT_GRAD_ACCUM:
        args.train_batch = LARGE_TRAIN_BATCH
        args.grad_accum = LARGE_GRAD_ACCUM
        args.eval_batch = LARGE_EVAL_BATCH
        print(
            "[提示] Large 级模型：已自动使用 train 4×4、eval 16；"
            "可手动指定 --train_batch / --grad_accum / --eval_batch 覆盖。"
        )


def prepare_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "label" not in df.columns or "content" not in df.columns:
        raise ValueError(f"{name} 需包含列 label, content")

    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out[out["label"].isin(list(ORIGINAL_VALID))].copy()
    out["labels"] = out["label"].astype(int).map(ORIGINAL_TO_MODEL)
    if out["labels"].isna().any():
        out = out.dropna(subset=["labels"])

    out["content"] = out["content"].astype(str)
    out = out[out["content"].str.strip().str.len() > 0]

    return out


def require_cuda() -> None:
    """本脚本按单机单卡 GPU 优化；无 CUDA 直接退出，避免误跑 CPU。"""
    if not torch.cuda.is_available():
        print(
            "错误：未检测到 CUDA。本脚本针对 NVIDIA GPU（如 RTX 5070）优化，"
            "请安装 CUDA 版 PyTorch 并在 GPU 环境下运行。",
            file=sys.stderr,
        )
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    print(f"GPU: {props.name} | 显存: {total_gb:.1f} GB | CUDA {torch.version.cuda}")


def configure_cuda_performance() -> None:
    """Tensor Core / TF32：在 Ampere 及以后架构上通常能加速且对微调影响很小。"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_optim_and_amp(use_8bit: bool) -> tuple[bool, bool, str]:
    """
    RTX 5070 支持 bf16：训练用 bf16（与 fp16 互斥）。
    优化器：优先 adamw_8bit 省显存；否则 adamw_torch（与 transformers 版本兼容最好）。
    """
    bf16 = torch.cuda.is_bf16_supported()
    if not bf16:
        print("[警告] 当前驱动/GPU 报告不支持 bf16，将使用 fp16。")

    optim = "adamw_torch"
    if use_8bit:
        try:
            import bitsandbytes  # noqa: F401

            optim = "adamw_8bit"
            print("[提示] 使用 adamw_8bit 优化器（省显存）。")
        except ImportError:
            print("[提示] 未安装 bitsandbytes，使用 adamw_torch。")

    return bf16, not bf16, optim


def parse_args():
    p = argparse.ArgumentParser(
        description="微调中文三分类（默认 chinese-macbert-large；bf16 + 8bit Adam + 梯度检查点）"
    )
    p.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace 模型 id（默认 {DEFAULT_MODEL_NAME}）；其它见 --list-models",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="打印推荐基座列表后退出",
    )

    p.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"评论截断长度（默认 {DEFAULT_MAX_LENGTH}）；OOM 可降至 256/192",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="训练轮数上限；早停仍会提前结束",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="学习率；省略时 base 默认 2e-5，Large 级模型默认 1e-5",
    )
    p.add_argument(
        "--train_batch",
        type=int,
        default=DEFAULT_TRAIN_BATCH,
        help="每卡 batch；OOM 改为 8",
    )
    p.add_argument(
        "--grad_accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
        help="梯度累积步数；有效 batch = train_batch × grad_accum",
    )
    p.add_argument("--eval_batch", type=int, default=DEFAULT_EVAL_BATCH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.02,
        help="CE 标签平滑；与 --focal_gamma>0 互斥（Focal 路径不用 smoothing）",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help=">0 启用 Focal Loss（如 2.0），侧重难例；0 为普通加权 CE",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.06,
        help="AdamW weight decay；略大有助于抑制过拟合与决策边界过激（可试 0.05~0.1）",
    )
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--no_8bit_adam", action="store_true")
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=8,
        help="验证指标连续多少个 epoch 无提升则早停（略大以免欠停）",
    )
    p.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=1e-4,
        help=(
            "早停：相对历史最优，指标改善幅度需大于该值才算一次有效提升（减轻验证集抖动）；"
            "设为 0 则任意严格变好即重置耐心"
        ),
    )
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=("cosine", "linear", "cosine_with_restarts"),
        help="学习率调度类型（默认 cosine，与 Large 微调常见配置一致）",
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪；loss 不稳定时可略降（如 0.5）",
    )
    p.add_argument(
        "--adam_beta2",
        type=float,
        default=None,
        help="AdamW beta2；省略时 Large 骨干自动用 0.98，其余用 transformers 默认",
    )
    p.add_argument(
        "--train_sampler",
        type=str,
        default="sqrt",
        choices=("none", "sqrt", "inv"),
        help=(
            "训练集 batch 抽样：sqrt=按类频平方根反比（默认，抬负向 recall）；"
            "none=均匀随机；inv=更强过采样少数类"
        ),
    )
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.12,
        help="预热占总步数比例；Large 略长 warmup 常更稳",
    )
    p.add_argument(
        "--torch_compile",
        action="store_true",
        help="PyTorch 2.x 可选 torch.compile（部分环境可能不兼容）",
    )
    p.add_argument(
        "--best_metric",
        type=str,
        default="f1_neg",
        choices=(
            "f1_macro",
            "f1_weighted",
            "f1_neg",
            "f1_neu",
            "f1_pos",
            "accuracy",
            "eval_loss",
        ),
        help=(
            "验证集选优写入 export_dir；默认 f1_neg 少漏负向；整体均衡可改 f1_macro；eval_loss 越小越好"
        ),
    )
    p.add_argument(
        "--print-hparams",
        action="store_true",
        help="打印超参说明后退出",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="实验子目录名；checkpoint 写入 comment/training_runs/<run_name>/，避免覆盖",
    )
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="磁盘保留最近多少个 epoch 的 checkpoint（多阶段对比可调大，注意磁盘）",
    )
    p.add_argument(
        "--export_dir",
        type=str,
        default="saved_model",
        help="验证集最优模型导出目录；相对路径相对 comment/，也可用绝对路径",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        metavar="PATH_OR_LATEST",
        help="续训：latest 从本 run 的 output_dir 加载最近 checkpoint；或指向 checkpoint-* 目录",
    )
    p.add_argument(
        "--class_weight_scheme",
        type=str,
        default="sqrt",
        choices=("balanced", "sqrt", "none"),
        help=(
            "损失类权重：sqrt=默认，温和抬少数类 recall；balanced 更强；"
            "none=不加权（负向误报高时用）"
        ),
    )
    p.add_argument(
        "--class_weight_cap",
        type=float,
        default=2.0,
        help=(
            "类权重相对均值倍数上限（默认 2.0）；仅 balanced/sqrt 时实质生效。"
            "负向仍过判可维持 2.0；需更强少数类可试 2.5~3；≈关闭可用 50"
        ),
    )
    p.add_argument(
        "--classifier_lr_multiplier",
        type=float,
        default=4.0,
        help="分类头相对 encoder 的 LR 倍数；1.0 关闭分层学习率",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_models:
        print(MODEL_CATALOG)
        return
    if args.print_hparams:
        print(HPARAM_GUIDE)
        return
    apply_large_model_batch_defaults(args)
    if args.lr is None:
        args.lr = 1e-5 if is_large_encoder(args.model_name) else 2e-5
    if args.classifier_lr_multiplier < 0:
        print("错误: --classifier_lr_multiplier 须 >= 0", file=sys.stderr)
        sys.exit(1)
    if args.classifier_lr_multiplier == 0:
        args.classifier_lr_multiplier = 1.0
    if args.class_weight_cap is not None and args.class_weight_cap <= 1.0:
        print("错误: --class_weight_cap 须 > 1.0", file=sys.stderr)
        sys.exit(1)
    if args.early_stopping_threshold < 0:
        print("错误: --early_stopping_threshold 须 >= 0", file=sys.stderr)
        sys.exit(1)
    require_cuda()
    configure_cuda_performance()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "bert_data")
    output_dir = output_dir_for_run(current_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    eff_batch = args.train_batch * args.grad_accum
    print("微调中文三分类情感（默认 MacBERT-large；Large 自动 train 4×4、默认 lr 1e-5）")
    loss_desc = (
        f"focal_gamma={args.focal_gamma}"
        if args.focal_gamma and args.focal_gamma > 0
        else f"label_smoothing={args.label_smoothing}"
    )
    print(
        f"  model={args.model_name} | max_length={args.max_length} | epochs<={args.epochs} | "
        f"lr={args.lr} (encoder) | head_lr×{args.classifier_lr_multiplier} | "
        f"scheduler={args.lr_scheduler} | warmup={args.warmup_ratio} | "
        f"有效batch≈{eff_batch} ({args.train_batch}×{args.grad_accum}) | "
        f"train_sampler={args.train_sampler} | class_weight={args.class_weight_scheme} | "
        f"{loss_desc} | weight_decay={args.weight_decay} | best_metric={args.best_metric} | "
        f"early_stop(patience={args.early_stopping_patience}, Δ>{args.early_stopping_threshold})"
    )
    print(f"  output_dir={os.path.abspath(output_dir)} | save_total_limit={args.save_total_limit}")
    if args.resume_from_checkpoint:
        print(f"  resume_from_checkpoint={args.resume_from_checkpoint!r}")
    export_resolved = (
        args.export_dir
        if os.path.isabs(args.export_dir)
        else os.path.join(current_dir, args.export_dir)
    )
    print(f"  export_dir（最优模型）={os.path.abspath(export_resolved)}")

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    df_train = prepare_dataframe(
        pd.read_csv(train_path, encoding="utf-8-sig"), "train"
    )
    df_val = prepare_dataframe(pd.read_csv(val_path, encoding="utf-8-sig"), "val")

    print(f"  训练集: {len(df_train)} 条 | 验证集: {len(df_val)} 条")

    y_train = df_train["labels"].astype(int).to_numpy()
    classes = np.array([0, 1, 2], dtype=int)
    class_weights = build_class_weights(
        y_train, classes, args.class_weight_scheme, args.class_weight_cap
    )
    cap_note = ""
    if args.class_weight_scheme != "none" and args.class_weight_cap is not None:
        cap_note = f" cap={args.class_weight_cap}"
    print(
        f"  类权重[{args.class_weight_scheme}]{cap_note} (neg/neu/pos): "
        f"{class_weights.detach().cpu().numpy().round(4).tolist()}"
    )
    if args.train_sampler != "none" and args.class_weight_scheme != "none":
        print(
            "[提示] train_sampler + class_weight 同时开启（默认用于抬负向 recall）。"
            "若负向 precision 崩：--class_weight_scheme none --train_sampler none --best_metric f1_macro"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        label2id=LABEL2ID_STR,
        id2label=ID2LABEL_STR,
    )

    train_dataset = Dataset.from_pandas(
        df_train[["content", "labels"]], preserve_index=False
    )
    val_dataset = Dataset.from_pandas(
        df_val[["content", "labels"]], preserve_index=False
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["content"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_train = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["content"],
        desc="Tokenize train",
    )
    tokenized_val = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["content"],
        desc="Tokenize val",
    )

    use_gc = not args.no_gradient_checkpointing
    if use_gc:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    use_bf16, use_fp16, optim_name = resolve_optim_and_amp(
        use_8bit=not args.no_8bit_adam
    )
    print(f"  精度: bf16={use_bf16} fp16={use_fp16} | 优化器: {optim_name} | 梯度检查点: {use_gc}")

    num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)

    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        per_device_train_batch_size=args.train_batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.eval_batch,
        bf16=use_bf16,
        fp16=use_fp16,
        optim=optim_name,
        gradient_checkpointing=use_gc,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=max(1, args.save_total_limit),
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=max(
            10,
            len(tokenized_train)
            // max(1, args.train_batch * args.grad_accum * 5),
        ),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    if "logging_first_step" in ta_sig:
        ta_kwargs["logging_first_step"] = True
    if "adam_beta2" in ta_sig:
        if args.adam_beta2 is not None:
            ta_kwargs["adam_beta2"] = args.adam_beta2
        elif is_large_encoder(args.model_name):
            ta_kwargs["adam_beta2"] = 0.98
    # 仅在新版 transformers 中存在；旧版传入会 TypeError
    if "save_safetensors" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["save_safetensors"] = True

    metric_key = args.best_metric
    if metric_key == "eval_loss":
        ta_kwargs["metric_for_best_model"] = "eval_loss"
        ta_kwargs["greater_is_better"] = False
    else:
        ta_kwargs["metric_for_best_model"] = f"eval_{metric_key}"
        ta_kwargs["greater_is_better"] = True

    training_args = TrainingArguments(**ta_kwargs)

    est = float(args.early_stopping_threshold)
    es_cb = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=est,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        classifier_lr_multiplier=args.classifier_lr_multiplier,
        train_sampler_mode=args.train_sampler,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[es_cb],
    )

    if args.torch_compile:
        try:
            trainer.model = torch.compile(trainer.model)  # type: ignore[assignment]
            print("[提示] 已启用 torch.compile。")
        except Exception as e:
            print(f"[警告] torch.compile 失败，跳过: {e}")

    resume_kw = resolve_resume_kw(output_dir, args.resume_from_checkpoint)
    print("开始训练…")
    trainer.train(**resume_kw)

    best_m = getattr(trainer.state, "best_metric", None)
    if best_m is not None:
        print(
            f"\n[验证集最优] {training_args.metric_for_best_model}={best_m:.6f}"
        )
        bck = getattr(trainer.state, "best_model_checkpoint", None)
        if bck:
            print(f"  checkpoint: {bck}")

    run_summary = {
        "best_metric_name": training_args.metric_for_best_model,
        "best_metric": best_m,
        "best_model_checkpoint": getattr(
            trainer.state, "best_model_checkpoint", None
        ),
        "global_step": trainer.state.global_step,
        "epoch": float(trainer.state.epoch),
        "class_weight_scheme": args.class_weight_scheme,
        "classifier_lr_multiplier": args.classifier_lr_multiplier,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "label_smoothing": args.label_smoothing,
        "focal_gamma": args.focal_gamma,
        "class_weight_cap": args.class_weight_cap,
        "train_batch": args.train_batch,
        "grad_accum": args.grad_accum,
        "lr_scheduler": args.lr_scheduler,
        "train_sampler": args.train_sampler,
        "early_stopping_threshold": est,
        "max_grad_norm": args.max_grad_norm,
        "adam_beta2": ta_kwargs.get("adam_beta2"),
    }
    summary_json = os.path.join(output_dir, "run_summary.json")
    try:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
        print(f"[提示] 本次训练摘要（验证集选优信息）: {os.path.abspath(summary_json)}")
    except OSError as e:
        print(f"[警告] 无法写入 run_summary.json: {e}")

    summary_path = write_checkpoint_eval_summary(output_dir)
    if summary_path:
        print(f"[提示] 各 checkpoint 验证指标摘要: {os.path.abspath(summary_path)}")

    save_path = (
        args.export_dir
        if os.path.isabs(args.export_dir)
        else os.path.join(current_dir, args.export_dir)
    )
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    abs_save = os.path.abspath(save_path)
    print(f"\n最佳验证模型已保存（绝对路径）:\n  {abs_save}")
    print("目录内文件（权重一般为 model.safetensors 或 pytorch_model.bin）：")
    for name in sorted(os.listdir(save_path)):
        fp = os.path.join(save_path, name)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            if sz >= 1024 * 1024:
                print(f"  - {name}  ({sz / (1024 * 1024):.1f} MB)")
            else:
                print(f"  - {name}  ({sz} B)")

    if os.path.isfile(test_path):
        df_test = prepare_dataframe(
            pd.read_csv(test_path, encoding="utf-8-sig"), "test"
        )
        if len(df_test) > 0:
            test_dataset = Dataset.from_pandas(
                df_test[["content", "labels"]], preserve_index=False
            )
            tokenized_test = test_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=["content"],
                desc="Tokenize test",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_metrics = trainer.evaluate(tokenized_test)
            print("\n=== 测试集（独立 hold-out）===")
            for k, v in test_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print("[提示] test.csv 无有效行，跳过测试集评估。")
    else:
        print("[提示] 未找到 test.csv，跳过测试集评估。")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
