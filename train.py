import os
import sys
import json, math
import argparse
import copy
import logging
import contextlib
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from src.utils.standalone_diffusion import (
    FlowMatchEulerDiscreteScheduler,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    get_scheduler,
    set_library_verbosity_error,
    set_library_verbosity_info,
)

from datetime import datetime
from time import time

REPO_ROOT = Path(__file__).resolve().parent
KERNELS_DIR = REPO_ROOT / "kernels"

def _prepend_import_path(path: Path):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_prepend_import_path(REPO_ROOT)
_prepend_import_path(KERNELS_DIR)

from src.datatsets.TiffDataset import NpzDataset, custom_collate_fn
from src.models.z_image.hl_model import Z_Transformer
from managers.grad_manager import AsyncGradSyncManager
import ProcessGroupHMPI
from unigemm_linear import kpLinear,replace_linear_with_kplinear
import my_allocator
from kp_normalization import replace_rmsNorm_with_kpRMSNorm
from managers.mem_manager import use_hbm_allocator

from src.models.z_image.standalone_transformer import FeedForward, ZImageTransformer2DModel

from async_rt.async_ops import MySiLU
from async_rt.runtime import AsyncRunMode, RUNTIME
from async_rt import is_asyncrt_enabled

import accumulate_sp
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from managers.weight_manager import (
    setup_shared_model,
    build_balanced_param_groups,
    AsyncParamAllGatherManager,
    build_chunk_param,
    set_opt_hook
)

def setup_logger(log_file: str, is_main_process: bool):
    log_level = logging.INFO if is_main_process else logging.ERROR
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if is_main_process:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="training script.")
    parser.add_argument( "--pretrained_model_name_or_path", type=str, default="", required=True)
    parser.add_argument("--work_dir", type=str, default="work_dir",)
    parser.add_argument("--comp_ckpt", type=str)
    parser.add_argument("--sub_work_dir", type=str, default="")
    parser.add_argument("--use_8bit_adam", type=bool, default=False,)
    parser.add_argument( "--data_root",type=str,default="")
    parser.add_argument("--checkpointing_steps",type=int,default=4000,)
    parser.add_argument("--rank", type=int, default=8, help="The dimension of the LoRA rank.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
    parser.add_argument("--cache_dir",type=str,default="cache",)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=60000,)
    parser.add_argument("--learning_rate",type=float,default=1e-4)
    parser.add_argument("--scale_lr",action="store_true",default=False,)
    parser.add_argument("--lr_scheduler",type=str,default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500,)
    parser.add_argument("--weighting_scheme",type=str,default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument("--dataloader_num_workers",type=int,default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--log_interval", type=int, default=1, help="Log interval.")
    parser.add_argument("--sp_size", type=int, default=4, help="sp group size")
    parser.add_argument("--rs_size", type=int, default=16, help="sp group size")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable torch profiler for 2 warmup + 3 active steps.")
    args = parser.parse_args(input_args)
    args.revision = None
    args.variant = None
    if len(args.sub_work_dir) > 0:
        args.work_dir = os.path.join(args.work_dir, args.sub_work_dir)
    args.run_timestamp = datetime.now().strftime("%y_%m_%d-%H:%M:%S")
    return args

def zero_shared_grad(model,param_rank,local_rank):
    for p in model.parameters():
        if hasattr(p, "state"):
            p.state.fill_(0)
def main(args):
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    
    if args.sp_size != 4 or world_size % 4:
        raise ValueError("HARP training requires groups of 4 sequence-parallel ranks")
    if args.train_batch_size != 1:
        raise ValueError("The ring attention kernel supports one sample per rank")

    torch.ops.load_library(KERNELS_DIR / "libring_attention_ops.so")
    torch.ops.load_library(KERNELS_DIR / "async_op.so")


    sp_size=args.sp_size
    rs_size=args.rs_size

    if world_size < sp_size * rs_size:
        rs_size = world_size // sp_size
    dp_size=world_size//sp_size

    local_rank = rank % sp_size
    if rank % 32 == 0:
        print(f"[rank{rank}] after init")

    rs_rank = ((rank % (rs_size * sp_size)) // sp_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    memory_node = int(os.environ.get("HARP_MEMORY_NODE", (rank % 16) + 16))
    my_allocator.init_allocator(memory_node, False)
    dist.init_process_group(backend="hmpi", world_size=world_size, rank=rank)
    device = torch.device("cpu")

    is_main_process = (rank == 0)

    run_group = world_size // 16
    args.work_dir = os.path.join(args.work_dir, str(run_group), args.run_timestamp)
    args.log_file = os.path.join(args.work_dir, "train.log")
    if is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)


    if args.seed is not None:
        torch.manual_seed(args.seed+(rank // sp_size))

    if is_main_process:
        with open(os.path.join(args.work_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    dist.barrier() 

    if is_main_process:
        set_library_verbosity_info()
        setup_logger(args.log_file, is_main_process)
    else:
        set_library_verbosity_error()
        setup_logger(args.log_file, is_main_process)

    weight_dtype = torch.bfloat16

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    train_dataset = NpzDataset(args.data_root)
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=dp_size,
        rank=rank // sp_size,
        shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.dataloader_num_workers,
        sampler=train_sampler,
        pin_memory=False
    )
    
    if is_main_process:
        logger.info("Training dataset and Dataloader initialized successfully.")

    config = ZImageTransformer2DModel.load_config(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
    )
    group_offset_1 = ProcessGroupHMPI.Options(thread_id_offset=1)
    group_offset_2 = ProcessGroupHMPI.Options(thread_id_offset=2)
    group_offset_3 = ProcessGroupHMPI.Options(thread_id_offset=3)
    

    if world_size < sp_size * rs_size:
        rs_size = world_size // sp_size
    assert world_size % (sp_size * rs_size) == 0
    
    sp_ranks = [[] for _ in range(dp_size)]
    
    rs_ranks = [[] for _ in range(world_size // rs_size)]
    
    dp_ranks = [[] for _ in range(sp_size)]


    for i in range(world_size):
        sp_ranks[i // sp_size].append(i)
        rs_ranks[i // (sp_size * rs_size) * sp_size + i % sp_size].append(i)
        dp_ranks[i % (sp_size)].append(i)
    
    logger.info(f"sp group: {len(sp_ranks)}x{len(sp_ranks[0])}")
    logger.info(f"rs group: {len(rs_ranks)}x{len(rs_ranks[0])}")
    logger.info(f"dp group: {len(dp_ranks)}x{len(dp_ranks[0])}")

    for sp_rank in sp_ranks:
        g_model_group = dist.new_group(ranks = sp_rank, pg_options=group_offset_2)
        if rank in sp_rank:
            sp_group_model = g_model_group
    
    for dp_rank in dp_ranks:
        g_group = dist.new_group(ranks = dp_rank, pg_options=group_offset_1)
        if rank in dp_rank:
            dp_group = g_group

    for rs_rank_group in rs_ranks:
        g_group = dist.new_group(ranks=rs_rank_group, pg_options=group_offset_3)
        if rank in rs_rank_group:
            rs_group = g_group

    from torch.utils._python_dispatch import TorchDispatchMode
    class FastInitMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            op_name = func.__name__
            if op_name == "uniform_.default":
                torch.ops.async_op.random_uniform(*args, **kwargs)
                return args[0]
            return func(*args, **kwargs)
    
    torch.set_default_dtype(weight_dtype)

    with FastInitMode():
        transformer_base = ZImageTransformer2DModel.from_config(config)

    transformer = Z_Transformer(
        transformer_base, num_metadata=train_dataset.num_metadata, use_cond=True, sp_group_model=sp_group_model,sp_size=sp_size
    ).to(device, dtype=weight_dtype)
  
    torch.set_default_dtype(torch.float32)



    if args.scale_lr:
        args.learning_rate = args.learning_rate * world_size * args.train_batch_size

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.work_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if is_main_process:
                logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            if is_main_process:
                logger.info(f"Resuming from checkpoint {path}")
            checkpoint_path = os.path.join(args.work_dir, path)
            
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
            transformer.module.load_state_dict(state_dict)
            
            if os.path.exists(os.path.join(checkpoint_path, "optimizer.bin")):
                optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.bin"), map_location="cpu"))
            if os.path.exists(os.path.join(checkpoint_path, "scheduler.bin")):
                lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scheduler.bin")))
            
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
    

    def new_gating(self, x1, x3):
        return MySiLU.apply(x1) * x3
    
    def patch_feedforward(module):
        for name, child in module.named_children():
            if isinstance(child, FeedForward):
                child._forward_silu_gating = new_gating.__get__(child, FeedForward)
            else:
                patch_feedforward(child)

    patch_feedforward(transformer)

    replace_linear_with_kplinear(transformer, kpLinear)
    replace_rmsNorm_with_kpRMSNorm(transformer)

    (
        my_params,
        param_rank,
    ) = build_balanced_param_groups(transformer, sp_size, local_rank)


    my_params_set = set(my_params)
    shm,shm_grad,shm_state = setup_shared_model(transformer,param_rank, rank = rank,local_rank=local_rank, master_rank=3)

    opt_params = build_chunk_param(my_params_set,transformer, rs_size, rs_rank)

    grad_sync_manager = AsyncGradSyncManager(
        model=transformer, 
        local_rank=local_rank,
        local_die = (rank % 16) // 4,
        my_params_set=my_params_set,
        param_rank = param_rank
    )
    base_core_id = (rank % 16) * 38
    
    param_sync_manager = AsyncParamAllGatherManager(
        rs_group=rs_group,
        rs_size=rs_size,
        core_id=base_core_id+35
    )

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        opt_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        foreach=True
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        num_training_steps_for_scheduler = args.max_train_steps
    else:
        num_training_steps_for_scheduler = args.max_train_steps
    
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_training_steps_for_scheduler,
    )
    if is_main_process:
        logger.info(f"lr_scheduler:{args.lr_scheduler} initialized successfully.")

    total_batch_size = args.train_batch_size * dp_size
    if is_main_process:
        logger.info("***** Running training (CPU) *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    profiler = None
    profiler_steps = 0
    profiler_output_dir = os.path.join("./ignores/tb_profiler", f"{datetime.now().strftime('%y_%m_%d-%H_%M')}")
    if args.enable_profiler:
        print("starting profiler")
        if is_main_process:
            os.makedirs(profiler_output_dir, exist_ok=True)
        dist.barrier()

        import torch.profiler as prof

        profiler = prof.profile(
            activities=[prof.ProfilerActivity.CPU],
            schedule=prof.schedule(wait=0, warmup=2, active=4, repeat=1),
            on_trace_ready=prof.tensorboard_trace_handler(profiler_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()

    dist.barrier()
    accumulate_sp.init([base_core_id + 32,base_core_id+35, base_core_id + 34],base_core_id+33, dp_group,rs_group)

    s_print = logger.info

    my_allocator.enable()
    for epoch in range(first_epoch, args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        transformer.train()
        for step, batch in enumerate(train_dataloader):

            with torch.no_grad():
                
                norm_metadata = batch["norm_metadata"].to(device=device, dtype=weight_dtype)
                prompt_embeds = [p.to(device=device, dtype=weight_dtype) for p in batch["prompt_embeds"]]
                latent_image = batch["latent_image"].to(device=device, dtype=weight_dtype)
                cond = [c.to(device=device, dtype=weight_dtype) for c in batch["cond"]]

                noise = torch.randn_like(latent_image)
                bsz = latent_image.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)
                dist.barrier()
                start_time = time()
                sigmas = get_sigmas(timesteps, n_dim=latent_image.ndim, dtype=latent_image.dtype)
                noisy_model_input = (1.0 - sigmas) * latent_image + sigmas * noise

            with contextlib.nullcontext():
                fwd_time = time()
                context = AsyncRunMode() if is_asyncrt_enabled() else contextlib.nullcontext()
                RUNTIME.offloader_enable = False
                with context:
                    model_pred = transformer(
                        x=list(noisy_model_input.unsqueeze(2).unbind(dim=0)),
                        t=(1000 - timesteps) / 1000,
                        text_embedding=prompt_embeds,
                        return_dict=False,
                        metadata=norm_metadata,
                        cond_list=[list(c.unbind(dim=0)) for c in cond]
                    )[0]
                RUNTIME.sync()
                s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]:forward finished {time() - fwd_time}s")
                model_pred = torch.stack(model_pred, dim=0).squeeze(2)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = latent_image - noise

            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )

            loss = loss.mean()
            step_loss = loss.detach().item()
            loss = loss/dp_size
            barr_time = time()
            dist.barrier()
            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]:barrier1 {time() - barr_time}s")
            
            bak_time = time()
            RUNTIME.offloader_enable = False
            with use_hbm_allocator():
                context = context = AsyncRunMode() if is_asyncrt_enabled() else contextlib.nullcontext()
                with context:
                    loss.backward()

            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]:backward finish {time() - bak_time}s")
            comm_time_start=time()
            tq_time, sp_time, dp_time, barr_time = grad_sync_manager.wait_for_completion()

            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]:dp task queue time: {tq_time}s")
            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]:sp time: {sp_time}s")
            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]:dp time: {dp_time}s")
            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]:bwd barr time: {barr_time}s")
            opt_time = time()
            optimizer.step()
            lr_scheduler.step()
            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]: optimizer finish {time() - opt_time}")
            zero_time = time()
            zero_shared_grad(transformer,param_rank=param_rank,local_rank=local_rank)
            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]: after zero_grad {time() - zero_time}s")
            zero_barrier = time()
            dist.barrier()
            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]: after zero_grad_barrier {time() - zero_time}s")
            param_manager_time = time()
            for p in my_params:
                if hasattr(p, "chunk_part"):
                    if not hasattr(p, "state"):
                        raise "[ERROR] why params has no state param"
                    param_sync_manager.push(p)
            s_print(f"[rank{rank}],[die{(rank % 16) // sp_size}], [numa{rank % 4}]: after push param {time() - param_manager_time}s")
                
            if step % args.log_interval == 0:
                if is_main_process:
                    logs = {
                        "global_step": global_step,
                        "step_loss": step_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    end_time = time()
                    logs["time"] = end_time - start_time
                    s_print(logs)
                    start_time = end_time

            global_step += 1
            if profiler is not None:
                profiler.step()
                profiler_steps += 1

            if global_step % args.checkpointing_steps == 0:
                
                save_path = os.path.join(args.work_dir, f"checkpoint-{global_step}")
                if is_main_process:
                    os.makedirs(save_path, exist_ok=True)
                dist.barrier()
                
                if is_main_process:
                    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
                    torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.bin"))
                    s_print(f"Saved state to {save_path}")
                    
            del loss, model_pred, target, noisy_model_input, sigmas, timesteps, prompt_embeds, cond, latent_image, noise
            if global_step >= args.max_train_steps or (profiler is not None and profiler_steps >= 5):
                break
        if profiler is not None and profiler_steps >= 6:
            break

    if profiler is not None:
        profiler.stop()
        print(
            f"[rank {rank}] profiler operators:\n"
            f"{profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=50)}"
        )
    param_sync_manager.wait_all()
    param_sync_manager.shutdown()
    grad_sync_manager.shutdown()
    dist.barrier()
    accumulate_sp.shutdown()
    RUNTIME.shutdown()
    torch.ops.ring_attention_ops.shutdown()
    dist.barrier()
    my_allocator.disable()
    for segment, owner in ((shm, 3), (shm_grad, 2), (shm_state, 3)):
        if local_rank == owner:
            segment.unlink()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    main(args)
