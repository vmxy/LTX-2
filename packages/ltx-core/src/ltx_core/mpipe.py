import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

class MPipe:
    def __init__(self, model=None):
        self.rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.model = model
        self.pipeline_model = None
        self.schedule = None
        # 初始化时将模型移动到正确的设备 
        if model is not None:
            device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        
    def is_first_rank(self)->bool:
        return self.rank == 0
    
    def is_last_rank(self)->bool:
        return self.rank == self.world_size - 1
    
    def setup(self, rank=None, world_size=None):
        """
        Setup distributed environment.
        
        Args:
            rank (int): Rank of the current process.
            world_size (int): Number of processes.
        """
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '11355')
        self.rank = rank if rank is not None else self.rank
        self.world_size = world_size if world_size is not None else self.world_size

        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend="nccl", 
            rank=self.rank, 
            world_size=self.world_size,
            device_id=torch.device(f'cuda:{self.rank}'),  # 明确指定设备
        )

    def sync(self):
        """
        同步所有进程
        """
        if self.world_size > 1:
            dist.barrier()

    def cleanup(self):
        """
        Cleanup distributed environment.
        """
        if dist.is_initialized():
            dist.destroy_process_group()

    def get_world_size(self)->int:
        return self.world_size

    def get_device(self)->torch.device:
        """
        Get device for current process.
        
        Returns:
            torch.device: Device for current process.
        """
        return torch.device(f"cuda:{self.rank}")
    
   
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """
        Broadcast tensor from src rank to all other ranks.
        
        Args:
            tensor (torch.Tensor): Tensor to broadcast.
            src (int): Source rank.
        """
        if self.world_size > 1:
            dist.broadcast(tensor, src=src)

    def paralle_model(self, model: nn.Module, blocks=None, init_model_weights=None):
        """
        Create a PipelineParallel model from the given model.
        Args:
            model (torch.nn.Module): Model to parallelize.
            blocks (List[nn.Module]): List of blocks to parallelize.
            
        Returns:
            torch.nn.Module: PipelineParallel model if world_size > 1, original model otherwise.
        """
        if self.world_size <= 1:
            print(f"paralle_model world_size <= 1, return original model {self.get_device()} ")
            # 确保模型在正确的设备上
            model.to(self.get_device())
            return model
        
        device = self.get_device()
        num_stages = self.world_size
        num_microbatches = self.world_size        
        num_blocks = len(blocks)
        blocks_per_stage = num_blocks // num_stages
        
        # 计算分割点
        start_idx = self.rank * blocks_per_stage
        end_idx = (self.rank + 1) * blocks_per_stage if self.rank < num_stages - 1 else num_blocks
        
        # 创建当前阶段的模型
        stage_model = nn.Sequential(*[blocks[i] for i in range(start_idx, end_idx)])
        print(f"stage_model start={start_idx} end={end_idx} num_blocks={num_blocks}")
        stage_model.to_empty(device=device) #非常重要 给模型初始化空权重，分配存储空间
        if init_model_weights is not None:
            stage_model.apply(init_model_weights)
        stage_model.eval()  # 推理模式 ==stage_model.train(False)      
        
        # 创建PipelineStage
        stage = PipelineStage(
            stage_model,
            stage_index=self.rank,
            num_stages=num_stages,
            device=device,
            # ⚠️ 重要：不传input_args，让schedule step时自动推断
            input_args=None,
        )
        
        # 创建调度器
        self.schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches)
        
        return stage    

    def step(self, inputs=None):
        """
        Run the pipeline with the given inputs.
        
        Args:
            inputs: Inputs to the pipeline.
            
        Returns:
            Output of the pipeline.
        """
        if self.world_size <= 1:
            # 确保模型在正确的设备上
            if self.model is not None:
                device = self.get_device()
                self.model.to(device)
                # 确保输入也在正确的设备上
                if inputs is not None:
                    inputs = inputs.to(device)
                return self.model(inputs)
            else:
                raise ValueError("Model not initialized")
        
        if self.schedule is None:
            raise ValueError("Pipeline not initialized. Call parallelize_model first.")
        print(f"stage_index={self.rank} shape={inputs.shape if inputs is not None else 'None'}")
        # 只有第一阶段（rank 0）才传递输入参数
        if self.is_first_rank():
            output = self.schedule.step(inputs)
        else:
            # 非第一阶段不传递输入参数
            output = self.schedule.step()
        return output