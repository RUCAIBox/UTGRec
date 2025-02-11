
import torch
import torch.nn as nn
import transformers
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled, logging
from packaging import version



if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False



logger = logging.get_logger(__name__)

class CustomizedTrainer(Trainer):


    def create_optimizer(self):


        # if self.model_wrapped.lora:
        #     return super().create_optimizer()




        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:

            logger.info("Using different learning rates for LLM and other parameters")

            decay_parameters = self.get_decay_parameter_names(opt_model)
            special_lr_parameters = [name for name, _ in opt_model.named_parameters() if
                                     any(module_keyword in name for module_keyword in ["llm.", "decoder_embeddings", "lm_head"])]
            # logger.info(special_lr_parameters[:10])
            # logger.info(special_lr_parameters[-10:])
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n in special_lr_parameters and n in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.llm_learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n in special_lr_parameters and n not in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.llm_learning_rate,
                    "weight_decay": 0.0,
                },

                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n not in special_lr_parameters and n in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n not in special_lr_parameters and n not in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                },

            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.pop("lr")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer






