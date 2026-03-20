from learning_llms_from_first_principles.trainer.classify import main as classify_main
from learning_llms_from_first_principles.trainer.instruct_finetuning import (
    main as instruct_finetuning_main,
)
from learning_llms_from_first_principles.trainer.train import main

__all__ = ["main", "classify_main", "instruct_finetuning_main"]
