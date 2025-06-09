# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, List, Optional

from ...data import SingleDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from ..trainer_utils import create_ref_model
from .trainer import MaskGoalTrainer
from ..callbacks import fix_valuehead_checkpoint

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_outlier_check(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    stage="orm_dg"
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage=stage, **tokenizer_module)
    if stage == "orm_dg":
        model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    else:
        model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    #data_collator = PairwiseDataCollatorWithPadding(template=template, pad_to_multiple_of=8, **tokenizer_module)
    data_collator = SingleDataCollatorWithPadding(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module
    )

    # Update arguments
    training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset

    # Create reference model
    if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
        ref_model = model
    else:
        ref_model = create_ref_model(model_args, finetuning_args)

    # Initialize our Trainer
    trainer = MaskGoalTrainer(
        model=model,
        ref_model=ref_model,
        stage="outlier_check",
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        #compute_metrics=ComputeAccuracy(),
        **dataset_module,
        **tokenizer_module
    )

    predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
    trainer.save_pred_features(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)