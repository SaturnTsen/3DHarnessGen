from harness_training.data.trajectory import TrajectoryDataset
from harness_training.models.qwen import QwenModel
from harness_training.templates.qwen import QwenTemplate
from harness_training.training.collator import SFTCollator
from harness_training.training.methods.sft import SFTMethod


def test_native_chunked_loss_preserves_multimodal_lora_gradients(config, tiny_qwen_path):
    """Check equal objectives and gradients using TRL's public loss_type option."""
    import torch
    from datasets import Dataset
    from trl import SFTTrainer

    config.model.path = tiny_qwen_path
    config.model.lora.rank = 2
    config.model.lora.alpha = 4
    config.model.lora.dropout = 0
    config.model.lora.target_modules = ["q_proj", "v_proj"]
    config.method.use_cpu = True
    config.method.gradient_checkpointing = False
    config.method.backend.eval_strategy = "no"
    config.method.backend.do_eval = False
    data = TrajectoryDataset(config.data, config.experiment.scene, config.experiment.seed).prepare()
    results = []
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for loss_type in ["nll", "chunked_nll"]:
            torch.manual_seed(config.experiment.seed)
            config.method.backend.loss_type = loss_type
            model = QwenModel(config.model, config.template)
            template = QwenTemplate(config.template, model.processor)
            collator = SFTCollator(
                template, model.processor.tokenizer.pad_token_id, data.train,
                label_ignore_index=-100, attention_padding_value=0, token_type_padding_value=0,
            )
            backend = SFTTrainer(
                model=model.load(), processing_class=model.processor,
                args=SFTMethod(config.method, config.experiment.seed).build_arguments(
                    config.experiment.output_dir / loss_type, None),
                train_dataset=Dataset.from_dict({"example_index": [0]}),
                data_collator=collator, **config.method.trainer.model_dump(),
            )
            inputs = collator([{"example_index": 0}])
            output = backend.model(**inputs)
            output.loss.backward()
            gradients = {name: parameter.grad.detach().clone()
                         for name, parameter in backend.model.named_parameters() if parameter.requires_grad}
            results.append((output.loss.detach(), gradients))
    finally:
        torch.set_num_threads(threads)
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-5, atol=1e-6)
    assert any(gradient.count_nonzero() for name, gradient in results[1][1].items() if "lora_B" in name)
    assert results[0][1].keys() == results[1][1].keys()
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=1e-4, atol=1e-6)
