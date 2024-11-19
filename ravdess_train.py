import os
import torch
import speechbrain as sb
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization

# Custom ECAPA_TDNN model with InstanceNorm1d replacing BatchNorm1d
class CustomECAPA_TDNN(ECAPA_TDNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_batchnorm_with_instancenorm()

    def replace_batchnorm_with_instancenorm(self):
        # Recursively replace all BatchNorm1d with InstanceNorm1d
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                new_module = nn.InstanceNorm1d(
                    module.num_features,
                    affine=module.affine,
                    track_running_stats=False
                )
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                if parent_name == "":
                    setattr(self, attr_name, new_module)
                else:
                    parent_module = dict(self.named_modules())[parent_name]
                    setattr(parent_module, attr_name, new_module)

    def forward(self, x):
        residual = x
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i == 0:
                residual = x
            elif i <= 2:
                residual = torch.cat((residual, x), dim=1)

        x = self.mfa(residual)
        x = self.asp_bn(x)
        x = self.fc(x)
        return x

# Data preparation function
def dataio_prep(data_folder, train_json, test_json):
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        yield label_encoder.encode_label_torch(emo)

    datasets = {
        "train": sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=train_json,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo", "emo_encoded"],
        ),
        "test": sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=test_json,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo", "emo_encoded"],
        ),
    }

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.load_or_create(
        path=os.path.join(data_folder, "label_encoder.txt"),
        from_didatasets=[datasets["train"]],
    )
    return datasets, label_encoder

# Custom Brain
class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        feats = self.hparams.fbank(wavs)
        feats = self.hparams.normalizer(feats, lens)

        feats = feats.transpose(1, 2)

        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        loss = sb.nnet.losses.nll_loss(predictions, batch.emo_encoded)
        return loss

# Main function
if __name__ == "__main__":
    data_folder = "./organized_ravdess"
    train_json = os.path.join(data_folder, "train.json")
    test_json = os.path.join(data_folder, "test.json")
    output_folder = "./results"
    os.makedirs(output_folder, exist_ok=True)

    model = CustomECAPA_TDNN(input_size=80, lin_neurons=192)

    modules = {
        "embedding_model": model,
        "classifier": torch.nn.Linear(3072, 8),
    }

    hparams = {
        "fbank": Fbank(n_mels=80),
        "normalizer": InputNormalization(),
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    brain = EmoIdBrain(
        modules=modules,
        opt_class=lambda x: optimizer,
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=sb.utils.checkpoints.Checkpointer(
            checkpoints_dir=os.path.join(output_folder, "checkpoints"),
            recoverables={"model": model, "optimizer": optimizer},
        ),
    )

    datasets, label_encoder = dataio_prep(data_folder, train_json, test_json)

    brain.fit(
        epoch_counter=range(1, 11),
        train_set=datasets["train"],
        valid_set=None,
    )

    brain.evaluate(datasets["test"])
