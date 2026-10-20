"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Submit `finetuning/finetune_ozone.py` as an Azure Machine Learning command job on a single GPU.

Requires `azure-ai-ml` and `azure-identity`. The workspace is read from an AML `config.json`
(see `MLClient.from_config`); on an AML compute instance this is `~/.azureml/config.json`.

    # Pipeline test on random data:
    python finetuning/submit_aml_ozone.py --compute <gpu-cluster> --synthetic 16 --max-steps 20
    # Real data, previously downloaded with `prepare_cams_ozone.py` to an AML datastore:
    python finetuning/submit_aml_ozone.py --compute <gpu-cluster> \\
        --data-uri azureml://datastores/<datastore>/paths/<path>/ --lr 1e-5
"""

import argparse
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import AmlTokenConfiguration
from azure.identity import DefaultAzureCredential

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = (
    "azureml://registries/azureml/environments/acpt-pytorch-2.8-cuda12.6/versions/19"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit the ozone fine-tuning test to AML.")
    parser.add_argument("--compute", required=True, help="AML GPU cluster (>= 80 GB GPU).")
    parser.add_argument("--data-uri", help="URI of a folder with `prepare_cams_ozone.py` output.")
    parser.add_argument("--synthetic", type=int, help="Use N random time steps instead.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--experiment-name", default="aurora-ozone-finetune")
    parser.add_argument("--config", help="Path to an AML `config.json`.")
    args = parser.parse_args()
    if (args.data_uri is None) == (args.synthetic is None):
        parser.error("Provide exactly one of --data-uri and --synthetic.")

    train_args = f"--max-steps {args.max_steps} --eval-every {args.eval_every} --lr {args.lr}"
    inputs = {}
    if args.synthetic:
        train_args += f" --synthetic {args.synthetic}"
    else:
        inputs["data"] = Input(
            type=AssetTypes.URI_FOLDER, path=args.data_uri, mode=InputOutputModes.RO_MOUNT
        )
        train_args += " --data-dir ${{inputs.data}}"
    if args.dry_run:
        train_args += " --dry-run"

    job = command(
        display_name="aurora-ozone-finetune" + ("-synthetic" if args.synthetic else ""),
        experiment_name=args.experiment_name,
        compute=args.compute,
        environment=args.environment,
        code=str(REPO),
        command=(
            "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --quiet -e . && "
            "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync "
            f"python finetuning/finetune_ozone.py {train_args} --out-dir ${{{{outputs.out}}}}"
        ),
        inputs=inputs,
        outputs={"out": Output(type=AssetTypes.URI_FOLDER)},
        identity=AmlTokenConfiguration(),
        shm_size="16g",
    )

    ml_client = MLClient.from_config(DefaultAzureCredential(), path=args.config)
    returned = ml_client.jobs.create_or_update(job)
    print(f"Submitted {returned.name}: {returned.studio_url}")


if __name__ == "__main__":
    main()
