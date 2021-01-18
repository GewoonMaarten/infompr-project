import argparse
import tensorflow_cloud as tfc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_local.py',
        description='')
    parser.add_argument(
        '--trainer',
        choices=['dual', 'image', 'text'],
        required=True,
        help='which dataset should be used to create the minidataset')
    args = parser.parse_args()

    if args.trainer == 'image':
        file = 'efficientnet.py'
    elif args.trainer == 'text':
        file = 'bert_roberta.py'
    elif args.trainer == 'dual':
        file = 'dual.py'
    else:
        raise ValueError('Trainer does not have a valid value.')

    tfc.run(
        entry_point=f"trainers/{file}",
        requirements_txt="requirements.txt",
        distribution_strategy="auto",
        chief_config=tfc.MachineConfig(
            cpu_cores=8,
            memory=30,
            accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=1,
        ),
        worker_count=0,
        stream_logs=True
    )
