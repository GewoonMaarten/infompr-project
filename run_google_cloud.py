import tensorflow_cloud as tfc

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog='run_google_cloud.py',
#         description='')
#     parser.add_argument(
#         '--trainer',
#         choices=['dual', 'image', 'text'],
#         required=True,
#         help='which trainer to use')
#     args = parser.parse_args()

tfc.run(
    # entry_point=f"trainers/{file}",
    requirements_txt="requirements.txt",
    distribution_strategy="auto",
    # chief_config=tfc.MachineConfig(
    #     cpu_cores=8,
    #     memory=30,
    #     accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
    #     accelerator_count=1,
    # ),
    chief_config=tfc.COMMON_MACHINE_CONFIGS["K80_1X"],
    # worker_count=0,
    # stream_logs=True
)

import trainers.efficientnet
