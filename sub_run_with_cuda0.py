import  subprocess
CUDA_INDEX = 0
for file in (
        # "ablation__fno_param.py",
        # "ablation__rfno_param.py",
        # "ablation__pfno_param.py",
        # "ablation__prfno_param.py",
    ):
    print(f"Run [{file}] with [cuda:{CUDA_INDEX}]")
    subprocess.run(
        [
            "python",       file,
            "--cuda_index", str(CUDA_INDEX),
        ]
    )