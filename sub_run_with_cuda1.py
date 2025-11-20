import  subprocess
CUDA_INDEX = 1
for file in (
        "ablation__pfno_param.py",
        "ablation__prfno_param.py",
    ):
    subprocess.run(
        [
            "python",       file,
            "--cuda_index", str(CUDA_INDEX),
        ]
    )