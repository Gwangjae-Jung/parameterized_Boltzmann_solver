nohup python main__pfno_param.py --cuda_index 0 &
nohup python main__prfno_param.py --cuda_index 1 &
wait
nohup python main__fno_param.py --cuda_index 0 &
nohup python main__rfno_param.py --cuda_index 1 &
wait