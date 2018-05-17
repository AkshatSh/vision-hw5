run_all: set_up run_21 run_22 run_23 test_batch_size test_learning_rate
	python main.py --model BoringNet --epochs 50 --logdir run_start

run_21: FORCE
	python main.py --model LazyNet --epochs 50 --logdir run_21

run_22: FORCE
	python main.py --model BoringNet --epochs 50 --logdir run_22

run_23: FORCE
	python main.py --model CoolNet --epochs 50 --logdir run_23

test_batch_size: FORCE
	python main.py --model CoolNet --epoch 50 --batchSize 16 --logdir batch_16

	python main.py --model CoolNet --epoch 50 --batchSize 32 --logdir batch_32

	python main.py --model CoolNet --epoch 50 --batchSize 64 --logdir batch_64

	python main.py --model CoolNet --epoch 50 --batchSize 128 --logdir batch_128

	python main.py --model CoolNet --epoch 50 --batchSize 256 --logdir batch_256

test_learning_rate: FORCE
	python main.py --lr 10 --model CoolNet --epoch 50  --logdir lr_10

	python main.py --lr 0.1 --model CoolNet --epoch 50 --logdir lr_0.1

	python main.py --lr 0.01 --model CoolNet --epoch 50 --logdir lr_0.01

	python main.py --lr 0.0001 --model CoolNet --epoch 50 --logdir lr_0.0001

set_up: FORCE
	mkdir records/

# Phony target to force clean
FORCE: ;

