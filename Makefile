run_all: set_up run_21 run_22 run_23 test_batch_size test_learning_rate
	python main.py --model BoringNet --epochs 50

run_21: FORCE
	python main.py --model LazyNet --epochs 50
	mkdir records/run_21
	mv logs/ records/run_21/

run_22: FORCE
	python main.py --model BoringNet --epochs 50
	mkdir records/run_22
	mv logs/ records/run_22/

run_23: FORCE
	python main.py --model CoolNet --epochs 50
	mkdir records/run_23
	mv logs/ records/run_23/

test_batch_size: FORCE
	python main.py --model CoolNet --epoch 50 --batchSize 16

	mkdir records/test_batch_16
	mv logs/ records/test_batch_16/

	python main.py --model CoolNet --epoch 50 --batchSize 32

	mkdir records/test_batch_32
	mv logs/ records/test_batch_32/

	python main.py --model CoolNet --epoch 50 --batchSize 64

	mkdir records/test_batch_64
	mv logs/ records/test_batch_64/

	python main.py --model CoolNet --epoch 50 --batchSize 128

	mkdir records/test_batch_128
	mv logs/ records/test_batch_128/

	python main.py --model CoolNet --epoch 50 --batchSize 256

	mkdir records/test_batch_256
	mv logs/ records/test_batch_256/

test_learning_rate: FORCE
	python main.py --lr 10 --model CoolNet --epoch 50

	mkdir records/test_learning_rate_10
	mv logs/ records/test_learning_rate_10/

	python main.py --lr 0.1 --model CoolNet --epoch 50

	mkdir records/test_learning_rate_0.1
	mv logs/ records/test_learning_rate_0.1/

	python main.py --lr 0.01 --model CoolNet --epoch 50

	mkdir records/test_learning_rate_0.01
	mv logs/ records/test_learning_rate_0.01/

	python main.py --lr 0.0001 --model CoolNet --epoch 50

	mkdir records/test_learning_rate_0.0001
	mv logs/ records/test_learning_rate_0.0001/

set_up: FORCE
	mkdir records/

# Phony target to force clean
FORCE: ;

