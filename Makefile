.PHONY: run chat eval test

run:
	python3 -m src.main --pid-a "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf" --pid-b "data/samples/pair_01/Lift Gas compressor-P&ID.pdf"

chat:
	python3 -m src.main --pid-a "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf" --pid-b "data/samples/pair_01/Lift Gas compressor-P&ID.pdf" --chat

eval:
	python3 -m eval.run_eval

test:
	pytest tests/ -v
