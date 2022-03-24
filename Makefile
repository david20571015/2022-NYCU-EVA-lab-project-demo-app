.PHONY: build run clean

build:
	pyrcc5 -o ./src/resources/icons.py ./src/resources/icons.qrc
run:
	python ./src/app.py
clean:
	rm -rf ./src/resources/icons.py
	find . | grep -E "\(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf