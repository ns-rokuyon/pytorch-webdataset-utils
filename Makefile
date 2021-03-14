all: build

build:
	@python setup.py bdist_wheel

clean:
	$(RM) -r build dist *.egg-info

.PHONY: all build clean