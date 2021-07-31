# Utilities for cloning test data, etc.

TESTDATA := $(PWD)/testdata
MNIST := $(TESTDATA)/mnist

.PHONY: all mnist

all: \
	mnist

# Information: http://yann.lecun.com/exdb/mnist/
# Download: https://deepai.org/dataset/mnist
mnist: $(MNIST)/train-labels-idx1-ubyte.gz

$(MNIST)/train-labels-idx1-ubyte.gz: \
		$(PWD)/testdata/mnist/mnist.zip
	unzip -f -d $(dir $@) $<

$(MNIST)/mnist.zip:
	mkdir -p $(dir $@)
	curl -fsSL https://data.deepai.org/mnist.zip -o $@
