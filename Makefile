# Utilities for cloning test data, etc.

# Directories for cloning data.
TESTDATA := $(PWD)/testdata
MNIST := $(TESTDATA)/mnist
FASHION := $(TESTDATA)/mnist-fashion

.PHONY: all
all: \
	mnist \
	fashion

.PHONY: clean
clean:
	rm -rf $(TESTDATA)

# Information: http://yann.lecun.com/exdb/mnist/
# Download: https://deepai.org/dataset/mnist
.PHONY: mnist
mnist: $(MNIST)/train-labels-idx1-ubyte.gz

$(MNIST)/train-labels-idx1-ubyte.gz: \
		$(PWD)/testdata/mnist/mnist.zip
	unzip -u -d $(dir $@) $<

$(MNIST)/mnist.zip:
	mkdir -p $(dir $@)
	curl -fsSL https://data.deepai.org/mnist.zip -o $@

# Information: https://github.com/zalandoresearch/fashion-mnist
.PHONY: fashion
fashion: \
	$(FASHION)/train-images-idx3-ubyte.gz \
	$(FASHION)/train-labels-idx1-ubyte.gz \
	$(FASHION)/t10k-images-idx3-ubyte.gz \
	$(FASHION)/t10k-labels-idx1-ubyte.gz

# NOTE: It appears that https is not supported.
FASHION_BASE_URL := http://fashion-mnist.s3-website.eu-central-1.amazonaws.com

$(FASHION)/%.gz:
	mkdir -p $(dir $@)
	curl -fsSL $(FASHION_BASE_URL)/$(notdir $@) -o $@
	# Expected md5 hash values available here:
	# https://github.com/zalandoresearch/fashion-mnist#get-the-data
	md5sum $@
