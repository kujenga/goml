# ML in Go

[![GoDoc reference example](https://img.shields.io/badge/godoc-reference-blue.svg)](https://pkg.go.dev/github.com/kujenga/goml)
[![CircleCI](https://circleci.com/gh/kujenga/goml/tree/main.svg?style=svg)](https://circleci.com/gh/kujenga/goml/tree/main)
[![codecov](https://codecov.io/gh/kujenga/goml/branch/main/graph/badge.svg?token=JD4534GVK7)](https://codecov.io/gh/kujenga/goml)

This repository contains various ML experiments written in Go. It endeavors to
use just the standard library, with a handful of exceptions around testing, to
keep things simpler and all in one place.

The goal of this project is to facilitate understanding of various ML
techniques, and is not intended for any sort of production-like usage.

## Packages

- [`neural`](./neural): Implementations of neural networks.
- [`lin`](./lin): Data structures and functions for linear algebra.
- [`mnist`](./mnist): Parsed form of the [MNIST][MNISTArchive] handwriting dataset.
- [`idx`](./idx): Parser for the idx data format used by [`mnist`](./mnist).

## Running tests

You can run this repository's tests with the following commands:

```sh
# Download assets used in testing.
make
# Run the tests.
go test ./...
```


<!-- Links -->
[MNISTArchive]: https://web.archive.org/web/20211125025603/http://yann.lecun.com/exdb/mnist/
