# GONNX

![image info](./logo.png)

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)


The goal of this package is to provide an easy way of running ONNX models in Go. This package
is intended for inference usage of ONNX models. The package can be used to load an `.onnx` file
and perform inference using the model described by this file.  

Currently, we are implementing ONNX operation set 13, and we plan to add all opsets following this
one as well. Feel free to contribute by implementing operators!

## Getting started

### Installation & setup

First, [install Go on your machine](https://golang.org/doc/install).
Then, clone the repository:

```sh
git clone https://github.com/AdvancedClimateSystems/gonnx.git
cd gonnx
```

Then, install the dependencies as follows:
```sh
make install
make install_lint
make install_gotestsum
```

### Usage
A simple example is shown below:
```go
package main

import (
    "github.com/AdvancedClimateSystems/gonnx"
    "gorgonia.org/tensor"
)


func main() {
    model, err := gonnx.NewModel("./path_to_onnx/model.onnx")
    if err != nil {
        log.Fatal(err)
    }

    var inputs map[string]*tensor.Tensor
    inputs = // However you construct inputs. It must have a tensor for all inputs.

    result, err := model.Run(inputs)
    if err != nil {
        log.Fatal(err)
    }
}
```

### Tests
Most of the code should be tested. 
If you add operators (or an entire opset version) make sure you add unit tests as wel 
as tests for the ONNX test suite

The GONNX test suite consists of unit tests and integration tests, the
[standard](https://github.com/onnx/onnx/blob/master/docs/OnnxBackendTest.md)
provided by ONNX.

The ONNX test data is not stored in the repository, hence one needs to download it first:
```sh
make test_data
```

Now, one can run all tests using (which runs both integration and unit tests):
```sh
make test
```

Because the current implementation is lacking certain opset versions, some of the tests from
ONNX will not run. All tests that are skipped can be found in `ops_test.go`, as well as the
reason that particular test is skipped. We try to be explicit in which tests are ran and which
are skipped.


# How to Contribute
Any kind of contribution is welcome. Try to keep the style consistent and 
make sure all linter checks succeed before opening a pull request.


## Pull requests
Our workflow is based on the [github-flow](https://guides.github.com/introduction/flow/>) .

1. Create a new issue.
2. Fork the project.
3. Clone your fork and add the upstream.
    ```bash
    git remote add upstream https://github.com/AdvancedClimateSystems/gonnx.git
    ```

4. Pull new changes from the upstream.
    ```bash
    git checkout main
    git fetch upstream
    git merge upstream/main
    ```

5. Create a feature branch
    ```bash
    git checkout -b <branch-name>
    ```

6. Commit your changes and reference the issue number in your comment.
    ```bash
    git commit -m "Issue #<issue-ref> : <your message>"
    ```

7. Push the feature branch to your remote repository.
    ```bash
    git push origin <branch-name>
    ```

8. Open new pull request.


# Code of conduct
Welcome to the GONNX community! To make sure everyone has a blast while working together,
here's our quick code of conduct:

1. Be respectful and kind to everyone. No hate speech or harassment allowed.
2. Keep conversations on topic and respectful.
3. Respect other people's work and intellectual property rights.
4. Follow the guidelines for contributing to the project and make your first pull request! Anything is welcome :slightly_smiling_face:
5. If you see something, say something. Report any violations to the maintainers.
6. Have fun! We're all here to work on something awesome together, so let's make the most of it!

By participating in the GONNX project, you're agreeing to follow these guidelines and
make sure everyone has a good time. Whether you're a seasoned veteran or a newcomer,
we're excited to have you here. Let's make something amazing!

# Authors

- [@wipsel](https://www.github.com/wipsel)
- [@swopper050](https://www.github.com/swopper050)
- [@stensipma](https://www.github.com/stensipma)
