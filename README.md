# A Simple Wrapper for Python Deep learning Frameworks

## Rule of network definition files

- The file should be in one of the backend module folder such as "keras", "pytorch".
- The file name should be snake\_case converted from the class name of CamelCase.
- Only one class should be defined in the file.
- The class name should be CamelCase.
  - OK:  class name: FCN,         file name: fcn.py
  - OK:  class name: MobilenetV2, file name: mobilenet\_v2.py
  - NG:  class name: GoogLeNet,   file name: googlenet.py.
    - The file name must be goog\_le\_net.py for this case.
- The class constructor should have four positional arguments, namely `inputs`, `outputs`, `constraints`, and `weights` in this order.
