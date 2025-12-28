# FCN Segmentation TensorRT backend

This is the library for model inference using the TensorRT engine with stream.

## Generate the ONNX file
This will generate the onnx file in the `onnxs` directory.
```bash
python3 script/export_fcn_to_onnx.py \
        --height 370 \
        --width 1226 \
        --output-dir onnxs
```

## Convert to TensorRT engine
Then use trtexec to compile the .onnx format to TensorRT engine
```bash
trtexec --onnx=./onnxs/fcn_resnet101_370x1226.onnx \
        --saveEngine=./engines/fcn_resnet101_370x1226.engine \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --verbose
```
