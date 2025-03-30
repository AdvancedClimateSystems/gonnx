import onnx
from onnx import numpy_helper

def load_tensor_proto(pb_path):
    with open(pb_path, 'rb') as f:
        tensor_proto = onnx.TensorProto()
        tensor_proto.ParseFromString(f.read())
        tensor_np = numpy_helper.to_array(tensor_proto)
        return tensor_np

path = "./test_data/test_tril_zero/test_data_set_0/"
# Load your input tensor(s)
input_tensor_1 = load_tensor_proto(path + "input_0.pb")
input_tensor_2 = load_tensor_proto(path + "input_1.pb")

print(input_tensor_1.shape, input_tensor_2.shape)
import pdb;pdb.set_trace()
