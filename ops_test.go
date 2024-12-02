package gonnx

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/proto"
	"gorgonia.org/tensor"
)

// Currently we ignore some of tests provided by ONNX. This has to do with the
// fact that we started implementing from opset 13 and higher. Some of the tests
// however use opsets from lower versions, which crashes with our implementation.
// We do not want to exclude all tests for operators with a lower opset version, as
// most of the tests are still valid, hence we exclude some specific tests.
//
// Another reason is that some tests require an opset version higher than we have currently
// implemented, or lower, which we also haven't implemented yet.
var ignoredTests = []string{
	"test_add_uint8",                                 // Opset14
	"test_div_uint8",                                 // Opset14
	"test_gru_batchwise",                             // Opset14
	"test_logsoftmax_axis_1_expanded_ver18",          // Opset18
	"test_logsoftmax_example_1_expanded_ver18",       // Opset18
	"test_logsoftmax_negative_axis_expanded_ver18",   // Opset18
	"test_logsoftmax_large_number_expanded_ver18",    // Opset18
	"test_logsoftmax_default_axis_expanded_ver18",    // Opset18
	"test_logsoftmax_axis_0_expanded_ver18",          // Opset18
	"test_logsoftmax_axis_2_expanded_ver18",          // Opset18
	"test_lstm_batchwise",                            // Opset14
	"test_mul_uint8",                                 // Opset14
	"test_reduce_max_empty_set",                      // Opset20
	"test_reduce_max_do_not_keepdims_random",         // Opset18
	"test_reduce_max_keepdims_random",                // Opset18
	"test_reduce_max_default_axes_keepdims_random",   // Opset18
	"test_reduce_max_do_not_keepdims_example",        // Opset18
	"test_reduce_max_default_axes_keepdim_example",   // Opset18
	"test_reduce_max_negative_axes_keepdims_random",  // Opset18
	"test_reduce_max_negative_axes_keepdims_example", // Opset18
	"test_reduce_max_bool_inputs",                    // Opset20
	"test_reduce_max_keepdims_example",               // Opset18
	"test_reduce_min_keepdims_random",                // Opset18
	"test_reduce_min_keepdims_example",               // Opset18
	"test_reduce_min_do_not_keepdims_example",        // Opset18
	"test_reduce_min_negative_axes_keepdims_example", // Opset18
	"test_reduce_min_bool_inputs",                    // Opset18
	"test_reduce_min_do_not_keepdims_random",         // Opset18
	"test_reduce_min_default_axes_keepdims_example",  // Opset18
	"test_reduce_min_empty_set",                      // Opset18
	"test_reduce_min_default_axes_keepdims_random",   // Opset18
	"test_reduce_min_negative_axes_keepdims_random",  // Opset18
	"test_sub_uint8",                                 // Opset14
	"test_shape_clip_end",                            // Opset15
	"test_shape_clip_start",                          // Opset15
	"test_shape_end_1",                               // Opset15
	"test_shape_end_negative_1",                      // Opset15
	"test_shape_example",                             // Opset15
	"test_shape_start_1",                             // Opset15
	"test_shape_start_1_end_2",                       // Opset15
	"test_shape_start_1_end_negative_1",              // Opset15
	"test_shape_start_negative_1",                    // Opset15
	"test_softmax_default_axis_expanded_ver18",       // Opset18
	"test_softmax_axis_1_expanded_ver18",             // Opset18
	"test_softmax_negative_axis_expanded_ver18",      // Opset18
	"test_softmax_example_expanded_ver18",            // Opset18
	"test_softmax_axis_0_expanded_ver18",             // Opset18
	"test_softmax_large_number_expanded_ver18",       // Opset18
	"test_softmax_axis_2_expanded_ver18",             // Opset18
	"test_reshape_allowzero_reordered",               // Opset14

	"test_constant_pad",                      // Pad is not implemented yet.
	"test_constant_pad_axes",                 // Pad is not implemented yet.
	"test_logsoftmax_large_number_expanded",  // Requires 'Exp' operator.
	"test_logsoftmax_axis_0_expanded",        // Requires 'Exp' operator.
	"test_logsoftmax_axis_1_expanded",        // Requires 'Exp' operator.
	"test_logsoftmax_axis_2_expanded",        // Requires 'Exp' operator.
	"test_logsoftmax_example_1_expanded",     // Requires 'Exp' operator.
	"test_logsoftmax_default_axis_expanded",  // Requires 'Exp' operator.
	"test_logsoftmax_negative_axis_expanded", // Requires 'Exp' operator.
	"test_lstm_with_peepholes",               // Sequence lens attribute is not supported yet.
	"test_relu_expanded_ver18",               // CastLike operator not implemented yet.
	"test_softmax_axis_0_expanded",           // Requires 'Exp' operator.
	"test_softmax_negative_axis_expanded",    // Requires 'Exp' operator.
	"test_softmax_large_number_expanded",     // Requires 'Exp' operator.
	"test_softmax_axis_1_expanded",           // Requires 'Exp' operator.
	"test_softmax_example_expanded",          // Requires 'Exp' operator.
	"test_softmax_axis_2_expanded",           // Requires 'Exp' operator.
	"test_softmax_default_axis_expanded",     // Requires 'Exp' operator.
	"test_slice_start_out_of_bounds",         // ONNX expects nil output, but we throw an error.
	"test_slice_end_out_of_bounds",           // ONNX expects nil output, but we throw an error.
	"test_slice_neg_steps",                   // ONNX expects nil output, but we throw an error.
	"test_slice_neg",                         // ONNX expects nil output, but we throw an error.

	"test_equal_string",                               // Unsupported datatype String.
	"test_equal_string_broadcast",                     // Unsupported datatype String.
	"test_cast_INT4_to_INT8",                          // Unsupported datatype INT4.
	"test_cast_INT4_to_FLOAT",                         // Unsupported datatype INT4.
	"test_cast_FLOAT_to_INT4",                         // Unsupported datatype INT4.
	"test_cast_FLOAT_to_UINT4",                        // Unsupported datatype UINT4.
	"test_cast_INT4_to_FLOAT16",                       // Unsupported datatype INT4/FLOAT16.
	"test_cast_FLOAT16_to_UINT4",                      // Unsupported datatype FLOAT16.
	"test_cast_FLOAT16_to_INT4",                       // Unsupported datatype FLOAT16.
	"test_cast_UINT4_to_UINT8",                        // Unsupported datatype UINT4.
	"test_cast_UINT4_to_FLOAT",                        // Unsupported datatype UINT4.
	"test_cast_UINT4_to_FLOAT16",                      // Unsupported datatype UINT4.
	"test_cast_FLOAT_to_STRING",                       // Unsupported datatype STRING.
	"test_cast_STRING_to_FLOAT",                       // Unsupported datatype STRING.
	"test_cast_DOUBLE_to_FLOAT16",                     // Unsupported datatype FLOAT16.
	"test_cast_FLOAT_to_FLOAT16",                      // Unsupported datatype FLOAT16.
	"test_cast_FLOAT16_to_DOUBLE",                     // Unsupported datatype FLOAT16.
	"test_cast_FLOAT16_to_FLOAT",                      // Unsupported datatype FLOAT16.
	"test_cast_BFLOAT16_to_FLOAT",                     // Unsupported datatype BFLOAT16.
	"test_cast_FLOAT_to_BFLOAT16",                     // Unsupported datatype BFLOAT16.
	"test_cast_FLOAT_to_FLOAT8E5M2",                   // Unsupported datatype.
	"test_cast_FLOAT_to_FLOAT8E4M3FN",                 // Unsupported datatype.
	"test_cast_FLOAT_to_FLOAT8E4M3FNUZ",               // Unsupported datatype FLOAT8E4M3FNUZ.
	"test_cast_FLOAT_to_FLOAT8E5M2FNUZ",               // Unsupported datatype.
	"test_cast_FLOAT16_to_FLOAT8E5M2",                 // Unsupported datatype.
	"test_cast_FLOAT16_to_FLOAT8E4M3FN",               // Unsupported datatype.
	"test_cast_FLOAT16_to_FLOAT8E4M3FNUZ",             // Unsupported datatype.
	"test_cast_FLOAT16_to_FLOAT8E5M2FNUZ",             // Unsupported datatype.
	"test_cast_FLOAT8E5M2_to_FLOAT",                   // Unsupported datatype.
	"test_cast_FLOAT8E5M2_to_FLOAT16",                 // Unsupported datatype.
	"test_cast_FLOAT8E4M3FN_to_FLOAT",                 // Unsupported datatype.
	"test_cast_FLOAT8E4M3FN_to_FLOAT16",               // Unsupported datatype.
	"test_cast_FLOAT8E4M3FNUZ_to_FLOAT",               // Unsupported datatype.
	"test_cast_FLOAT8E4M3FNUZ_to_FLOAT16",             // Unsupported datatype.
	"test_cast_FLOAT8E5M2FNUZ_to_FLOAT",               // Unsupported datatype.
	"test_cast_FLOAT8E5M2FNUZ_to_FLOAT16",             // Unsupported datatype.
	"test_cast_no_saturate_FLOAT_to_FLOAT8E5M2",       // Unsupported datatype FLOAT8E5M2.
	"test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ",   // Unsupported datatype.
	"test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ",   // Unsupported datatype.
	"test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN",     // Unsupported datatype.
	"test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ", // Unsupported datatype.
	"test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ", // Unsupported datatype.
	"test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN",   // Unsupported datatype.
	"test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2",     // Unsupported datatype.

	"test_constantofshape_int_shape_zero",   // Empty tensors are not supported in gorgonia
	"test_gather_elements_0",                // Operator GatherElements is not implemented
	"test_gather_elements_1",                // Operator GatherElements is not implemented
	"test_gather_elements_negative_indices", // Operator GatherElements is not implemented

	"test_prelu_broadcast_expanded",   // Unsupported operator CastLike
	"test_prelu_example_expanded",     // Unsupported operator CastLike
	"test_constant_pad_negative_axes", // Unsupported operator Pad

	"test_argmax_keepdims_random_select_last_index",                // Unsupported attribute
	"test_argmax_keepdims_example_select_last_index",               // Unsupported attribute
	"test_argmax_no_keepdims_example_select_last_index",            // Unsupported attribute
	"test_argmax_no_keepdims_random_select_last_index",             // Unsupported attribute
	"test_argmax_default_axis_example_select_last_index",           // Unsupported attribute
	"test_argmax_default_axis_random_select_last_index",            // Unsupported attribute
	"test_argmax_negative_axis_keepdims_example_select_last_index", // Unsupported attribute
	"test_argmax_negative_axis_keepdims_random_select_last_index",  // Unsupported attribute
}

type ONNXTestCase struct {
	name    string
	model   *Model
	inputs  Tensors
	outputs Tensors
}

func TestOps(t *testing.T) {
	runnedTests := []string{}

	for opName := range operators {
		tests, err := getTestCasesForOp(opName)
		assert.Nil(t, err)

		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				outputs, err := test.model.Run(test.inputs)
				assert.Nil(t, err)

				for outputName := range test.outputs {
					expectedTensor := test.outputs[outputName]
					actualTensor := outputs[outputName]

					if expectedTensor.Dtype() == tensor.Bool {
						assert.ElementsMatch(t, expectedTensor.Data(), actualTensor.Data())
					} else {
						assert.InDeltaSlice(t, expectedTensor.Data(), actualTensor.Data(), 0.00001)
					}
				}
			})

			runnedTests = append(runnedTests, test.name)
		}
	}

	sort.Strings(expectedTests)
	sort.Strings(runnedTests)

	assert.Equal(t, expectedTests, runnedTests)
}

func getTestCasesForOp(opName string) ([]*ONNXTestCase, error) {
	testOpName := strings.ToLower(opName)
	// Because the naming of the ONNX test cases are not fully consistent, we need
	// to map some operator names to insert some '_' in the filter.
	if mappedFilter, ok := opNameMap[testOpName]; ok {
		testOpName = mappedFilter
	}

	opFilter := fmt.Sprintf("test_%v", testOpName)

	testDir, err := os.Open("./test_data")
	if err != nil {
		return nil, err
	}

	testFolders, err := testDir.Readdirnames(0)
	if err != nil {
		return nil, err
	}

	var tests []*ONNXTestCase

	for _, testFolder := range testFolders {
		if shouldRunTest(testFolder, opFilter) {
			testcase, err := getTestCase(fmt.Sprintf("./test_data/%v", testFolder))
			if err != nil {
				return nil, err
			}

			testcase.name = testFolder
			tests = append(tests, testcase)
		}
	}

	return tests, nil
}

func shouldRunTest(folder, opFilter string) bool {
	for _, ignoredTest := range ignoredTests {
		if folder == ignoredTest {
			return false
		}
	}

	if strings.Contains(folder, opFilter) {
		remaining := strings.ReplaceAll(folder, opFilter, "")
		if len(remaining) == 0 || remaining[:1] == "_" {
			return true
		}
	}

	return false
}

func getTestCase(folder string) (*ONNXTestCase, error) {
	testcase := &ONNXTestCase{}

	model, err := readTestModel(folder)
	if err != nil {
		return nil, err
	}

	basePath := fmt.Sprintf("%v/test_data_set_0", folder)

	inputs, err := readTestTensors(basePath, "input", model.mp.Graph.GetInput())
	if err != nil {
		return nil, err
	}

	outputs, err := readTestTensors(basePath, "output", model.mp.Graph.GetOutput())
	if err != nil {
		return nil, err
	}

	testcase.model = model
	testcase.inputs = inputs
	testcase.outputs = outputs

	return testcase, nil
}

func readTestModel(folder string) (*Model, error) {
	file, err := os.Open(folder + "/model.onnx")
	if err != nil {
		return nil, err
	}

	bytesModel, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	mp, err := ModelProtoFromBytes(bytesModel)
	if err != nil {
		return nil, err
	}

	// Currently we support Opset 7-13, hence we enforce this in our tests. All
	// tests that fail because of this are ignored.
	if mp.OpsetImport[0].Version < MinSupportedOpset {
		mp.OpsetImport[0].Version = MinSupportedOpset
	} else if mp.OpsetImport[0].Version > MaxSupportedOpset {
		mp.OpsetImport[0].Version = MaxSupportedOpset
	}

	model, err := NewModel(mp)
	if err != nil {
		return nil, err
	}

	return model, nil
}

func readTestTensors(basePath, baseFile string, inputs []*onnx.ValueInfoProto) (Tensors, error) {
	tensors := make(Tensors)

	for i := 0; i < len(inputs); i++ {
		filePath := fmt.Sprintf("%v/%v_%d.pb", basePath, baseFile, i)

		file, err := os.Open(filePath)
		if err != nil {
			return nil, err
		}

		bytesInput, err := io.ReadAll(file)
		if err != nil {
			return nil, err
		}

		tp := &onnx.TensorProto{}
		if err := proto.Unmarshal(bytesInput, tp); err != nil {
			return nil, err
		}

		t, err := onnx.TensorFromProto(tp)
		if err != nil {
			return nil, err
		}

		tensors[inputs[i].GetName()] = t
	}

	return tensors, nil
}

// With this we check if we truly run all tests we expected from the integration test.
var expectedTests = []string{
	"test_abs",
	"test_acos",
	"test_acos_example",
	"test_acosh",
	"test_acosh_example",
	"test_add",
	"test_add_bcast",
	"test_and_bcast3v1d",
	"test_and_bcast3v2d",
	"test_and_bcast4v2d",
	"test_and_bcast4v3d",
	"test_and_bcast4v4d",
	"test_argmax_default_axis_example",
	"test_argmax_default_axis_random",
	"test_argmax_keepdims_example",
	"test_argmax_keepdims_random",
	"test_argmax_negative_axis_keepdims_example",
	"test_argmax_negative_axis_keepdims_random",
	"test_argmax_no_keepdims_example",
	"test_argmax_no_keepdims_random",
	"test_asin",
	"test_asin_example",
	"test_asinh",
	"test_asinh_example",
	"test_atan",
	"test_atan_example",
	"test_atanh",
	"test_atanh_example",
	"test_cast_DOUBLE_to_FLOAT",
	"test_cast_FLOAT_to_DOUBLE",
	"test_concat_1d_axis_0",
	"test_concat_1d_axis_negative_1",
	"test_concat_2d_axis_0",
	"test_concat_2d_axis_1",
	"test_concat_2d_axis_negative_1",
	"test_concat_2d_axis_negative_2",
	"test_concat_3d_axis_0",
	"test_concat_3d_axis_1",
	"test_concat_3d_axis_2",
	"test_concat_3d_axis_negative_1",
	"test_concat_3d_axis_negative_2",
	"test_concat_3d_axis_negative_3",
	"test_constant",
	"test_constantofshape_float_ones",
	"test_constantofshape_int_zeros",
	"test_conv_with_autopad_same",
	"test_conv_with_strides_and_asymmetric_padding",
	"test_conv_with_strides_no_padding",
	"test_conv_with_strides_padding",
	"test_cos",
	"test_cos_example",
	"test_cosh",
	"test_cosh_example",
	"test_div",
	"test_div_bcast",
	"test_div_example",
	"test_equal",
	"test_equal_bcast",
	"test_expand_dim_changed",
	"test_expand_dim_unchanged",
	"test_flatten_axis0",
	"test_flatten_axis1",
	"test_flatten_axis2",
	"test_flatten_axis3",
	"test_flatten_default_axis",
	"test_flatten_negative_axis1",
	"test_flatten_negative_axis2",
	"test_flatten_negative_axis3",
	"test_flatten_negative_axis4",
	"test_gather_0",
	"test_gather_1",
	"test_gather_2d_indices",
	"test_gather_negative_indices",
	"test_gemm_default_single_elem_vector_bias",
	"test_gemm_all_attributes",
	"test_gemm_alpha",
	"test_gemm_default_matrix_bias",
	"test_gemm_default_no_bias",
	"test_gemm_default_scalar_bias",
	"test_gemm_default_vector_bias",
	"test_gemm_transposeA",
	"test_gemm_default_zero_bias",
	"test_gemm_beta",
	"test_gemm_transposeB",
	"test_greater",
	"test_greater_bcast",
	"test_greater_equal",
	"test_greater_equal_bcast",
	"test_greater_equal_bcast_expanded",
	"test_greater_equal_expanded",
	"test_gru_defaults",
	"test_gru_seq_length",
	"test_gru_with_initial_bias",
	"test_less",
	"test_less_bcast",
	"test_less_equal",
	"test_less_equal_bcast",
	"test_less_equal_bcast_expanded",
	"test_less_equal_expanded",
	"test_logsoftmax_axis_0",
	"test_logsoftmax_axis_1",
	"test_logsoftmax_axis_2",
	"test_logsoftmax_default_axis",
	"test_logsoftmax_example_1",
	"test_logsoftmax_large_number",
	"test_logsoftmax_negative_axis",
	"test_lstm_defaults",
	"test_lstm_with_initial_bias",
	"test_matmul_4d",
	"test_matmul_3d",
	"test_matmul_2d",
	"test_mul",
	"test_mul_bcast",
	"test_mul_example",
	"test_not_2d",
	"test_not_3d",
	"test_not_4d",
	"test_or_bcast3v1d",
	"test_or_bcast3v2d",
	"test_or_bcast4v2d",
	"test_or_bcast4v3d",
	"test_or_bcast4v4d",
	"test_prelu_broadcast",
	"test_prelu_example",
	"test_relu",
	"test_reshape_extended_dims",
	"test_reshape_negative_dim",
	"test_reshape_negative_extended_dims",
	"test_reshape_one_dim",
	"test_reshape_reduced_dims",
	"test_reshape_reordered_all_dims",
	"test_reshape_reordered_last_dims",
	"test_reshape_zero_and_negative_dim",
	"test_reshape_zero_dim",
	"test_rnn_seq_length",
	"test_shape",
	"test_sin",
	"test_sin_example",
	"test_sigmoid_example",
	"test_sigmoid",
	"test_sinh",
	"test_sinh_example",
	"test_slice_negative_axes",
	"test_slice_default_steps",
	"test_slice",
	"test_slice_default_axes",
	"test_softmax_axis_0",
	"test_softmax_axis_1",
	"test_softmax_axis_2",
	"test_softmax_default_axis",
	"test_squeeze_negative_axes",
	"test_softmax_example",
	"test_softmax_large_number",
	"test_softmax_negative_axis",
	"test_squeeze",
	"test_sub",
	"test_sub_bcast",
	"test_sub_example",
	"test_tan",
	"test_tan_example",
	"test_tanh",
	"test_tanh_example",
	"test_transpose_all_permutations_2",
	"test_transpose_all_permutations_0",
	"test_transpose_all_permutations_1",
	"test_transpose_all_permutations_3",
	"test_transpose_all_permutations_4",
	"test_transpose_all_permutations_5",
	"test_transpose_default",
	"test_unsqueeze_axis_0",
	"test_unsqueeze_axis_1",
	"test_unsqueeze_axis_2",
	"test_unsqueeze_negative_axes",
	"test_unsqueeze_three_axes",
	"test_unsqueeze_two_axes",
	"test_unsqueeze_unsorted_axes",
	"test_xor_bcast3v1d",
	"test_xor_bcast3v2d",
	"test_xor_bcast4v2d",
	"test_xor_bcast4v3d",
	"test_xor_bcast4v4d",
}

var opNameMap = map[string]string{
	"reducemax": "reduce_max",
	"reducemin": "reduce_min",
}
