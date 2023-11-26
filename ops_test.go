package gonnx

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops/opset13"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/proto"
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
	"test_add_uint8",                    // Opset14
	"test_div_uint8",                    // Opset14
	"test_gru_defaults",                 // Opset14
	"test_gru_batchwise",                // Opset14
	"test_gru_seq_length",               // Opset14
	"test_gru_with_initial_bias",        // Opset14
	"test_mul_uint8",                    // Opset14
	"test_sub_uint8",                    // Opset14
	"test_shape_clip_end",               // Opset15
	"test_shape_clip_start",             // Opset15
	"test_shape_end_1",                  // Opset15
	"test_shape_end_negative_1",         // Opset15
	"test_shape_example",                // Opset15
	"test_shape_start_1",                // Opset15
	"test_shape_start_1_end_2",          // Opset15
	"test_shape_start_1_end_negative_1", // Opset15
	"test_shape_start_negative_1",       // Opset15
	"test_reshape_allowzero_reordered",  // Opset14

	"test_constant_pad",              // Pad is not implemented yet.
	"test_constant_pad_axes",         // Pad is not implemented yet.
	"test_gemm_alpha",                // For gemm in opset 11.
	"test_gemm_default_no_bias",      // For gemm in opset 11.
	"test_gemm_default_scalar_bias",  // For gemm in opset 11.
	"test_relu_expanded_ver18",       // CastLike operator not implemented yet.
	"test_slice_start_out_of_bounds", // ONNX expects nil output, but we throw an error.
	"test_slice_end_out_of_bounds",   // ONNX expects nil output, but we throw an error.
	"test_slice_neg_steps",           // ONNX expects nil output, but we throw an error.
	"test_slice_neg",                 // ONNX expects nil output, but we throw an error.
	"test_transpose_default",         // For transpose in opset 9.

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

	"test_unsqueeze_axis_3",                 // Tests an old version of Unsqueeze (<= 11)
	"test_constantofshape_int_shape_zero",   // Empty tensors are not supported in gorgonia
	"test_gather_elements_0",                // Operator GatherElements is not implemented
	"test_gather_elements_1",                // Operator GatherElements is not implemented
	"test_gather_elements_negative_indices", // Operator GatherElements is not implemented

	"test_prelu_broadcast_expanded",   // Unsupported operator CastLike
	"test_prelu_example_expanded",     // Unsupported operator CastLike
	"test_constant_pad_negative_axes", // Unsupported operator Pad
}

type ONNXTestCase struct {
	name    string
	model   *Model
	inputs  Tensors
	outputs Tensors
}

func TestOps(t *testing.T) {
	runnedTests := []string{}
	opNames := opset13.GetOpNames()

	for _, opName := range opNames {
		tests, err := getTestCasesForOp(opName)
		assert.Nil(t, err)

		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				outputs, err := test.model.Run(test.inputs)
				assert.Nil(t, err)

				for outputName := range test.outputs {
					expectedTensor := test.outputs[outputName]
					actualTensor := outputs[outputName]
					assert.InDeltaSlice(t, expectedTensor.Data(), actualTensor.Data(), 0.00001)
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
	opFilter := fmt.Sprintf("test_%v", strings.ToLower(opName))

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

	// Currently we only implemented Opset13, hence we enforce this in our tests. All
	// tests that fail because of this are ignored.
	mp.OpsetImport[0].Version = 13

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
	"test_asin",
	"test_asin_example",
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
	"test_div",
	"test_div_bcast",
	"test_div_example",
	"test_gather_0",
	"test_gather_1",
	"test_gather_2d_indices",
	"test_gather_negative_indices",
	"test_gemm_default_single_elem_vector_bias",
	"test_gemm_all_attributes",
	"test_gemm_default_matrix_bias",
	"test_gemm_default_vector_bias",
	"test_gemm_transposeA",
	"test_gemm_default_zero_bias",
	"test_gemm_beta",
	"test_gemm_transposeB",
	"test_matmul_4d",
	"test_matmul_3d",
	"test_matmul_2d",
	"test_mul",
	"test_mul_bcast",
	"test_mul_example",
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
	"test_shape",
	"test_sin",
	"test_sin_example",
	"test_sigmoid_example",
	"test_sigmoid",
	"test_slice_negative_axes",
	"test_slice_default_steps",
	"test_slice",
	"test_slice_default_axes",
	"test_squeeze_negative_axes",
	"test_squeeze",
	"test_sub",
	"test_sub_bcast",
	"test_sub_example",
	"test_tanh",
	"test_tanh_example",
	"test_transpose_all_permutations_2",
	"test_transpose_all_permutations_0",
	"test_transpose_all_permutations_1",
	"test_transpose_all_permutations_3",
	"test_transpose_all_permutations_4",
	"test_transpose_all_permutations_5",
	"test_unsqueeze_axis_0",
	"test_unsqueeze_axis_1",
	"test_unsqueeze_axis_2",
	"test_unsqueeze_negative_axes",
	"test_unsqueeze_three_axes",
	"test_unsqueeze_two_axes",
	"test_unsqueeze_unsorted_axes",
}
