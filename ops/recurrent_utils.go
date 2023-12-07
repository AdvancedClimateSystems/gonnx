package ops

// SequenceProcessDirection is the direction in which a sequential input is processed.
// We can process sequential inputs forward (from first to last), in reverse (from
// last to first) or bidirectional (which is both forward and reverse added together).
type SequenceProcessDirection string

const (
	Forward       SequenceProcessDirection = "forward"
	Reverse       SequenceProcessDirection = "reverse"
	Bidirectional SequenceProcessDirection = "bidirectional"
)

// These constants define attributes that are applicable to GRU, LSTM and RNN operators.
const (
	ActivationAlphaAttr = "activation_alpha"
	ActivationBetaAttr  = "activation_beta"
	ActivationsAttr     = "activations"
	ClipAttr            = "clip"
	DirectionAttr       = "direction"
	HiddenSizeAttr      = "hidden_size"
)
