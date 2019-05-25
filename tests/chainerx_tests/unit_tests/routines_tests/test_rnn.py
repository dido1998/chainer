import chainer

import chainerx

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils

n_step_lstm_dtypes_valid = dtype_utils._permutate_dtype_mapping([
    # Floats.
    (('float16', 'float16', 'float16', 'float16', 'float16'),
     ('float16', 'float16', 'float16')),
    (('float32', 'float32', 'float32', 'float32', 'float32'),
     ('float32', 'float32', 'float32')),
    (('float64', 'float64', 'float64', 'float64', 'float64'),
     ('float64', 'float64', 'float64')),
])


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'n_layers,hidden_size,input_size,batches', [
                (1, 2, 1, (1, 1, 1)),
                (2, 6, 8, (4, 2, 2)),
                (3, 8, 4, (4, 2, 1)),
                (4, 12, 4, (4, 3, 2)),

            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', n_step_lstm_dtypes_valid)
    ])
))
@chainer.testing.parameterize_pytest('cover_all', [True, False])
class TestNStepLstm(op_utils.ChainerOpTest):

    def setup(self):
        if (self.in_dtypes[0] == 'float16'):
            self.check_forward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-3, 'atol': 5e-2})

    def generate_inputs(self):
        h_shape = (self.n_layers, self.batches[0], self.hidden_size)
        h_dtype = self.in_dtypes[0]
        ws_dtype = self.in_dtypes[2]
        bs_dtype = self.in_dtypes[3]

        h = array_utils.uniform(h_shape, h_dtype)
        c = array_utils.uniform(h_shape, h_dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array_utils.uniform((self.batches[b], in_size), ws_dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            return in_size if i == 0 and j < 4 else out_size

        inputs = []
        inputs.append(h)
        inputs.append(c)
        for i in range(len(self.batches)):
            inputs.append(xs[i])

        for n in range(self.n_layers):
            for i in range(8):
                inputs.append(array_utils.uniform(
                    (out_size, w_in(n, i)), ws_dtype))
            for i in range(8):
                inputs.append(array_utils.uniform((out_size,), bs_dtype))
        return tuple(inputs)

    def process_input(self, inputs):
        h = inputs[0]
        c = inputs[1]
        xs = inputs[2:2 + len(self.batches)]
        ws = []
        bs = []
        index = 2 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 8])
            bs.append(inputs[index + 8: index + 16])
            index += 16
        return h, c, ws, bs, xs

    def forward_chainerx(self, inputs):
        h, c, ws, bs, xs = self.process_input(inputs)
        out = chainerx.n_step_lstm(self.n_layers, h, c, ws, bs, xs)
        rets = []
        rets.append(out[0][0])
        rets.append(out[0][1])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_chainer(self, inputs):
        h, c, ws, bs, xs = self.process_input(inputs)
        out = chainer.functions.n_step_lstm(
            self.n_layers, 0.0, h, c, ws, bs, xs)
        rets = []
        rets.append(out[0])
        rets.append(out[1])
        for i in range(len(out[2])):
            rets.append(out[2][i])

        return tuple(rets)