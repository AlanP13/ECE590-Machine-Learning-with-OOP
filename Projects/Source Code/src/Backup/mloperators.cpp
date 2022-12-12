#include <assert.h>
#include <stdio.h>

#include "mloperators.h"

eval_op_prototypes::eval_op_prototypes()
{
    eval_const::store_prototype(proto_map_);
    eval_input::store_prototype(proto_map_);

    eval_add::store_prototype(proto_map_);
    eval_sub::store_prototype(proto_map_);
    eval_mul::store_prototype(proto_map_);
    eval_neg::store_prototype(proto_map_);

    eval_relu::store_prototype(proto_map_);
    eval_flatten::store_prototype(proto_map_);
    eval_input2d::store_prototype(proto_map_);
    eval_linear::store_prototype(proto_map_);
    eval_maxpool2d::store_prototype(proto_map_);
    eval_conv2d::store_prototype(proto_map_);
}

void eval_relu::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("ReLU") == proto_map.end());
    proto_map["ReLU"] = std::make_shared<eval_relu>();
}

void eval_flatten::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Flatten") == proto_map.end());
    proto_map["Flatten"] = std::make_shared<eval_flatten>();
}

void eval_input2d::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Input2d") == proto_map.end());
    proto_map["Input2d"] = std::make_shared<eval_input2d>();
}

void eval_linear::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Linear") == proto_map.end());
    proto_map["Linear"] = std::make_shared<eval_linear>();
}

void eval_maxpool2d::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("MaxPool2d") == proto_map.end());
    proto_map["MaxPool2d"] = std::make_shared<eval_maxpool2d>();
}

void eval_conv2d::store_prototype(
    eval_op_proto_map &proto_map)
{
    assert(proto_map.find("Conv2d") == proto_map.end());
    proto_map["Conv2d"] = std::make_shared<eval_conv2d>();
}

std::shared_ptr<eval_op> eval_relu::clone(
    const expression &expr)
{
    return std::make_shared<eval_relu>(expr);
}

std::shared_ptr<eval_op> eval_flatten::clone(
    const expression &expr)
{
    return std::make_shared<eval_flatten>(expr);
}

std::shared_ptr<eval_op> eval_input2d::clone(
    const expression &expr)
{
    return std::make_shared<eval_input2d>(expr);
}

std::shared_ptr<eval_op> eval_linear::clone(
    const expression &expr)
{
    return std::make_shared<eval_linear>(expr);
}

std::shared_ptr<eval_op> eval_maxpool2d::clone(
    const expression &expr)
{
    return std::make_shared<eval_maxpool2d>(expr);
}

std::shared_ptr<eval_op> eval_conv2d::clone(
    const expression &expr)
{
    return std::make_shared<eval_conv2d>(expr);
}

eval_ml::eval_ml()
{
}

eval_ml::~eval_ml()
{
}

eval_relu::eval_relu()
{
}

eval_flatten::eval_flatten()
{
}

eval_input2d::eval_input2d()
    : height_(),
    width_(),
    in_channels_()
{
}

eval_linear::eval_linear()
{
}

eval_maxpool2d::eval_maxpool2d()
    : kernel_size_(),
    stride_()
{
}

eval_conv2d::eval_conv2d()
    : in_channels_(),
    out_channels_(),
    kernel_size_()
{
}

eval_ml::eval_ml(const expression &expr)
    : eval_unary(expr)
{
}

eval_relu::eval_relu(const expression &expr)
    : eval_ml(expr)
{
}

eval_flatten::eval_flatten(const expression &expr)
    : eval_ml(expr)
{
}

eval_input2d::eval_input2d(const expression &expr)
    : eval_op(expr),
    height_(expr.get_op_param_double("height")),
    width_(expr.get_op_param_double("width")),
    in_channels_(expr.get_op_param_double("in_channels"))
{
}

eval_linear::eval_linear(const expression &expr)
    : eval_unary(expr),
    weight_(expr.get_op_param_ndarray("weight")),
    bias_(expr.get_op_param_ndarray("bias"))
{
}

eval_maxpool2d::eval_maxpool2d(const expression &expr)
    : eval_unary(expr),
    kernel_size_(expr.get_op_param_double("kernel_size")),
    stride_(expr.get_op_param_double("stride"))
{
}

eval_conv2d::eval_conv2d(const expression &expr)
    : eval_unary(expr),
    in_channels_(expr.get_op_param_double("in_channels")),
    out_channels_(expr.get_op_param_double("out_channels")),
    kernel_size_(expr.get_op_param_double("kernel_size")),
    weight_(expr.get_op_param_ndarray("weight")),
    bias_(expr.get_op_param_ndarray("bias"))
{
}

tensor eval_ml::compute(const tensor &t)
{
    size_t dim = t.get_dim();
    const size_t *s = t.get_shape_array();

    size_t d_size = t.get_data_size();
    const double *d = t.get_data_array();

    return compute(dim, d_size, s, d);
}

tensor eval_relu::compute(
    size_t dim,
    size_t d_size,
    const size_t *s,
    const double *d)
{
    if (dim == 0)
        return d[0] > 0? tensor(d[0]): tensor(0.0);

    size_t shape[dim];
    for (size_t i = 0; i < dim; ++i)
        shape[i] = s[i];

    double data[d_size];
    for (size_t i = 0; i < d_size; ++i)
        data[i] = d[i] > 0? d[i]: 0;

    return tensor(dim, shape, data);
}

tensor eval_flatten::compute(
    size_t dim,
    size_t d_size,
    const size_t *s,
    const double *d)
{
    assert(dim >= 2);

    size_t shape[] = {s[0], 1};
    for (size_t i = 1; i < dim; ++i)
        shape[1] *= s[i];

    double data[d_size];
    for (size_t i = 0; i < d_size; ++i)
        data[i] = d[i];

    return tensor(2, shape, data);
}

void eval_input2d::eval(
    vars_type &variables,
    const kwargs_type &kwargs)
{
    auto iter = kwargs.find(op_name_);
    assert(iter != kwargs.end());
    tensor input = iter->second;

    size_t dim = input.get_dim();
    size_t d_size = input.get_data_size();

    const size_t *s = input.get_shape_array();
    const double *d = input.get_data_array();

    assert(dim == 4);
    assert(s[1] == height_);
    assert(s[2] == width_);
    assert(s[3] == in_channels_);

    size_t shape[] = {s[0], s[3], s[1], s[2]};

    size_t HWC = height_*width_*in_channels_;
    size_t HW = height_*width_;
    size_t WC = width_*in_channels_;
    size_t W = width_;
    size_t C = in_channels_;

    double data[d_size];
    for (size_t n = 0; n < s[0]; ++n)
        for (size_t h = 0; h < height_; ++h)
            for (size_t w = 0; w < width_; ++w)
                for (size_t c = 0; c < in_channels_; ++c)
                    data[n*HWC+c*HW+h*W+w] = d[n*HWC+h*WC+w*C+c];

    variables[expr_id_] = tensor(4, shape, data);
}

tensor eval_linear::compute(const tensor &t)
{
    assert(t.get_dim() == 2);
    assert(weight_.get_dim() == 2);
    assert(bias_.get_dim() == 1);
    tensor b = bias_.as_column_vector();

    const size_t *s = t.get_shape_array();
    size_t N = s[0];
    size_t I = s[1];
    size_t O = weight_.get_shape_array()[0];

    assert(weight_.get_shape_array()[1] == I);
    assert(bias_.get_shape_array()[0] == O);

    size_t shape[] = {N, O};

    double data[N*O];
    for (size_t i = 0; i < N; ++i) {
        tensor x = t.row(i).as_row_vector();

        weight_
        .mul(x.transpose())
        .add(b)
        .transpose()
        .copy_data_array(&data[i*O]);
    }

    return tensor(2, shape, data);
}

tensor eval_maxpool2d::compute(const tensor &t)
{
    assert(t.get_dim() == 4);

    const size_t *s = t.get_shape_array();
    size_t N = s[0];
    size_t C = s[1];
    size_t H = s[2];
    size_t W = s[3];

    size_t new_H = (H-kernel_size_)/stride_+1;
    size_t new_W = (W-kernel_size_)/stride_+1;

    size_t shape[] = {
        N, C, new_H, new_W
    };

    double data[N*C*new_H*new_W];
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t h = 0; h+kernel_size_ <= H; h += stride_)
                for (size_t w = 0; w+kernel_size_ <= W; w += stride_) {

                    double max = t.at(n, c, h, w);
                    for (int i = 0; i < kernel_size_; ++i)
                        for (int j = 0; j < kernel_size_; ++j)
                            if (t.at(n, c, h+i, w+j) > max)
                                max = t.at(n, c, h+i, w+j);
                    
                    size_t new_h = h/stride_;
                    size_t new_w = w/stride_;

                    data[n*C*new_H*new_W
                        + c*new_H*new_W
                        + new_h*new_W
                        + new_w] = max;
                }

    return tensor(4, shape, data);
}

/** Things to note for David:
 *  Tensors are constructed with tensor(# of dimensions, shape array, data array)
 *  The shape array contains the size for each dimension
 *  So if we have a tensor of dimension 2 (matrix),
 *      shape[0] = R (number of rows)
 *      shape[1] = C (number of columns)
 *  For a tensor of dimension 4,
 *      shape[0] = N (number of images)
 *      shape[1] = C (number of channels per image - usually 3 for red, green, and blue)
 *      shape[2] = H (height of image)
 *      shape[3] = W (width of image)
 * 
 *  All element values are stored in the data array in row-major order
 *  This means any element (n, c, h, w) can be accessed at index n*C*H*W + c*H*W + h*W + w
 *  where capital N is the number of ns, capital C is the number of cs, etc.
 *  In this case, C = in_channels for the input tensor and out_channels for the output tensor
 * 
 *  Lines of code I used to debug are commented out
 *  freopen is used to redirect stdout to a file. When debugging, I do this before printing out tensors
 *  so I can more easily browse through them in a file editor
 *  Use tensor.print() to print tensor info and tensor.print(true) to print info and data
 *  Tensors of dimension 1 are printed out as arrays
 *  Tensors of dimension 2 are printed out as matrices
 *  Tensors of dimensions 3 and 4 are printed out one element at a time in the form <index>: <value>
 *  
 *  Here are the important format characters for printf
 *      %d => int
 *      %f => double
 *      %lu => size_t (long unsigned int)
 *  Even though in_channels, out_channels, and kernel_size are read in as doubles,
 *  they are expected to be ints
 */
tensor eval_conv2d::compute(const tensor &t)
{
    assert(t.get_dim() == 4);
    assert(weight_.get_dim() == 4);
    assert(bias_.get_dim() == 1);

    const size_t *s = t.get_shape_array();
    size_t N = s[0];
    assert(s[1] == in_channels_);
    size_t H = s[2];
    size_t W = s[3];

    const size_t *w_shape = weight_.get_shape_array();
    assert(w_shape[0] == out_channels_);
    assert(w_shape[1] == in_channels_);
    assert(w_shape[2] == kernel_size_);
    assert(w_shape[3] == kernel_size_);

    assert(bias_.get_shape_array()[0] == out_channels_);

    size_t new_H = H-kernel_size_+1;
    size_t new_W = W-kernel_size_+1;
    // shape of result tensor
    size_t shape[] = {
        N,
        (size_t)out_channels_,
        new_H,
        new_W
    };

    size_t batch_shape[] = {
        (size_t)in_channels_,
        (size_t)kernel_size_,
        (size_t)kernel_size_
    };
    size_t batch_size = in_channels_*kernel_size_*kernel_size_;
    double input_data[batch_size];
    double weight_data[batch_size];

    /*
    printf("\nN: %lu", N);
    printf("\nH: %lu", H);
    printf("\nW: %lu", W);
    printf("\nIn channels: %d", (int)in_channels_);
    printf("\nOut channels: %d", (int)out_channels_);
    printf("\nKernel size: %d", (int)kernel_size_);
    printf("\nnew_H: %lu", new_H);
    printf("\nnew_W: %lu", new_W);
    printf("\nBatch size: %lu\n", batch_size);
    */

    /*
    freopen("input", "w", stdout);
    t.print(true);
    freopen("weights", "w", stdout);
    weight_.print(true);
    freopen("bias", "w", stdout);
    bias_.print(true);
    */
    
    // data of result tensor
    double data[N
        *(size_t)out_channels_
        *new_H
        *new_W];
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h+kernel_size_ <= H; h += kernel_size_)
            for (size_t w = 0; w+kernel_size_ <= W; w += kernel_size_) {

                for (size_t c = 0; c < in_channels_; ++c)
                    for (int i = 0; i < kernel_size_; ++i)
                        for (int j = 0; j < kernel_size_; ++j) {
                            size_t index = c*kernel_size_*kernel_size_
                                + i*kernel_size_
                                + j;
                            input_data[index] = t.at(n, c, h+i, w+j);

                            /*
                            printf("\nInput data index: %lu", index);
                            printf("\nFrom: %lu, %lu, %lu, %lu\n", n, c, h+i, w+j);
                            */
                        }
                const tensor input_batch = tensor(3, batch_shape, input_data);

                /*
                freopen("ib", "w", stdout);
                input_batch.print(true);
                */

                for (size_t c = 0; c < out_channels_; ++c) {
                    // printf("\nc = %lu\n", c);
                    for (size_t i = 0; i < batch_size; ++i) {
                        weight_data[i] = weight_.get(c*batch_size+i);
                        /*
                        printf("\nWeight data index: %lu", i);
                        printf("\nFrom: %lu\n", c*batch_size+i);
                        */
                    }

                    const tensor weight_batch = tensor(3, batch_shape, weight_data);
                    /*
                    std::string c_st = "wb"+std::to_string(c);
                    freopen(c_st.c_str(), "w", stdout);
                    weight_batch.print(true);
                    */

                    // I checked the following line of code, and it seems to work
                    // so I doubt there's anything wrong with my tensor implementation
                    tensor output_batch = input_batch
                        .hadamardmul(weight_batch)  // multiply each element in input_bath with the same element in weight_batch and return result
                        .add(bias_.get(c)); // get the cth value of bias vector
                    /*
                    c_st = "ob"+std::to_string(c);
                    freopen(c_st.c_str(), "w", stdout);
                    output_batch.print(true);
                    */
                    
                    size_t index = n*out_channels_*new_H*new_W
                        + c*new_H*new_W
                        + h*new_W
                        + w;
                    output_batch.copy_data_array(&data[index]);

                    // printf("Output index: %lu\n", index);
                }
                // exit(0);
        }
    }

    /*
    freopen("final", "w", stdout);
    tensor(4, shape, data).print(true);
    */
    return tensor(4, shape, data);
}