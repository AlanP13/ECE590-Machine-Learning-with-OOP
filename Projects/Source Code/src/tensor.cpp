#include <assert.h>
#include <stdio.h>
#include <omp.h>

#include "tensor.h"

tensor::tensor()
    : data_(1, 0)
{
}

tensor::tensor(
    double v)
    : data_(1, v)
{
}

tensor::tensor(
    size_t dim,
    size_t shape[],
    double data[])
    : shape_(shape, shape+dim)
{
    size_t N = 1;
    for (size_t i = 0; i < dim; ++i)
        N *= shape[i];

    data_.assign(data, data+N);
}

tensor tensor::copy()
const {
    size_t dim = get_dim();

    if (dim == 0)
        return tensor(item());
    
    size_t shape[dim];
    for (size_t i = 0; i < dim; ++i)
        shape[i] = shape_[i];
    
    double data[data_.size()];
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i)
        data[i] = data_[i];
    
    return tensor(dim, shape, data);
}

size_t tensor::get_dim()
const {
    if (shape_.empty())
        return 0;
    
    return shape_.size();
}

size_t tensor::get_data_size()
const {
    return data_.size();
}

double tensor::item()
const {
    assert(get_dim() == 0);
    return data_[0];
}

double &tensor::item()
{
    assert(get_dim() == 0);
    return data_[0];
}

double tensor::at(
    size_t i)
const {
    assert(get_dim() == 1
        || (get_dim() == 2 && shape_[0] == 1)
        || (get_dim() == 2 && shape_[1] == 1));
    assert(i < data_.size());
    return data_[i];
}

double tensor::at(
    size_t i,
    size_t j)
const {
    assert(get_dim() == 2);
    assert((i < shape_[0]) && (j < shape_[1]));
    return data_[i*shape_[1]+j];
}

double tensor::at(
    size_t n,
    size_t c,
    size_t h,
    size_t w)
const {
    assert(get_dim() == 4);

    size_t N = shape_[0];
    size_t C = shape_[1];
    size_t H = shape_[2];
    size_t W = shape_[3];

    assert((n < N)
        && (c < C)
        && (h < H)
        && (w < W));
    
    return data_[n*C*H*W
        + c*H*W
        + h*W
        + w];
}

double tensor::get(size_t i)
const {
    assert(i < data_.size());
    return data_[i];
}

void tensor::set(size_t i, double val)
{
    assert(i < data_.size());
    data_[i] = val;
}

tensor &tensor::as_row_vector()
{
    assert(get_dim() == 1);

    shape_.push_back(shape_[0]);
    shape_[0] = 1;
    
    return *this;
}

tensor tensor::as_row_vector() const
{
    assert(get_dim() == 1);

    size_t shape[] = {1, shape_[0]};
    double data[data_.size()];
    for (size_t i = 0; i < data_.size(); ++i)
        data[i] = data_[i];
    
    return tensor(2, shape, data);
}

tensor &tensor::as_column_vector()
{
    assert(get_dim() == 1);

    shape_.push_back(1);
    
    return *this;
}

tensor tensor::as_column_vector() const
{
    assert(get_dim() == 1);

    size_t shape[] = {shape_[0], 1};
    double data[data_.size()];
    for (size_t i = 0; i < data_.size(); ++i)
        data[i] = data_[i];
    
    return tensor(2, shape, data);
}

tensor tensor::row(size_t i) const
{
    assert(get_dim() == 2);

    size_t C = shape_[1];
    size_t shape[] = {C};

    double data[C];
    for (size_t j = 0; j < C; ++j)
        data[j] = at(i, j);

    return tensor(1, shape, data);
}

tensor tensor::col(size_t i) const
{
    assert(get_dim() == 2);

    size_t R = shape_[0];
    size_t shape[] = {R};

    double data[R];
    for (size_t j = 0; j < R; ++j)
        data[j] = at(j, i);
    
    return tensor(1, shape, data);
}

size_t *tensor::get_shape_array()
{
    return shape_.empty()? nullptr: &shape_[0];
}

const size_t *tensor::get_shape_array()
const {
    return shape_.empty()? nullptr: &shape_[0];
}

void tensor::copy_shape_array(double *shape)
const {
    for (size_t i = 0; i < shape_.size(); ++i)
        shape[i] = shape_[i];
}

double *tensor::get_data_array()
{
    return &data_[0];
}

const double *tensor::get_data_array()
const {
    return &data_[0];
}

void tensor::copy_data_array(double *data)
const {
    for (size_t i = 0; i < data_.size(); ++i)
        data[i] = data_[i];
}

double scalar_add(
    const double &a,
    const double &b)
{
    return a+b;
}

double scalar_sub(
    const double &a,
    const double &b)
{
    return a-b;
}

double scalar_mul(
    const double &a,
    const double &b)
{
    return a*b;
}

tensor &tensor::compute(
    double d,
    scalar_op op)
{
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] = op(data_[i], d);
    
    return *this;
}

tensor tensor::compute(
    const double &d,
    scalar_op op)
const {
    size_t dim = get_dim();

    size_t shape[dim];
    for (size_t i = 0; i < dim; ++i)
        shape[i] = shape_[i];
    
    double data[data_.size()];
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i)
        data[i] = op(data_[i], d);
    
    return tensor(dim, shape, data);
}

tensor &tensor::compute(
    tensor &t,
    scalar_op op)
{
    size_t dim = get_dim();
    assert(dim == t.get_dim());

    if (dim == 0) {
        item() = op(item(), t.item());
        return *this;
    }
    
    for (size_t i = 0; i < dim; ++i)
        assert(shape_[i] == t.shape_[i]);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] = op(data_[i], t.data_[i]);
    
    return *this;
}

tensor tensor::compute(
    const tensor &t,
    scalar_op op)
const {
    size_t dim = get_dim();
    assert(dim == t.get_dim());

    if (dim == 0)
        return tensor(op(item(), t.item()));
    
    size_t shape[dim];
    for (size_t i = 0; i < dim; ++i) {
        assert(shape_[i] == t.shape_[i]);
        shape[i] = shape_[i];
    }

    double data[data_.size()];
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i)
        data[i] = op(data_[i], t.data_[i]);
    
    return tensor(dim, shape, data);
}

tensor &tensor::add(double d)
{
    return compute(d, scalar_add);
}

tensor tensor::add(const double &d)
const {
    return compute(d, scalar_add);
}

tensor &tensor::add(tensor &t)
{
    return compute(t, scalar_add);
}

tensor tensor::add(const tensor &t)
const {
    return compute(t, scalar_add);
}

tensor &tensor::sub(double d)
{
    return compute(d, scalar_sub);
}

tensor tensor::sub(const double &d)
const {
    return compute(d, scalar_sub);
}

tensor &tensor::sub(tensor &t)
{
    return compute(t, scalar_sub);
}

tensor tensor::sub(const tensor &t)
const {
    return compute(t, scalar_sub);
}

tensor &tensor::mul(double d)
{
    return compute(d, scalar_mul);
}

tensor tensor::mul(const double &d)
const {
    return compute(d, scalar_mul);
}

/* we expect 2 dim tensors to be small enough that the cost
   of separating into multiple threads would be greater
   than the benefits, so no openmp here */
tensor &tensor::matmul(tensor &t)
{
    assert(get_dim() == 2 && t.get_dim() == 2);
    assert(shape_[1] == t.shape_[0]);

    for(size_t i = 0; i < shape_[0]; ++i)
        for(size_t j = 0; j < t.shape_[1]; ++j) {
            data_[i*t.shape_[1]+j] = 0;
            for(size_t k = 0; k < shape_[1]; ++k)
                data_[i*t.shape_[1]+j] += at(i, k) * t.at(k, j);
        }

    shape_[1] = t.shape_[1];
    return *this;
}

tensor tensor::matmul(const tensor &t)
const {
    assert(get_dim() == 2 && t.get_dim() == 2);
    assert(shape_[1] == t.shape_[0]);
    size_t shape[] = {
        shape_[0],
        t.shape_[1]
    };

    double data[shape[0]*shape[1]];
    for(size_t i = 0; i < shape_[0]; ++i)
        for(size_t j = 0; j < t.shape_[1]; ++j) {
            data[i*shape[1]+j] = 0;
            for(size_t k = 0; k < shape_[1]; ++k)
                data[i*shape[1]+j] += at(i, k) * t.at(k, j);
        }

    return tensor(2, shape, data);
}

tensor &tensor::hadamardmul(tensor &t)
{
    return compute(t, scalar_mul);
}

tensor tensor::hadamardmul(const tensor &t)
const {
    return compute(t, scalar_mul);
}

tensor &tensor::mul(tensor &t)
{
    size_t dim_a = get_dim();
    size_t dim_b = t.get_dim();

    if (dim_a == 0 && dim_b == 0) {
        item() = item()*t.item();
        return *this;
    }
    
    if (dim_a == 0)
        return t.mul(item());
    
    if (dim_b == 0)
        return mul(t.item());
    
    if (dim_a == 2 && dim_b == 2
        && shape_[1] == t.shape_[0])
        return matmul(t);
    
    return hadamardmul(t);
}

tensor tensor::mul(const tensor &t)
const {
    size_t dim_a = get_dim();
    size_t dim_b = t.get_dim();

    if (dim_a == 0 && dim_b == 0)
        return tensor(item()*t.item());
    
    if (dim_a == 0)
        return t.mul(item());
    
    if (dim_b == 0)
        return mul(t.item());
    
    if (dim_a == 2 && dim_b == 2
        && shape_[1] == t.shape_[0])
        return matmul(t);
    
    return hadamardmul(t);
}

tensor &tensor::neg()
{
    if (get_dim() == 0) {
        item() = -item();
        return *this;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] = -data_[i];
    
    return *this;
}

tensor tensor::neg()
const {
    size_t dim = get_dim();

    if (dim == 0)
        return tensor(-item());

    size_t shape[dim];
    for (size_t i = 0; i < dim; ++i)
        shape[i] = shape_[i];

    double data[data_.size()];
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i)
        data[i] = -data_[i];
    
    return tensor(dim, shape, data);
}

tensor &tensor::transpose()
{
    assert(get_dim() == 2);

    size_t R = shape_[0];
    size_t C = shape_[1];
    
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < C; ++c)
            data_[c*R+r] = at(r, c);
    
    shape_[0] = C;
    shape_[1] = R;
    return *this;
}

tensor tensor::transpose()
const {
    assert(get_dim() == 2);

    size_t R = shape_[0];
    size_t C = shape_[1];
    size_t shape[] = {C, R};

    double data[data_.size()];
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < C; ++c)
            data[c*R+r] = at(r, c);
    
    return tensor(2, shape, data);
}

/* for debugging only */
void tensor::print()
const {
    print(false);
}

void tensor::print(bool show_values)
const {
    if (get_dim() == 0) {
        printf("Scalar\n");
        if (show_values)
            printf("%f\n\n", item());
    }

    else if (get_dim() == 1) {
        printf("Elements: %lu\n", data_.size());
        if (show_values) {
            printf("[");
            for (size_t i = 0; i < data_.size()-1; ++i)
                printf("%f, ", data_[i]);
            printf("%f]\n\n", data_[data_.size()-1]);
        }
    }

    else if (get_dim() == 2) {
        size_t R = shape_[0];
        size_t C = shape_[1];
        printf("Elements: %lu\n", data_.size());
        printf("Rows: %lu\n", R);
        printf("Columns: %lu\n", C);

        if (show_values) {
            printf("[");
            for (size_t r = 0; r < R-1; ++r) {
                for (size_t c = 0; c < C-1; ++c)
                    printf("%f, ", data_[r*C+c]);
                printf("%f\n", data_[r*C+(C-1)]);
            }
            for (size_t c = 0; c < C-1; ++c)
                printf("%f, ", data_[(R-1)*C+c]);
            printf("%f]\n\n", data_[(R-1)*C+(C-1)]);
        }
    }

    else if (get_dim() == 3) {
        size_t C = shape_[0];
        size_t H = shape_[1];
        size_t W = shape_[2];
        printf("Elements: %lu\n", data_.size());
        printf("Channels: %lu\n", C);
        printf("Height: %lu\n", H);
        printf("Width: %lu\n\n", W);

        if (show_values) {
            for (size_t c = 0; c < C; ++c)
                for (size_t h = 0; h < H; ++h)
                    for (size_t w = 0; w < W; ++w)
                        printf("%lu, %lu, %lu: %f\n",
                            c, h, w,
                            data_[c*H*W+h*W+w]);
            printf("\n");
        }
    }

    else if (get_dim() == 4) {
        size_t N = shape_[0];
        size_t C = shape_[1];
        size_t H = shape_[2];
        size_t W = shape_[3];
        printf("Elements: %lu\n", data_.size());
        printf("Samples (N): %lu\n", N);
        printf("Channels: %lu\n", C);
        printf("Height: %lu\n", H);
        printf("Width: %lu\n\n", W);

        if (show_values) {
            for (size_t n = 0; n < N; ++n)
                for (size_t c = 0; c < C; ++c)
                    for (size_t h = 0; h < H; ++h)
                        for (size_t w = 0; w < W; ++w)
                            printf("%lu, %lu, %lu, %lu: %f\n",
                                n, c, h, w,
                                at(n, c, h, w));
            printf("\n");
        }
    }

    else
        ; // to be implemented
}