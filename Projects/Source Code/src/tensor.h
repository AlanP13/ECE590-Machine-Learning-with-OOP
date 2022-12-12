#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

typedef double (*scalar_op)(const double &, const double &);

class tensor {
private:
    std::vector<size_t> shape_;
    std::vector<double> data_;

    tensor &compute(
        double d,
        scalar_op op);
    tensor compute(
        const double &d,
        scalar_op op)
        const;
    
    tensor &compute(
        tensor &t,
        scalar_op op);
    tensor compute(
        const tensor &t,
        scalar_op op)
        const;
public:
    tensor(); // scalar 0
    explicit tensor(double v); // scalar v
    tensor(
        size_t dim,
        size_t shape[],
        double data[]); // from C
    tensor copy() const;
    
    size_t get_dim() const;
    size_t get_data_size() const;

    // scalar only
    double &item();
    double item() const;

    double at(size_t i) const;
    double at(size_t i, size_t j) const;
    double at(size_t i, size_t j, size_t k, size_t l) const;
    double get(size_t i) const;
    void set(size_t i, double val);

    tensor &as_row_vector();
    tensor as_row_vector() const;

    tensor &as_column_vector();
    tensor as_column_vector() const;

    tensor row(size_t i) const;
    tensor col(size_t i) const;

    size_t *get_shape_array();
    const size_t *get_shape_array() const;
    void copy_shape_array(double *shape) const;
    double *get_data_array();
    const double *get_data_array() const;
    void copy_data_array(double *data) const;

    tensor &add(double d);
    tensor add(const double &d) const;

    tensor &add(tensor &t);
    tensor add(const tensor &t) const;

    tensor &sub(double d);
    tensor sub(const double &d) const;

    tensor &sub(tensor &t);
    tensor sub(const tensor &t) const;

    tensor &mul(double d);
    tensor mul(const double &d) const;

    tensor &matmul(tensor &t);
    tensor matmul(const tensor &t) const;

    tensor &hadamardmul(tensor &t);
    tensor hadamardmul(const tensor &t) const;

    tensor &mul(tensor &t);
    tensor mul(const tensor &t) const;

    tensor &neg();
    tensor neg() const;

    tensor &transpose();
    tensor transpose() const;

    void print() const;
    void print(bool show_values) const;
}; // class tensor

#endif // TENSOR_H