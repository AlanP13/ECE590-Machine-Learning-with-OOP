#include <string.h>

#include "expression.h"

expression::expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int *inputs,
    int num_inputs)
    : expr_id_(expr_id), op_name_(op_name), op_type_(op_type), inputs_(inputs, inputs+num_inputs)
{
}

void expression::add_op_param_double(
    const char *key,
    double value)
{
    op_param_[key] = tensor(value);
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    op_param_[key] = tensor(dim, shape, data);
}

int expression::get_expr_id() const
{
    return expr_id_;
}

std::string expression::get_op_name() const
{
    return op_name_;
}

std::string expression::get_op_type() const
{
    return op_type_;
}

std::vector<int> expression::get_inputs() const
{
    return inputs_;
}

double expression::get_op_param_double (const char *key) const
{
    return get_op_param_ndarray(key).item();
}

tensor expression::get_op_param_ndarray(const char *key) const
{
    auto iter = op_param_.find(key);
    if (iter != op_param_.end()) // optional parameter provided
        return iter->second;
    else                         // optional parameter missing
        return tensor();         // return 0 as tensor
}