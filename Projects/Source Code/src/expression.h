#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>

#include "tensor.h"

class evaluation;

class expression
{
private:
    int expr_id_;
    std::string op_name_;
    std::string op_type_;
    std::vector<int> inputs_;
    std::map<std::string, tensor> op_param_;
public:
    expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int *inputs,
        int num_inputs);

    void add_op_param_double(
        const char *key,
        double value);

    void add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);
    
    int get_expr_id() const;
    std::string get_op_name() const;
    std::string get_op_type() const;
    std::vector<int> get_inputs() const;

    double get_op_param_double(const char *key) const;
    tensor get_op_param_ndarray(const char *key) const;
}; // class expression

#endif // EXPRESSION_H
