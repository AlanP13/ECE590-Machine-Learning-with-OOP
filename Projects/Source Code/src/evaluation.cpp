#include <assert.h>

#include "evaluation.h"

evaluation::evaluation(const std::vector<expression> &exprs)
{
    result_ = -1;
    for (auto &expr: exprs) {
        std::shared_ptr<eval_op> p
            = eval_op_prototypes::instance().locate(expr.get_op_type());
        ops_.push_back(p->clone(expr));
        
        if (expr.get_expr_id() > result_)
            result_ = expr.get_expr_id();
    }
}

void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    kwargs_[key] = tensor(value);
}

void evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    kwargs_[key] = tensor(dim, shape, data);
}

int evaluation::execute()
{
    variables_.clear();
    for (auto &op: ops_)
        op->eval(variables_, kwargs_);
    
    return 0;
}

tensor &evaluation::get_result()
{
    assert(result_ >= 0);
    return variables_[result_];
}